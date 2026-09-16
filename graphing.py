import tomllib

import numpy as np
import matplotlib.pyplot as plt

def delta_function(peaks):
    """Takes (weight, position) pairs and returns a sum of Gaussians and the peaks."""
    func = lambda x: sum(weight * np.exp(-((x - pos) ** 2) / 10) for weight, pos in peaks)
    return func

def find_visibility(peaks, path_delay, intensity):
    """Calculate visibility V = (I_max - I_min) / (I_max + I_min) for each peak region."""
    visibility_texts = []

    for _, pos in peaks:
        mask = np.abs(path_delay - pos) < 50
        if not np.any(mask):
            continue

        region_intensity = intensity[mask]
        I_max = np.max(region_intensity)
        I_min = np.min(region_intensity)

        if I_max + I_min > 0:
            visibility = (I_max - I_min) / (I_max + I_min)
            visibility_texts.append(f"Visibility at z={pos}: {visibility:.3f}")

    return visibility_texts

def find_fwhm(baseline, peaks, path_delay, intensity, *, dips=False):
    fwhm_texts = [] # infobox contents
    fwhm_lines = [] # list of (half_max, left_x, right_x) for each peak

    for _, pos in peaks:
        mask = np.abs(path_delay - pos) < 50 # Boolean array telling whether each point is near a peak
        if not np.any(mask):
            continue

        # grab x and y values at these points
        region_intensity = intensity[mask]
        region_path = path_delay[mask]

        if dips:
            extremum = np.min(region_intensity)
            half_max = baseline - (baseline - extremum) / 2
            below_half = region_intensity <= half_max
        else:
            extremum = np.max(region_intensity)
            half_max = baseline + (extremum - baseline) / 2
            below_half = region_intensity >= half_max

        crossings = np.where(np.diff(below_half.astype(int)))[0]

        if len(crossings) >= 2:
            left_idx = crossings[0]
            right_idx = crossings[-1]
            fwhm = region_path[right_idx] - region_path[left_idx]
            fwhm_texts.append(f"FWHM at z={pos}: {fwhm:.2f} μm")
            fwhm_lines.append((half_max, region_path[left_idx], region_path[right_idx]))

    return fwhm_lines, fwhm_texts

def plot_interferogram(cfg, peaks, time_limit, y_limit):
    y_axis: str = "intensity"

    # TODO: write validator
    oct_type: str = cfg['general']['oct_type']
    grover: bool = cfg['general']['grover']

    sample_length: float = cfg['sample']['length'] # meters, or None if omitted
    refractive_index: float = cfg['sample']['refractive_index']

    central_wavelength: float = cfg['laser']['central_wavelength'] # micrometers
    spectral_width: float = cfg['laser']['spectral_width'] # extend about 100nm for broadband light source

    # facts
    speed_of_light = 299792458 # meters / second
    central_frequency = 2 * pi * speed_of_light / (central_wavelength * 1e-6)  # radians / second
    pump_frequency = 2 * central_frequency # radians / second
    wavenumber = 2 * pi / central_wavelength # radians / micrometers
    time_shift = np.linspace(0, time_limit, 10000) # picoseconds
    path_delay = speed_of_light * time_shift * 1e-6 # micrometers

    plt.figure(figsize=(14, 6))

    match oct_type, grover:
        case 'monochromatic', False:
            intensity = monochromatic_standard(peaks, path_delay, wavenumber)

    match oct_type:
        case 'monochromatic standard':
            sample_func = delta_function(peaks)
            sample = sample_func(path_delay)
            monochromatic_intensity = 1/4 * (1 + np.abs(sample) ** 2 + 2 * sample * np.cos(wavenumber * path_delay))
            plt.plot(path_delay, monochromatic_intensity, label='Intensity')
            fwhm_lines, fwhm_texts = find_fwhm(0.25, peaks, path_delay, monochromatic_intensity)
            visibility_texts = find_visibility(peaks, path_delay, monochromatic_intensity)
        case 'nonmonochromatic standard':
            1
        case 'quantum standard':
            spectral_power_distribution = lambda w: np.exp(-(w ** 2) / (2 * spectral_width ** 2)) # function of frequency (radians / second)
            # For Gaussian s(w) = exp(-w^2/(2*sigma^2)), FT gives gamma(tau) = sigma*sqrt(2*pi)*exp(-tau^2*sigma^2/2)
            _coherence_raw = lambda tau: spectral_width * np.sqrt(2 * pi) * np.exp(-(tau ** 2) * (spectral_width ** 2) / 2)
            _coherence_at_zero = _coherence_raw(0.0) # in order to normalize the coherence function
            coherence_function = lambda tau: _coherence_raw(tau) / _coherence_at_zero

            y_axis = "norm. QOCT C(τ_q)"

            (r1, z1), (r2, z2) = peaks

            # graph this for reference, but not actually used in calculations
            sample_func = lambda frequency: r1 + r2 * np.exp(1j * 2 * frequency * refractive_index * sample_length / speed_of_light)
            #TODO: fix this
            sample = sample_func(path_delay)

            tau_q = path_delay * 1e-6 / speed_of_light  # seconds
            tau_z1 = z1 * 1e-6 / speed_of_light # seconds (delay until reaching first reflectance)
            tau_z2 = z2 * 1e-6 / speed_of_light # seconds (delay until reaching second reflectance)
            tau_d = tau_z2 - tau_z1

            # A0 and A(tau_q) use the same normalized coherence function (gamma(0)=1)
            background_term = np.abs(r1)**2 + np.abs(r2)**2

            # Interference term: dips at front (z1) and back (z2) interfaces
            # A(tau-q) = |r1|^2 * s(tau-q) + |r2|^2 * s(tau-q - 2 * tau-d) + 2Re(r1 * r2' * s(tau-q - tau-d) * e^(i * w-p * n * L / c))
            # tau_d = tau_z2 - tau_z1

            # also, we subtract a tau_z1 from each argument for the rightward shift
            term1 = np.abs(r1)**2 * coherence_function(tau_q - tau_z1)
            term2 = np.abs(r2)**2 * coherence_function(tau_q - 2 * tau_d - tau_z1)
            cross_term_phase = pump_frequency * refractive_index * sample_length / speed_of_light
            cross_term = 2 * np.real(r1 * np.conj(r2) * coherence_function(tau_q - tau_d - tau_z1) * np.exp(1j * cross_term_phase))
            interference_term = term1 + term2 + cross_term

            # R = A0 - A(tau_q), normalized so baseline = 1
            coincidence_rate = (background_term - interference_term) / background_term

            '''fix this
            # create gaussian to represent sample
            sample = sum(
                weight * np.exp(-((path_delay - pos) ** 2) / 10)
                for weight, pos in peaks
            )
            '''

            plt.plot(path_delay, coincidence_rate, label='Coincidence Rate')
            fwhm_lines, fwhm_texts = find_fwhm(1.0, peaks, path_delay, coincidence_rate, dips=True)
            visibility_texts = find_visibility(peaks, path_delay, coincidence_rate)

        case 'monochromatic', True:
            # the sample needs to take both arguments of z and w
            # how do I properly handle the nonmonochromatic configuration?
            frequency = speed_of_light / experimental_parameters["central_wavelength"]
            sample_func = delta_function(peaks)
            sample = sample_func(path_delay)

            tau = path_delay * 1e-6 / speed_of_light

            reference_arm = np.exp(1j * frequency * tau)
            exiting_state = (1 + reference_arm - sample) * (1 - reference_arm + sample) / (1 + reference_arm + sample)

            grover_monochromatic_intensity = np.abs(exiting_state) ** 2
            plt.plot(path_delay, grover_monochromatic_intensity, label='Intensity')
            fwhm_lines, fwhm_texts = find_fwhm(0.25, peaks, path_delay, grover_monochromatic_intensity)
            visibility_texts = find_visibility(peaks, path_delay, grover_monochromatic_intensity)
        case 'nonmonochromatic', True:
            pass

        case 'quantum', True:
            pass

    plt.plot(path_delay, sample, label='Sample')

    # plot fwhm lines
    for half_max, left_x, right_x in fwhm_lines:
        plt.hlines(half_max, left_x, right_x, colors='red', linestyles='-', linewidth=2)

    # display infobox with FWHM and visibility
    infobox_texts = fwhm_texts + visibility_texts
    if infobox_texts:
        plt.text(0.02, 0.98, '\n'.join(infobox_texts), transform=plt.gca().transAxes,
                verticalalignment='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # generate latex for the delta function
    terms = []
    for weight, pos in peaks:
        if weight == 1:
            terms.append(rf"\delta(z - {pos})")
        else:
            terms.append(rf"{weight}\delta(z - {pos})")
    label = " + ".join(terms)

    #TODO: fix this title for quantum
    plt.title(f'Graph of {y_axis} vs. path delay with sample $r(z) = {label}$')
    plt.xlabel(r'Path delay $c \tau$ ($\mu$m)')
    plt.ylabel(f'{y_axis}')
    plt.xlim(0, max(path_delay))
    plt.ylim(0, y_limit)
    plt.legend()

    plt.show()

#peaks = ((1, 180), (1.4, 450))

with open("config.toml", "rb") as f:
    cfg = tomllib.load(f)

peaks = new_peaks(0.2, 0.2, 180.0, cfg['sample']['refractive_index'], cfg['sample']['length'])

plot_interferogram(cfg, peaks, cfg['general']['time_limit'], cfg['general']['y_limit'])
