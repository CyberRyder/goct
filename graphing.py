'''
This program graphs intensity at detector versus the length of the reference arm.

Monochromatic case:
I = 1/4 * I0 * (1 + |r(z)|^2 + 2 * r(z) * cos(2 * k * z - w * tau))

where
I is the intensity at the detector
I0 is the initial intensity of the light
r(z) is the reflectance of the sample
z is the depth into the sample (equate to delta l = c * tau)
c is the speed of light
l is the length of the reference arm
w is the frequency of the light
tau is the time shift of the reference arm


Nonmonochromatic case:
I = 1/4 * (L0 + L1 * 2 * Re(e^(-i * w0 * tau)))

where
I is the intensity at the detector
L0 is the integral of s(w) * (1 + |H(w)|^2) * dw, which is the background term
L1(tau) is the integral of s(w) * 2 * Re(H(w) * e^(-i * w * tau)) * dw, which is the interference term
s(w) is the spectral power density (a known function that depends on the light used)
H(w) is the integral of e^(2 * i * phi(z, w)) * r(z, w) * dz, which is the sample

r(z, w) is the reflectance of the sample
phi(z, w) is the phase shift of the sample
z is the depth into the sample (equate to delta l = c * tau)
c is the speed of light
l is the length of the reference arm
w is the frequency distribution, centered around w0
tau is the time shift of the reference arm


Quantum case:
R = A0 - Re(A1(2 * tau))

where
R is the coincidence rate at the detectors
A0 is the integral of 2 * s(w) * |H(w)|^2 * dw, which is the background term
A1(tau) is the integral of 2 * s(w) * e^(i * w * tau) * H(w) * H'(-w) * dw, which is the interference term
s(w) is the spectral power density (a known function that depends on the light used)
H(w) is the integral of e^(2 * i * phi(z, w)) * r(z, w) * dz, which is the sample
H'(w) is the complex conjugate of H(w)

r(z, w) is the reflectance of the sample
phi(z, w) is the phase shift of the sample
z is the depth into the sample (equate to delta l = c * tau)
c is the speed of light
l is the length of the reference arm
w is the frequency distribution, centered around w0
tau is the time shift of the reference arm



For the nonquantum cases, the sample r(w) is modeled by gaussians representing the reflection points.
For the quantum cases, the sample H(w) is modeled also by a sum of reflectances:

H(w) = r1 + r2 * e^(i * 2 * w * n * L / c)
with reflectance points r1 and r2
where
n is the refractive index
L is the sample thickness
c is the speed of light

From this definition we obtain that

A0 = |r1|^2 + |r2|^2
A(tau-q) = |r1|^2 * s(tau-q) + |r2|^2 * s(tau-q - 2 * tau-d) + 2Re(r1 * r2' * s(tau-q - tau-d) * e^(i * w-p * n * L / c))

where
tau-q is the time delay of the interferometer
tau-d = 2 * n * L / c is the time delay from passing through the sample
w-p = 2 * w0 is the frequency of the original photon before the spectral down conversion
r1 and r2 are the reflectance points
n is the refractive index
L is the sample thickness
c is the speed of light
'''

from math import pi
import numpy as np
import matplotlib.pyplot as plt

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

def plot_interferogram(oct_type, sample_func, peaks, time_limit, y_limit):
    y_axis = "intensity"
    
    # experimental parameters
    wavelength = 0.800 # micrometers
    sample_length = 90 * 1e-6 # meters
    refractive_index = 1.5
    spectral_width = 3.8 * 1e13  # extend about 100nm for broadband light source

    # facts
    speed_of_light = 299792458 # meters / second
    center_frequency = 2 * pi * speed_of_light / (wavelength * 1e-6)  # radians / second
    pump_frequency = 2 * center_frequency # radians / second
    wavenumber = 2 * pi / wavelength # radians / micrometers
    time_shift = np.linspace(0, time_limit, 10000) # picoseconds
    path_delay = speed_of_light * time_shift * 1e-6 # micrometers

    spectral_power_distribution = lambda w: np.exp(-(w ** 2) / (2 * spectral_width ** 2)) # function of frequency (radians / second)
    # For Gaussian s(w) = exp(-w^2/(2*sigma^2)), FT gives gamma(tau) = sigma*sqrt(2*pi)*exp(-tau^2*sigma^2/2)
    _coherence_raw = lambda tau: spectral_width * np.sqrt(2 * pi) * np.exp(-(tau ** 2) * (spectral_width ** 2) / 2)
    _coherence_at_zero = _coherence_raw(0.0) # in order to normalize the coherence function
    coherence_function = lambda tau: _coherence_raw(tau) / _coherence_at_zero

    # much like how we define r(w) in prior cases, we go right to defining H(w), so we don't mess with the integral
    sample = sample_func(path_delay) # function of micrometers

    plt.figure(figsize=(14, 6))


    match oct_type:
        case 'monochromatic':
            monochromatic_intensity = 1/4 * (1 + np.abs(sample) ** 2 + 2 * sample * np.cos(wavenumber * path_delay))
            plt.plot(path_delay, monochromatic_intensity, label='Intensity')
            fwhm_lines, fwhm_texts = find_fwhm(0.25, peaks, path_delay, monochromatic_intensity)
            visibility_texts = find_visibility(peaks, path_delay, monochromatic_intensity)
        case 'nonmonochromatic':
            1
        case 'quantum':
            y_axis = "norm. QOCT C(τ_q)"
            
            (r1, z1), (r2, z2) = peaks

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
            phase = pump_frequency * refractive_index * sample_length / speed_of_light
            cross_term = 2 * np.real(r1 * np.conj(r2) * coherence_function(tau_q - tau_d - tau_z1) * np.exp(1j * phase))
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
            
        case 'grover-michelson':
            1

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

def delta_function(*peaks):
    """Takes (weight, position) pairs and returns a sum of Gaussians and the peaks."""
    func = lambda x: sum(weight * np.exp(-((x - pos) ** 2) / 10) for weight, pos in peaks)
    return func, peaks

#TODO: clean up delta_function
#sample, peaks = delta_function((1, 180), (1.4, 450))
#plot_interferogram('monochromatic', sample, peaks, 2, 2)

# Quantum case: H(w) = r1 + r2 * e^(i * 2 * w * n * L / c)
r1 = 0.2  # |r1|^2 = 0.04
r2 = 0.2  # |r2|^2 = 0.04
n = 1.5   # refractive index
L = 90 * 1e-6   # sample thickness (meters)
c = 299792458 # speed of light, meters/second

# H(w) = r1 + r2 * e^(i * 2 * w * n * L / c); reflections happen at z1 and z2
z1 = 180.0 # micrometers
z2 = z1 + n * L * 1e6 # micrometers
sample = lambda w: r1 + r2 * np.exp(1j * 2 * w * n * L / c)
peaks = [(r1, z1), (r2, z2)] # (reflectance, location)

plot_interferogram('quantum', sample, peaks, 2, 2.5)