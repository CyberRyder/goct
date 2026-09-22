from math import pi

import matplotlib.pyplot as plt
import numpy as np
import tomllib

from custom import IndependentVariable, DependentVariable
from models import monochromatic_grover, monochromatic_standard, quantum_standard
from peaks import delta_function, new_peaks, Peak


def find_visibility(peaks: list[Peak], path_delay: IndependentVariable, intensity: DependentVariable):
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

def find_fwhm(baseline: float, peaks: list[Peak], path_delay: IndependentVariable, intensity: DependentVariable, *, dips=False):
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

def plot_interferogram(cfg, peaks: list[Peak], time_limit: float, y_limit: float):
    y_axis: str = "intensity"

    # TODO: write validator
    oct_type: str = cfg['general']['oct_type']
    grover: bool = cfg['general']['grover']

    sample_length: float = cfg['sample']['length'] # meters, or None if omitted
    refractive_index: float = cfg['sample']['refractive_index']

    central_wavelength: float = cfg['laser']['central_wavelength'] # micrometers
    spectral_width: float = cfg['laser']['spectral_width'] # extend about 100nm for broadband light source

    # facts
    speed_of_light: int = 299792458 # meters / second
    central_frequency: float = 2 * pi * speed_of_light / (central_wavelength * 1e-6)  # radians / second
    pump_frequency: float = 2 * central_frequency # radians / second
    wavenumber: float = 2 * pi / central_wavelength # radians / micrometers
    time_shift: IndependentVariable = np.linspace(0, time_limit, 10000) # picoseconds
    path_delay: IndependentVariable = speed_of_light * time_shift * 1e-6 # micrometers

    plt.figure(figsize=(14, 6))

    match oct_type, grover:
        case 'monochromatic', False:
            intensity = monochromatic_standard(peaks, path_delay, wavenumber)

            plt.plot(path_delay, intensity, label='Intensity')
            fwhm_lines, fwhm_texts = find_fwhm(0.25, peaks, path_delay, intensity)
            visibility_texts = find_visibility(peaks, path_delay, intensity)
        case 'nonmonochromatic', False:
            pass

        case 'quantum', False:
            coincidence_rate = quantum_standard(peaks, path_delay, spectral_width, speed_of_light, pump_frequency, refractive_index, sample_length)
            y_axis = "norm. QOCT C(τ_q)"

            plt.plot(path_delay, coincidence_rate, label='Coincidence Rate')
            fwhm_lines, fwhm_texts = find_fwhm(1.0, peaks, path_delay, coincidence_rate, dips=True)
            visibility_texts = find_visibility(peaks, path_delay, coincidence_rate)

        case 'monochromatic', True:
            intensity = monochromatic_grover(peaks, path_delay, speed_of_light, central_wavelength)

            plt.plot(path_delay, intensity, label='Intensity')
            fwhm_lines, fwhm_texts = find_fwhm(0.25, peaks, path_delay, intensity)
            visibility_texts = find_visibility(peaks, path_delay, intensity)
        case 'nonmonochromatic', True:
            pass

        case 'quantum', True:
            pass

    sample_func = delta_function(peaks)
    sample = sample_func(path_delay)
    plt.plot(path_delay, sample, label='Sample')

    # TODO: fix this
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
