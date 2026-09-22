from math import pi

import matplotlib.pyplot as plt
import numpy as np
import tomllib

from analysis import find_fwhm, find_visibility, scale_interferogram
from custom import FWHM, IndependentVariable
from models import (
    monochromatic_grover,
    monochromatic_standard,
    quantum_grover,
    quantum_standard,
)
from peaks import Peak, build_sample, new_peaks


def plot_interferogram(cfg, peaks: list[Peak], time_limit: float, y_limit: float):
    y_axis: str = "intensity"

    # TODO: write validator
    oct_type: str = cfg['graphing']['oct_type']
    grover: bool = cfg['graphing']['grover']

    sample_length: float = cfg['sample']['length'] # meters, or None if omitted
    refractive_index: float = cfg['sample']['refractive_index']
    sample_shape: str = cfg['sample']['shape']

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

    fwhms: list[FWHM] = []
    visibility_texts: list[list[str]] = []

    match oct_type, grover:
        case 'monochromatic', False:
            intensity = monochromatic_standard(peaks, sample_shape, path_delay, wavenumber)

            plt.plot(path_delay, intensity, label='Intensity')
            fwhms.append(FWHM(*find_fwhm(0.25, peaks, path_delay, intensity)))
            visibility_texts.append(find_visibility(peaks, path_delay, intensity))
        case 'nonmonochromatic', False:
            pass

        case 'quantum', False:
            coincidence_rate = quantum_standard(peaks, path_delay, spectral_width, speed_of_light, pump_frequency, refractive_index, sample_length)
            y_axis = "norm. QOCT C(τ_q)"

            plt.plot(path_delay, coincidence_rate, label='Coincidence Rate')
            fwhms.append(FWHM(*find_fwhm(1.0, peaks, path_delay, coincidence_rate, dips=True)))
            visibility_texts.append(find_visibility(peaks, path_delay, coincidence_rate))
        case 'monochromatic', True:
            intensity = monochromatic_grover(peaks, sample_shape, path_delay, speed_of_light, central_wavelength)

            if cfg['graphing']['display_interferogram']:
                plt.plot(path_delay, intensity, label='Intensity')
                fwhms.append(FWHM(*find_fwhm(0, peaks, path_delay, intensity)))
                visibility_texts.append(find_visibility(peaks, path_delay, intensity))

            if cfg['graphing']['display_scaled_interferogram']:
                sample = build_sample(peaks, sample_shape, path_delay)
                scaled_interferogram = scale_interferogram(0, peaks, path_delay, sample, intensity)
                plt.plot(path_delay, scaled_interferogram, label='Scaled Intensity')

                fwhms.append(FWHM(*find_fwhm(0, peaks, path_delay, scaled_interferogram)))
                visibility_texts.append(find_visibility(peaks, path_delay, scaled_interferogram))

        case 'nonmonochromatic', True:
            pass

        case 'quantum', True:
            interferogram = quantum_grover(peaks, sample_shape, path_delay, speed_of_light, central_wavelength)

            plt.plot(path_delay, interferogram, label='Coincidence Rate')
            fwhms.append(FWHM(*find_fwhm(1.0, peaks, path_delay, interferogram, dips=True)))
            visibility_texts.append(find_visibility(peaks, path_delay, interferogram))

    sample = build_sample(peaks, sample_shape, path_delay)
    plt.plot(path_delay, sample, label='Sample')

    infobox_texts = []
    # plot fwhm lines
    print(len(fwhms))
    print(fwhms)

    for fwhm in fwhms:
        infobox_texts += fwhm.texts

        print(fwhm.texts)

        for line in fwhm.lines:
            half_max, left_x, right_x = line
            plt.hlines(half_max, left_x, right_x, colors='red', linestyles='-', linewidth=2)

    # display infobox with FWHM and visibility
    for texts in visibility_texts:
        infobox_texts += texts
    if infobox_texts:
        plt.text(0.02, 0.98, '\n'.join(infobox_texts), transform=plt.gca().transAxes,
                verticalalignment='top', fontsize=10, bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})

    # generate latex for the delta function
    terms = []
    for reflectance, pos, _ in peaks:
        if reflectance == 1:
            terms.append(rf"\delta(z - {pos})")
        else:
            terms.append(rf"{reflectance}\delta(z - {pos})")
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

peaks = new_peaks(0.2, 0.2, 10, 10, 180.0, cfg['sample']['refractive_index'], cfg['sample']['length'])

plot_interferogram(cfg, peaks, cfg['graphing']['time_limit'], cfg['graphing']['y_limit'])
