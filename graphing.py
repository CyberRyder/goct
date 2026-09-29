from math import pi

import matplotlib.pyplot as plt
import numpy as np

from analysis import find_fwhm, find_visibility, root_mean_square_distance, scale_interferogram
from custom import FWHM, DependentVariable, IndependentVariable
from models import (
    monochromatic_grover,
    monochromatic_standard,
    quantum_grover,
    quantum_standard,
)
from peaks import Peak, build_sample, new_peaks
from utils import ExperimentConfig, load_config


def plot_interferogram(cfg: ExperimentConfig, peaks: list[Peak], time_limit: float, y_limit: float):
    """Simulate an OCT scan, generating a graph relating path delay to intensity or coincidence according to the sample configured."""
    y_axis: str = "intensity"

    # TODO: write validator
    oct_type: str = cfg.graphing.oct_type
    grover: bool = cfg.graphing.grover

    sample_length: float = cfg.sample.length # meters, or None if omitted
    refractive_index: float = cfg.sample.refractive_index
    sample_shape: str = cfg.sample.shape

    central_wavelength: float = cfg.laser.central_wavelength # micrometers
    spectral_width: float = cfg.laser.spectral_width # extend about 100nm for broadband light source

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
    rmsd: float = -1

    sample = build_sample(peaks, sample_shape, path_delay)

    match oct_type, grover:
        case 'monochromatic', False:
            intensity = monochromatic_standard(peaks, sample_shape, path_delay, wavenumber)

            plt.plot(path_delay, intensity, label='Intensity')

            baseline = 0.25
            fwhms, visibility_texts, rmsd = update_infobox(fwhms, visibility_texts, baseline, peaks, path_delay, sample, intensity)

        case 'nonmonochromatic', False:
            pass

        case 'quantum', False:
            coincidence_rate = quantum_standard(peaks, path_delay, spectral_width, speed_of_light, pump_frequency, refractive_index, sample_length)
            y_axis = "norm. QOCT C(τ_q)"

            plt.plot(path_delay, coincidence_rate, label='Coincidence Rate')

            baseline = 1.0
            fwhms, visibility_texts, rmsd = update_infobox(fwhms, visibility_texts, baseline, peaks, path_delay, sample, coincidence_rate)

        case 'monochromatic', True:
            intensity = monochromatic_grover(peaks, sample_shape, path_delay, speed_of_light, central_wavelength)
            baseline = 0.7

            if cfg.graphing.display_interferogram:
                plt.plot(path_delay, intensity, label='Intensity')

                fwhms, visibility_texts, rmsd = update_infobox(fwhms, visibility_texts, baseline, peaks, path_delay, sample, intensity)

            if cfg.graphing.display_scaled_interferogram:
                sample = build_sample(peaks, sample_shape, path_delay)
                scaled_interferogram = scale_interferogram(cfg.graphing.scale_max, baseline, peaks, path_delay, sample, intensity)
                plt.plot(path_delay, scaled_interferogram, label='Scaled Intensity')

                fwhms, visibility_texts, rmsd = update_infobox(fwhms, visibility_texts, baseline, peaks, path_delay, sample, scaled_interferogram)

        case 'nonmonochromatic', True:
            pass

        case 'quantum', True:
            interferogram = quantum_grover(peaks, sample_shape, path_delay, speed_of_light, central_wavelength)

            plt.plot(path_delay, interferogram, label='Coincidence Rate')
            baseline = 1.0
            fwhms, visibility_texts, rmsd = update_infobox(fwhms, visibility_texts, baseline, peaks, path_delay, sample, interferogram)

    if cfg.graphing.display_sample:
        plt.plot(path_delay, sample, label='Sample')

    # display infobox with FWHM and visibility
    infobox_texts = build_infobox(fwhms, visibility_texts, rmsd)
    if infobox_texts:
        plt.text(0.02, 0.98, '\n'.join(infobox_texts), transform=plt.gca().transAxes,
                verticalalignment='top', fontsize=10, bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})

    # generate latex for the delta function
    terms: list[str] = []
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

def build_infobox(fwhms: list[FWHM], visibility_texts: list[list[str]], rmsd: float) -> list[str]:
    # build infobox
    infobox_texts: list[str] = []

    # plot fwhm lines
    for fwhm in fwhms:
        infobox_texts += fwhm.texts

        print(fwhm.texts)

        for line in fwhm.lines:
            half_max, left_x, right_x = line
            plt.hlines(half_max, left_x, right_x, colors='red', linestyles='-', linewidth=2)

    for texts in visibility_texts:
        infobox_texts += texts

    infobox_texts.append("RMSD: " + f"{rmsd:.7g}")

    return infobox_texts

def update_infobox(
    fwhms: list[FWHM],
    visibility_texts: list[list[str]],
    baseline: float,
    peaks: list[Peak],
    path_delay: IndependentVariable,
    sample: DependentVariable,
    interferogram: DependentVariable) -> tuple[list[FWHM], list[list[str]], float]:

    res = []
    for fwhm in fwhms: res.append(fwhm.texts)
    print("fwhms before update", res)
    fwhms.append(FWHM(*find_fwhm(baseline, peaks, path_delay, interferogram, dips=True)))
    visibility_texts.append(find_visibility(peaks, path_delay, interferogram))
    res = []
    for fwhm in fwhms: res.append(fwhm.texts)
    print("fwhms after update", res)

    rmsd = root_mean_square_distance(sample, interferogram)

    return fwhms, visibility_texts, rmsd

if __name__ == "__main__":
    cfg: ExperimentConfig = load_config()

    peaks = new_peaks(
        cfg.sample.entering_reflectance,
        cfg.sample.exiting_reflectance,
        cfg.sample.entering_reflectance_narrowness,
        cfg.sample.exiting_reflectance_narrowness,
        cfg.sample.initial_depth,
        cfg.sample.refractive_index,
        cfg.sample.length)

    plot_interferogram(cfg, peaks, cfg.graphing.time_limit, cfg.graphing.y_limit)
