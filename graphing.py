"""
This file contains the main logic creating the interferogram graphs, pulling in functions from the rest of the codebase.

It pulls in the experimental configuration and plots and analyzes accordingly.
"""

from math import pi

import matplotlib.pyplot as plt
import numpy as np

from analysis import (
    find_fwhm,
    find_visibility,
    root_mean_square_distance,
    scale_interferogram,
)
from custom import FWHM, DependentVariable, IndependentVariable
from models import (
    monochromatic_grover,
    monochromatic_standard,
    quantum_grover,
    quantum_standard,
)
from peaks import Peak, build_sample, new_peaks
from utils import ExperimentConfig, load_config


def scan(
    oct_type: str,
    grover: bool,
    path_delay: IndependentVariable,
    wavenumber: float,
    sample: DependentVariable,
    spectral_width: float,
    speed_of_light: int,
    pump_frequency: float,
    refractive_index: float,
    sample_length: float,
    central_wavelength: float,
) -> tuple[
    list[FWHM], list[list[str]], float, DependentVariable, DependentVariable, str
]:
    fwhms: list[FWHM] = []
    visibility_texts: list[list[str]] = []
    rmsd: float = -1

    interferogram: DependentVariable = np.zeros(0, dtype=np.float64)
    scaled_interferogram: DependentVariable = np.zeros(0, dtype=np.float64)

    baseline: float = 0
    y_label: str = ""

    # TODO: derive baselines from analyzing graph rather than hardcoding values
    match oct_type, grover:
        case "monochromatic", False:
            interferogram = monochromatic_standard(path_delay, sample, wavenumber)
            y_label = "Intensity"
            baseline = 0.25

        case "nonmonochromatic", False:
            pass

        case "quantum", False:
            interferogram = quantum_standard(
                peaks,
                path_delay,
                spectral_width,
                speed_of_light,
                pump_frequency,
                refractive_index,
                sample_length,
            )
            y_label = "Coincidence Rate"
            baseline = 1.0

        case "monochromatic", True:
            interferogram = monochromatic_grover(
                path_delay, sample, speed_of_light, central_wavelength
            )
            y_label = "Intensity"
            baseline = 0.7

        case "nonmonochromatic", True:
            pass

        case "quantum", True:
            interferogram = quantum_grover(
                path_delay, sample, speed_of_light, central_wavelength
            )
            y_label = "Coincidence Rate"
            baseline = 1.0

    if cfg.graphing.display_interferogram:
        fwhms, visibility_texts, rmsd = update_infobox(
            fwhms, visibility_texts, baseline, peaks, path_delay, sample, interferogram
        )

    if cfg.graphing.display_scaled_interferogram:
        scaled_interferogram = scale_interferogram(
            cfg.graphing.scale_max,
            baseline,
            peaks,
            path_delay,
            sample,
            interferogram,
        )
        fwhms, visibility_texts, rmsd = update_infobox(
            fwhms,
            visibility_texts,
            baseline,
            peaks,
            path_delay,
            sample,
            scaled_interferogram,
        )

    return (fwhms, visibility_texts, rmsd, interferogram, scaled_interferogram, y_label)


def plot_interferogram(
    cfg: ExperimentConfig, peaks: list[Peak], time_limit: float, y_limit: float
):
    """Simulate an OCT scan, generating a graph relating path delay to intensity or coincidence according to the sample configured."""
    oct_type: str = cfg.graphing.oct_type
    grover: bool = cfg.graphing.grover

    sample_length: float = cfg.sample.length  # meters, or None if omitted
    refractive_index: float = cfg.sample.refractive_index
    sample_shape: str = cfg.sample.shape

    central_wavelength: float = cfg.laser.central_wavelength  # micrometers
    spectral_width: float = (
        cfg.laser.spectral_width
    )  # extend about 100nm for broadband light source

    # facts
    speed_of_light: int = 299792458  # meters / second
    central_frequency: float = (
        2 * pi * speed_of_light / (central_wavelength * 1e-6)
    )  # radians / second
    pump_frequency: float = 2 * central_frequency  # radians / second
    wavenumber: float = 2 * pi / central_wavelength  # radians / micrometers
    time_shift: IndependentVariable = np.linspace(0, time_limit, 10000)  # picoseconds
    path_delay: IndependentVariable = speed_of_light * time_shift * 1e-6  # micrometers

    # initialization
    plt.figure(figsize=(14, 6))

    fwhms: list[FWHM] = []
    visibility_texts: list[list[str]] = []
    rmsd: float = -1

    sample = build_sample(peaks, sample_shape, path_delay)

    # conduct scan
    fwhms, visibility_texts, rmsd, interferogram, scaled_interferogram, y_label = scan(
        oct_type,
        grover,
        path_delay,
        wavenumber,
        sample,
        spectral_width,
        speed_of_light,
        pump_frequency,
        refractive_index,
        sample_length,
        central_wavelength,
    )

    # plot graphs accordingly
    if cfg.graphing.display_sample:
        plt.plot(path_delay, sample, label="Sample")

    if cfg.graphing.display_interferogram:
        plt.plot(path_delay, interferogram, label=y_label)

    if cfg.graphing.display_scaled_interferogram:
        plt.plot(path_delay, scaled_interferogram, label="Scaled " + y_label)

    # display infobox with FWHM and visibility
    infobox_texts = build_infobox(fwhms, visibility_texts, rmsd)
    if infobox_texts:
        plt.text(
            0.02,
            0.98,
            "\n".join(infobox_texts),
            transform=plt.gca().transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    # generate latex for the delta function
    terms: list[str] = []
    for reflectance, pos, _ in peaks:
        if reflectance == 1:
            terms.append(rf"\delta(z - {pos})")
        else:
            terms.append(rf"{reflectance}\delta(z - {pos})")
    sample_label = " + ".join(terms)

    # finalize graph
    plt.title(f"Graph of {y_label} vs. path delay with sample $r(z) = {sample_label}$")
    plt.xlabel(r"Path delay $c \tau$ ($\mu$m)")
    plt.ylabel(f"{y_label}")
    plt.xlim(0, max(path_delay))
    plt.ylim(0, y_limit)
    plt.legend()

    plt.show()


# peaks = ((1, 180), (1.4, 450))


def build_infobox(
    fwhms: list[FWHM], visibility_texts: list[list[str]], rmsd: float
) -> list[str]:
    # build infobox
    infobox_texts: list[str] = []

    # plot fwhm lines
    for fwhm in fwhms:
        infobox_texts += fwhm.texts

        print(fwhm.texts)

        for line in fwhm.lines:
            half_max, left_x, right_x = line
            plt.hlines(
                half_max, left_x, right_x, colors="red", linestyles="-", linewidth=2
            )

    # populate infobox
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
    interferogram: DependentVariable,
) -> tuple[list[FWHM], list[list[str]], float]:

    res = []
    for fwhm in fwhms:
        res.append(fwhm.texts)
    print("fwhms before update", res)
    fwhms.append(
        FWHM(*find_fwhm(baseline, peaks, path_delay, interferogram, dips=True))
    )
    visibility_texts.append(find_visibility(peaks, path_delay, interferogram))
    res = []
    for fwhm in fwhms:
        res.append(fwhm.texts)
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
        cfg.sample.length,
    )

    plot_interferogram(cfg, peaks, cfg.graphing.time_limit, cfg.graphing.y_limit)
