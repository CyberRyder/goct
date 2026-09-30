"""
This file contains the code for generating the interferograms of each OCT type.
A more complete explanation of the equations used can be found in README.md.
"""

import numpy as np

from custom import DependentVariable, IndependentVariable
from peaks import Peak, build_sample


def monochromatic_standard(
    peaks: list[Peak],
    sample_shape: str,
    path_delay: IndependentVariable,
    wavenumber: float,
) -> DependentVariable:
    sample = build_sample(peaks, sample_shape, path_delay)
    intensity: DependentVariable = (
        1 / 4 * (1 + np.abs(sample) ** 2 + 2 * sample * np.cos(wavenumber * path_delay))
    )

    return intensity


def quantum_standard(
    peaks: list[Peak],
    path_delay: IndependentVariable,
    spectral_width: float,
    speed_of_light: float,
    pump_frequency: float,
    refractive_index: float,
    sample_length: float,
) -> DependentVariable:
    _spectral_power_distribution = lambda w: np.exp(
        -(w**2) / (2 * spectral_width**2)
    )  # function of frequency (radians / second)
    # For Gaussian s(w) = exp(-w^2/(2*sigma^2)), FT gives gamma(tau) = sigma*sqrt(2*pi)*exp(-tau^2*sigma^2/2)
    _coherence_raw = lambda tau: (
        spectral_width
        * np.sqrt(2 * np.pi)
        * np.exp(-(tau**2) * (spectral_width**2) / 2)
    )
    _coherence_at_zero = _coherence_raw(
        0.0
    )  # in order to normalize the coherence function
    coherence_function = lambda tau: _coherence_raw(tau) / _coherence_at_zero

    [r1, z1, _], [r2, z2, _] = peaks

    # graph this for reference, but not actually used in calculations
    # _sample_func = lambda frequency: r1 + r2 * np.exp(1j * 2 * frequency * refractive_index * sample_length / speed_of_light)
    # TODO: fix this
    # _sample = _sample_func(path_delay)

    tau_q = path_delay * 1e-6 / speed_of_light  # seconds
    tau_z1 = (
        z1 * 1e-6 / speed_of_light
    )  # seconds (delay until reaching first reflectance)
    tau_z2 = (
        z2 * 1e-6 / speed_of_light
    )  # seconds (delay until reaching second reflectance)
    tau_d = tau_z2 - tau_z1

    # A0 and A(tau_q) use the same normalized coherence function (gamma(0)=1)
    background_term = np.abs(r1) ** 2 + np.abs(r2) ** 2

    # Interference term: dips at front (z1) and back (z2) interfaces
    # A(tau-q) = |r1|^2 * s(tau-q) + |r2|^2 * s(tau-q - 2 * tau-d) + 2Re(r1 * r2' * s(tau-q - tau-d) * e^(i * w-p * n * L / c))
    # tau_d = tau_z2 - tau_z1

    # also, we subtract a tau_z1 from each argument for the rightward shift
    term1 = np.abs(r1) ** 2 * coherence_function(tau_q - tau_z1)
    term2 = np.abs(r2) ** 2 * coherence_function(tau_q - 2 * tau_d - tau_z1)
    cross_term_phase = (
        pump_frequency * refractive_index * sample_length / speed_of_light
    )
    cross_term = 2 * np.real(
        r1
        * np.conj(r2)
        * coherence_function(tau_q - tau_d - tau_z1)
        * np.exp(1j * cross_term_phase)
    )
    interference_term = term1 + term2 + cross_term

    # R = A0 - A(tau_q), normalized so baseline = 1
    coincidence_rate: DependentVariable = (
        background_term - interference_term
    ) / background_term

    """fix this
    # create gaussian to represent sample
    sample = sum(
        weight * np.exp(-((path_delay - pos) ** 2) / 10)
        for weight, pos in peaks
    )
    """

    return coincidence_rate


def monochromatic_grover(
    peaks: list[Peak],
    sample_shape: str,
    path_delay: IndependentVariable,
    speed_of_light: float,
    central_wavelength: float,
) -> DependentVariable:
    # the sample needs to take both arguments of z and w
    # how do I properly handle the nonmonochromatic configuration?
    frequency = speed_of_light / central_wavelength
    tau = path_delay * 1e-6 / speed_of_light

    reference_arm = np.exp(1j * frequency * tau)
    sample = build_sample(peaks, sample_shape, path_delay)
    r_positive = reference_arm + sample
    r_negative = reference_arm - sample

    exiting_state = 0.5 * (1 + r_positive - (r_negative) ** 2 / (2 + r_positive))
    intensity: DependentVariable = np.abs(exiting_state) ** 2

    return intensity


# note: currently broken
def quantum_grover(
    peaks: list[Peak],
    sample_shape: str,
    path_delay: IndependentVariable,
    speed_of_light: float,
    central_wavelength: float,
) -> DependentVariable:
    frequency = speed_of_light / central_wavelength
    tau = path_delay * 1e-6 / speed_of_light

    reference_arm = np.exp(1j * frequency * tau)
    sample = build_sample(peaks, sample_shape, path_delay)

    coincidence_rate: DependentVariable = (
        reference_arm
        + sample
        + (((reference_arm - sample) ** 2) / (reference_arm + sample))
        * (4 / ((2 - reference_arm - sample) * (2 + reference_arm + sample)))
    )

    return coincidence_rate
