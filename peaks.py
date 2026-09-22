import numpy as np

from custom import IndependentVariable, SampleFunc

type Peak = tuple[float, float]

def delta_function(peaks: list[Peak]) -> SampleFunc:
    """Takes (reflectance, depth) pairs and returns a sum of Gaussians."""
    def func(x: IndependentVariable) -> IndependentVariable:
        return sum(
            (reflectance * np.exp(-((x - depth) ** 2) / 10) for reflectance, depth in peaks),
            np.zeros_like(x) # empty array as start value
        )
    return func

def cornered_function(peaks: list[Peak]) -> SampleFunc:
    """Takes (reflectance, depth) pairs and returns a sum of absolute values."""
    def func(x: IndependentVariable) -> IndependentVariable:
        return sum(
            (np.maximum(0, reflectance * (1 - np.abs(x - depth) / 10))
             for reflectance, depth in peaks),
            np.zeros_like(x)
        )
    return func

def new_peaks(
    reflectance_1: float,
    reflectance_2: float,
    depth_1: float,
    refractive_index: float,
    sample_length: float) -> list[Peak]:
    """A basic sample is simply a transparent slab with reflectance entering and exiting the slab.
    `new_peaks` generates the coordinates that define this slab."""

    depth_2 = depth_1 + refractive_index * sample_length * 1e6 # micrometers
    return [(reflectance_1, depth_1), (reflectance_2, depth_2)] # (reflectance, location)

def build_sample(peaks: list[Peak], sample_shape: str, path_delay: IndependentVariable) -> IndependentVariable:
    sample_func: SampleFunc = lambda x: x # placeholder default

    match sample_shape:
        case "gaussian":
            sample_func = delta_function(peaks)
        case "cornered":
            sample_func = cornered_function(peaks)

    sample = sample_func(path_delay)
    return sample
