from collections.abc import Callable

import numpy as np

from custom import DependentVariable

type Peak = tuple[float, float]

def delta_function(peaks: list[Peak]) -> Callable[[DependentVariable], DependentVariable]:
    """Takes (reflectance, depth) pairs and returns a sum of Gaussians."""
    def func(x: DependentVariable) -> DependentVariable:
        return sum(
            (reflectance * np.exp(-((x - depth) ** 2) / 10) for reflectance, depth in peaks),
            np.zeros_like(x) # empty array as start value
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
