import numpy as np
from peaks import Peak, delta_function
from custom import DependentVariable

def monochromatic_standard(
    peaks: list[Peak],
    path_delay: DependentVariable,
    wavenumber: float
):
    sample_func = delta_function(peaks)
    sample = sample_func(path_delay)
    intensity = 1/4 * (1 + np.abs(sample) ** 2 + 2 * sample * np.cos(wavenumber * path_delay))

    return intensity
