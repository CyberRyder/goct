import numpy as np

from custom import DependentVariable, FWHMLine, IndependentVariable
from peaks import Peak


def find_visibility(peaks: list[Peak], path_delay: IndependentVariable, intensity: DependentVariable) -> list[str]:
    """Calculate visibility V = (I_max - I_min) / (I_max + I_min) for each peak region."""
    visibility_texts = []

    for _, pos, _ in peaks:
        mask = np.abs(path_delay - pos) < 50
        if not np.any(mask):
            continue

        region_intensity = intensity[mask]
        i_max = np.max(region_intensity)
        i_min = np.min(region_intensity)

        if i_max + i_min > 0:
            visibility = (i_max - i_min) / (i_max + i_min)
            visibility_texts.append(f"Visibility at z={pos}: {visibility:.3f}")

    return visibility_texts

def find_fwhm(
    baseline: float,
    peaks: list[Peak],
    path_delay: IndependentVariable,
    intensity: DependentVariable,
    *, dips=False) -> tuple[list[FWHMLine], list[str]]:
    """Generate co-ordinates of line displaying the full width at half-maximum for each peak region."""

    fwhm_lines: list[FWHMLine] = [] # list of (half_max, left_x, right_x) for each peak
    fwhm_texts: list[str] = [] # infobox contents

    for _, pos, _ in peaks:
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

    return (fwhm_lines, fwhm_texts)

def scale_interferogram(baseline: float, peaks: list[Peak], path_delay: IndependentVariable, sample: DependentVariable, interferogram: DependentVariable) -> DependentVariable:
    """Scale the interferogram to have the same maximum as the sample."""
    scaled_interferogram = interferogram.copy()

    for _, pos, _ in peaks:
        mask = np.abs(path_delay - pos) < 50
        if not np.any(mask):
            continue

        sample_intensity = sample[mask]
        s_max = np.max(sample_intensity)

        interferogram_intensity = interferogram[mask]
        i_max = np.max(interferogram_intensity)

        scaled_interferogram[mask] = interferogram[mask] / i_max * s_max - baseline

    return scaled_interferogram
