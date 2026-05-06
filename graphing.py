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
'''

from math import pi
import numpy as np
import matplotlib.pyplot as plt

def find_fwhm(baseline, peaks, path_delay, intensity):
    fwhm_texts = [] # infobox contents
    fwhm_lines = [] # list of (half_max, left_x, right_x) for each peak
    
    for _, pos in peaks:
        mask = np.abs(path_delay - pos) < 50 # Boolean array telling whether each point is near a peak
        if not np.any(mask):
            continue
        
        # grab x and y values at these points
        region_intensity = intensity[mask]
        region_path = path_delay[mask]
        
        peak_max = np.max(region_intensity)
        half_max = baseline + (peak_max - baseline) / 2
        
        top_half = region_intensity >= half_max
        crossings = np.where(np.diff(top_half.astype(int)))[0] # 1 means low to high, -1 high to low, 0 same side. finally return where crossings occur
        
        # draw the line starting from when it crosses into the top and ending when it comes to the bottom
        if len(crossings) >= 2:
            left_idx = crossings[0]
            right_idx = crossings[-1]
            fwhm = region_path[right_idx] - region_path[left_idx]
            fwhm_texts.append(f"FWHM at z={pos}: {fwhm:.2f} μm")
            fwhm_lines.append((half_max, region_path[left_idx], region_path[right_idx]))
    
    return fwhm_lines, fwhm_texts

def plot_intensity(oct_type, sample_func, peaks, time_limit, intensity_limit):
    wavelength = 0.800 # micrometers
    speed_of_light = 299792458 # meters / second
    wavenumber = 2 * pi / wavelength # 1 / micrometers

    time_shift = np.linspace(0, time_limit, 10000) # picoseconds
    path_delay = speed_of_light * time_shift * (10 ** -6) # micrometers
    sample = sample_func(path_delay) # function of micrometers


    match oct_type:
        case 'monochromatic':
            monochromatic_intensity = 1/4 * (1 + np.abs(sample) ** 2 + 2 * sample * np.cos(wavenumber * path_delay))
        case 'nonmonochromatic':
            frequency_distribution = 1 #TODO: write actual frequency distribution (use Gaussian)
            
        case 'quantum':
            1
        case 'grover':
            1

    plt.figure(figsize=(14, 6))
    plt.plot(path_delay, monochromatic_intensity, label='Intensity')
    plt.plot(path_delay, sample, label='Sample')

    fwhm_lines, fwhm_texts = find_fwhm(0.25, peaks, path_delay, monochromatic_intensity)

    # plot fwhm lines
    for half_max, left_x, right_x in fwhm_lines:
        plt.hlines(half_max, left_x, right_x, colors='red', linestyles='-', linewidth=2)
    
    # display fwhm infobox
    if fwhm_texts:
        plt.text(0.02, 0.98, '\n'.join(fwhm_texts), transform=plt.gca().transAxes,
                verticalalignment='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # generate latex for the delta function
    terms = []
    for weight, pos in peaks:
        if weight == 1:
            terms.append(rf"\delta(z - {pos})")
        else:
            terms.append(rf"{weight}\delta(z - {pos})")
    label = " + ".join(terms)

    plt.title(f'Graph of intensity vs. path delay with sample $r(z) = {label}$')
    plt.xlabel(r'Path delay $c \tau$ ($\mu$m)')
    plt.ylabel('intensity')
    plt.xlim(0, max(path_delay))
    plt.ylim(0, intensity_limit)
    plt.legend()

    plt.show()

def delta_function(*peaks):
    """Takes (weight, position) pairs and returns a sum of Gaussians and the peaks."""
    func = lambda x: sum(weight * np.exp(-((x - pos) ** 2) / 10) for weight, pos in peaks)
    return func, peaks

sample, peaks = delta_function((1, 180), (1.4, 450))
plot_intensity('monochromatic', sample, peaks, 2, 2)