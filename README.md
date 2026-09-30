Below are the equations used to create interferograms.

Equations governing standard OCT:

Monochromatic case:
$I = \frac{1}{4} I_0 \left(1 + |r(z)|^{2} + 2r(z)\cos(2kz - \omega\tau)\right)$

where
$I$ is the intensity at the detector
$I_{0}$ is the initial intensity of the light
$r(z)$ is the reflectance of the sample
$z$ is the depth into the sample (equate to $\delta l = c \tau$)
$c$ is the speed of light
$l$ is the length of the reference arm
$k$ is the angular wavenumber (equate to $\frac{2\pi}{\text{wavelength}}$)
$\omega$ is the frequency of the light
$\tau$ is the time shift of the reference arm


Nonmonochromatic case (note: this equation is currently unused):
$I = \frac{1}{4} \left(L_{0} + L_{1} * 2 * Re(e^{-i * w_0 * \tau})\right)$

where
$I$ is the intensity at the detector
$L_{0}$ is $\int s(\omega) (1 + |H(\omega)|^2) \,d\omega$, which is the background term
$L_{1}(\tau)$ is $\int 2 s(\omega) \Re(H(\omega) * e^{-i * \omega * \tau}) \,d\omega$, which is the interference term
$s(\omega)$ is the spectral power density (a known function that depends on the light used)
$H(\omega)$ is $\int e^{2 i phi(z, \omega)} r(z, \omega)\,dz$, which is the sample

$r(z, \omega)$ is the reflectance of the sample
$\phi(z, \omega)$ is the phase shift of the sample
$z$ is the depth into the sample (equate to $\delta l = c \tau$)
$c$ is the speed of light
$l$ is the length of the reference arm
$\omega$ is the frequency distribution, centered around w0
$\tau$ is the time shift of the reference arm


Quantum case:
$R = A_{0} - \Re(A_{1}(2 \tau))$

where
$R$ is the coincidence rate at the detectors
$A_{0}$ is $\int 2 s(\omega) |H(\omega)|^{2} \,d\omega$, which is the background term
$A_{1}(\tau)$ is $\int 2 s(\omega) e^{i \omega \tau} H(\omega) H'(-\omega) \,d\omega$, which is the interference term
$s(\omega)$ is the spectral power density (a known function that depends on the light used)
$H(\omega)$ is the integral of $\int e^{2 i \phi(z, \omega)} r(z, \omega) \,dz$, which is the sample
$H'(\omega)$ is the complex conjugate of $H(\omega)$

$r(z, \omega)$ is the reflectance of the sample
$\phi(z, \omega)$ is the phase shift of the sample
$z$ is the depth into the sample (equate to $\delta l = c \tau$)
$c$ is the speed of light
$l$ is the length of the reference arm
$\omega$ is the frequency distribution, centered around $w_{0}$
$tau$ is the time shift of the reference arm



For the nonquantum cases, the sample $r(\omega)$ is modeled by Gaussians representing the reflection points.
For the quantum cases, the sample $H(\omega)$ is modeled also by a sum of reflectances:

$H(\omega) = r_{1} + r_{2} * e^{ 2 i \omega n L / c}$
with reflectance points $r_{1}$ and $r_{2}$
where
$n$ is the refractive index
$L$ is the sample thickness
$c$ is the speed of light

From this definition we obtain that

$A_{0} = |r_{1}|^{2} + |r_{2}|^{2}$
$A(\tau_{q}) = |r_{1}|^{2} s(\tau_{q}) + |r_{2}|^{2} s(\tau_{q} - 2 \tau_{d}) + 2\Re(r_{1} r_{2}' s(\tau_{q} - \tau_{d}) e^{i \omega_{p} n L / c})$

where
$\tau_{q}$ is the time delay of the interferometer
$\tau_{d} = 2 n L / c$ is the time delay from passing through the sample
$\omega_{p} = 2 \omega_{0}$ is the frequency of the original photon before the spectral down conversion
$r_{1}$ and $r_{2}$ are the reflectance points
$n$ is the refractive index
$L$ is the sample thickness
$c$ is the speed of light
