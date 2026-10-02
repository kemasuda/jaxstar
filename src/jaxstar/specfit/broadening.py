"""Combined rigid-rotation, radial-tangential macro and Gaussian broadening.

Numerical Fourier kernel ported from frozen jaxspec kernels.py (3b4ab913).
Small coefficient tuples retain precision across JAX x64 configuration changes.
No observation, continuum, Doppler shift or general IP abstraction lives here.
"""

import jax
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid as trapz

from .sampling import _C_KMS


def gaussian_sigma(resolving_power):
    """Gaussian IP standard deviation in km/s; frozen FWHM factor 2.354820."""
    return _C_KMS / jnp.asarray(resolving_power) / 2.354820


RP = tuple([-4.79443220978201773821E9, 1.95617491946556577543E12, -2.49248344360967716204E14, 9.70862251047306323952E15])
RQ = tuple([1., 4.99563147152651017219E2, 1.73785401676374683123E5, 4.84409658339962045305E7, 1.11855537045356834862E10, 2.11277520115489217587E12, 3.10518229857422583814E14, 3.18121955943204943306E16, 1.71086294081043136091E18])
DR1 =  5.78318596294678452118E0
DR2 = 3.04712623436620863991E1
PP = tuple([7.96936729297347051624E-4, 8.28352392107440799803E-2, 1.23953371646414299388E0, 5.44725003058768775090E0, 8.74716500199817011941E0, 5.30324038235394892183E0, 9.99999999999999997821E-1])
PQ = tuple([9.24408810558863637013E-4, 8.56288474354474431428E-2, 1.25352743901058953537E0, 5.47097740330417105182E0, 8.76190883237069594232E0, 5.30605288235394617618E0, 1.00000000000000000218E0])
QP = tuple([-1.13663838898469149931E-2, -1.28252718670509318512E0, -1.95539544257735972385E1, -9.32060152123768231369E1, -1.77681167980488050595E2, -1.47077505154951170175E2, -5.14105326766599330220E1, -6.05014350600728481186E0])
QQ = tuple([1., 6.43178256118178023184E1, 8.56430025976980587198E2, 3.88240183605401609683E3, 7.24046774195652478189E3, 5.93072701187316984827E3, 2.06209331660327847417E3, 2.42005740240291393179E2])
PIO4 = 0.78539816339744830962
SQ2OPI = 0.79788456080286535588


# Bessel function of the first kind, order zero.
def _j0(x):
    x = jnp.where(x > 0., x, -x)

    z = x * x
    ret = 1. - z / 4.

    p = (z - DR1) * (z - DR2)
    p = p * jnp.polyval(jnp.asarray(RP, dtype=x.dtype), z) / jnp.polyval(jnp.asarray(RQ, dtype=x.dtype), z)
    ret = jnp.where(x < 1e-5, ret, p)

    # Avoid a singular unselected branch when reverse-mode sees x=0.
    x_safe = jnp.where(x <= 5., 1., x)
    xinv5 = jnp.where(x <= 5., 0., 1. / x_safe)

    w = 5.0 * xinv5
    z = w * w
    p = jnp.polyval(jnp.asarray(PP, dtype=x.dtype), z) / jnp.polyval(jnp.asarray(PQ, dtype=x.dtype), z)
    q = jnp.polyval(jnp.asarray(QP, dtype=x.dtype), z) / jnp.polyval(jnp.asarray(QQ, dtype=x.dtype), z)
    xn = x - PIO4
    p = p * jnp.cos(xn) - w * q * jnp.sin(xn)
    ret = jnp.where(x <= 5., ret, p * SQ2OPI * jnp.sqrt(xinv5))

    return ret

# Hirano et al. (2011) ApJ 742, 69
# unlike in the paper, beta here is the Gaussian width
def combined_kernel(varr, vmacro, vsini, u1, u2, beta, Nt=500):
    """broadening kernel due to radial-tangential macroturbulence & vsini
    from Hirano et al. (2011) ApJ 742, 69, Appendix A, B

        Args:
            varr: velocities at which kernel is evaluated (length should be odd)
            vmacro: macroturbulence dispersion
            vsini: projected rotation velocity
            u1, u2: quadratic limb-darkening coefficient
            beta: additional Gaussian broadening (e.g., IP)
                NOTE: in Hirano+(2011), beta is introduced following Eq.(14),
                where beta is not a normal scale parameter of a Gaussian (sqrt(2) larger).
                beta in this function *is* defined as a scale parameter (standard deviation) of a Gaussian.
            Nt: length of grid to evalulate Fourier transform

        Returns:
            kernel whose sum is normalized to unity

    """
    tarr = jnp.linspace(0, 1, Nt)

    n, dv = len(varr), jnp.median(jnp.diff(varr))
    sigmas = jnp.fft.fftfreq(n, d=dv)

    t = tarr[:,None]
    piz2 = jnp.pi * jnp.pi * vmacro * vmacro
    t2 = t * t
    sig2 = sigmas * sigmas
    omint2 = 1. - t2
    projd = jnp.sqrt(omint2)
    ldfactor = (1. - (1. - projd) * (u1 + u2*(1. - projd))) / (1. - u1/3. - u2/6.) # limb-darkening
    rt = jnp.exp(-piz2*sig2*omint2) + jnp.exp(-piz2*sig2*t2) # macroturbulence
    ip = jnp.exp(-2*jnp.pi*jnp.pi*beta*beta*sig2) # instrumental profile (another convolution w/ Gaussian)
    ys = ldfactor * rt * ip * _j0(2*jnp.pi*sigmas*vsini*t) * t
    kernel_ft = trapz(ys.T)

    kernel = jnp.fft.fft(kernel_ft)
    kernel = jnp.real(kernel)
    kernel = jnp.fft.fftshift(kernel)

    return kernel / jnp.sum(kernel)


def default_broadening(wavelength, flux, *, broadening, resolving_power, velocity_grid):
    """Apply the combined kernel once per region, returning valid (wave, flux).

    Inputs have (region, pixel) layout; broadening values and resolving_power
    are already expanded to (region,). Output discards kernel half-support at
    both edges, so no zero padding enters requested model spectra. A replacement
    callable uses this signature and returns its own safe wavelength/flux arrays;
    it need not construct a convolution kernel or retain this output length.
    """
    def one(row, vsini, vmacro, u1, u2, resolution, velocity):
        kernel = combined_kernel(velocity, vmacro, vsini, u1, u2, gaussian_sigma(resolution))
        return jnp.convolve(row, kernel, mode="valid")

    values = jax.vmap(one)(flux, broadening["vsini"], broadening["vmacro"],
                           broadening["u1"], broadening["u2"], resolving_power, velocity_grid)
    half = velocity_grid.shape[-1] // 2
    return wavelength[:, half:-half], values
