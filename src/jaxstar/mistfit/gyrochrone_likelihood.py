""" gyrochrone likelihood from Angus et al. (2019), AJ 158, 173

* the following code has been adapted from https://github.com/RuthAngus/stardate
* here gyro models for stars with BP-RP>2.7 have not yet been implemented
"""

import jax.numpy as jnp
from jax import jit

#%%
@jit
def loglike_gyro(prot, bprp, mass, eep, logage, feh, sigma=0.05):
    """ gyrochronal log-likelihood from Angus et al. (2019), AJ 158, 173

        Args:
            prot: rotation period (days)
            bprp: Gaia DR2 BP-RP color
            mass: stellar mass (solar unit)
            eep: equivalent evolutionary phase
            logage: log10(age/yr)
            feh: Fe/H

        Returns:
            loglikelihood

    """
    sigma_p = getsigmap(eep, logage, feh, bprp)
    logp_model = gyro_model_praesepe(logage, bprp, mass)
    #var = sigma_p**2 + (sigma/jnp.log(10))**2
    var = (sigma_p + sigma/jnp.log(10))**2 # follow the original paper and code
    return -0.5 * ((jnp.log10(prot) - logp_model)**2 / var + jnp.log(var))

#%%
@jit
def gyro_model_praesepe(logage, bprp, mass, Ro_cutoff=2):
    """ gyrochrone model calibrated using Praesepe
    NOTE: BP-RP>2.7 not yet supported!

        Args:
            logage: log10(age/yr)
            bprp: Gaia DR2 BP-RP color
            mass: stellar mass (solar unit)
            Ro_cutoff: critical Rossby number for weakened magnetic braking

        Returns:
            log10(rotation period/days)

    """
    # Angus+19 table 1
    # c4, c3, c2, c1, c0, cA, b1, b0
    p = jnp.array([-38.957586198640314, 28.709418579540294, -4.919056437046026,
         0.7161114835620975, -4.716819674578521, 0.6470950862322454,
         -13.558898318835137, 0.9359250478865809])
    logprot = jnp.where(bprp < 0.56, 0.56, jnp.polyval(p[:5], jnp.log10(bprp)) + p[5]*logage)

    tau = convective_overturn_time(mass)
    logpmax = jnp.log10(Ro_cutoff * tau)
    Ro = 10**logprot / tau
    logprot = jnp.where((Ro >= Ro_cutoff) & (bprp >= 0.56), logpmax, logprot)

    return logprot

@jit
def sigmoid(k, x0, L, x):
    """ sigmoid function """
    return L / (jnp.exp(-k*(x - x0)) + 1)

@jit
def getsigmap(eep, logage, feh, color):
    """ width of the gyro relation

        Args:
            eep: equivalent evolutionary phase
            logage: log10(age/yr)
            feh: Fe/H
            color: Gaia DR2 BP-RP color

        Returns:
            sigma in Eq.(3) of Angus+19

    """
    kcool, khot, keep = 100, 100, .2
    Lcool, Lhot, Leep = .5, .5, 5
    x0eep = 454
    k_old, x0_old = 100, jnp.log10(10*1e9)
    k_young, x0_young = 20, jnp.log10(250*1e6)
    L_age = .5
    k_feh, L_feh, x0_feh = 50, .5, .2

    x0cool, x0hot = .4, .25
    logc = jnp.log10(jnp.where(color > 0, color, 1))
    sigma_color = jnp.where(color > 0, sigmoid(kcool, x0cool, Lcool, logc) + sigmoid(khot, x0hot, Lhot, -logc), 0.5)

    sigma_eep = sigmoid(keep, x0eep, Leep, eep)
    sigma_age = sigmoid(k_young, -x0_young, L_age, -logage) # + sigmoid(k_old, x0_old, L_age, log_age)
    sigma_feh = sigmoid(k_feh, x0_feh, L_feh, feh) + sigmoid(k_feh, x0_feh, L_feh, -feh)
    sigma_total = sigma_color + sigma_eep + sigma_feh + sigma_age

    return sigma_total

@jit
def convective_overturn_time(mass):
    """ convective overturn time given stellar mass

        Args:
            mass: stellar mass (solar)

        Returns:
            convective overturn time (days)

    """
    logm = jnp.log10(mass)
    logtau = 1.16 - 1.49 * logm - 0.54 * logm * logm
    return 10**logtau

@jit
def rossby_number(prot, mass):
    """ Rossby number prot/tau

        Args:
            prot: rotation period (days)
            mass: stellar mass (solar unit)

        Returns:
            Rossby number

    """
    tau = convective_overturn_time(mass)
    return prot / tau
