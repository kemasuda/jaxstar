import jax.numpy as jnp
from jax import jit

# gyrochrone likelihood from Angus et al. (2019)
# gyro model for stars with BP-RP>2.7 has not yet been implemented

#%%
@jit
def loglike_gyro(prot, bprp, mass, eep, logage, feh, sigma=0.05):
    sigma_p = getsigmap(eep, logage, feh, bprp)
    logp_model = gyro_model_praesepe(logage, bprp, mass)
    var = sigma_p**2 + (sigma/jnp.log(10))**2
    return -0.5 * ((jnp.log10(prot) - logp_model)**2 / var + jnp.log(var))

# for bprp < 2.7!!
@jit
def gyro_model_praesepe(logage, bprp, mass, Ro_cutoff=2):
    # angus+19 table 1
    # c4, c3, c2, c1, c0, cA, b1, b0
    p = [-38.957586198640314, 28.709418579540294, -4.919056437046026,
         0.7161114835620975, -4.716819674578521, 0.6470950862322454,
         -13.558898318835137, 0.9359250478865809]
    logprot = jnp.where(bprp < 0.56, 0.56, jnp.polyval(p[:5], jnp.log10(bprp)) + p[5]*logage)
    tau = convective_overturn_time(mass)
    logpmax = jnp.log10(Ro_cutoff * tau)
    Ro = 10**logprot / tau
    #logprot = jnp.where(Ro < 2, logprot, logpmax)
    logprot = jnp.where((Ro >= 2) & (bprp >= 0.56), logpmax, logprot)
    return logprot

@jit
def sigmoid(k, x0, L, x):
    return L / (jnp.exp(-k*(x - x0)) + 1)

@jit
def getsigmap(eep, logage, feh, color):
    kcool, khot, keep = 100, 100, .2
    Lcool, Lhot, Leep = .5, .5, 5
    x0eep = 454 #454  #100 #454
    k_old, x0_old = 100, jnp.log10(10*1e9)
    k_young, x0_young = 20, jnp.log10(250*1e6)
    L_age = .5
    # k_feh, L_feh, x0_feh = 5, .5, 3.
    k_feh, L_feh, x0_feh = 50, .5, .2
    # k_feh, L_feh, x0_feh = 50, .5, .25

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
    logm = jnp.log10(mass)
    logtau = 1.16 - 1.49 * logm - 0.54 * logm * logm
    return 10**logtau

@jit
def rossby_number(prot, mass):
    tau = convective_overturn_time(mass)
    return prot / tau
