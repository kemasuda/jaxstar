"""Laplace-mode SVI with model-dependent support transforms."""

import numpyro
from numpyro import handlers
from numpyro.infer import SVI, init_to_sample, init_to_value
from numpyro.infer.autoguide import AutoLaplaceApproximation
from numpyro.infer.elbo import ELBO
from numpyro.infer.util import potential_energy


class _LaplacePotentialLoss(ELBO):
    """Evaluate the point guide in the model's current unconstrained coordinates.

    AutoLaplaceApproximation's usual Trace_ELBO path transforms samples with
    prototype supports, whereas its median replays model-dependent supports.
    potential_energy replays those supports and includes their log Jacobians,
    matching the ordinary Laplace loss when supports are fixed.
    """

    def loss(self, rng_key, param_map, model, guide, *args, **kwargs):
        unconstrained = guide._unpack_latent(param_map[f"{guide.prefix}_loc"])
        model = handlers.substitute(model, data=param_map)
        return potential_energy(model, args, kwargs, unconstrained)


def find_map_svi(model, step_size, num_steps, *, rng_key, p_initial=None,
                 progress_bar=True):
    """Return native sample and deterministic sites at the Laplace guide mode."""
    initial = init_to_sample if p_initial is None else init_to_value(values=p_initial)
    guide = AutoLaplaceApproximation(model, init_loc_fn=initial)
    svi = SVI(model, guide, numpyro.optim.Adam(step_size), _LaplacePotentialLoss())
    result = svi.run(rng_key, num_steps, progress_bar=progress_bar)
    return guide.median(result.params)
