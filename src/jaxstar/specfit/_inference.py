"""Physical-coordinate MAP estimation, including model-dependent supports."""

from contextlib import ExitStack

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import SVI, Trace_ELBO, init_to_sample, init_to_value
from numpyro.infer.autoguide import AutoDelta
from numpyro.distributions.transforms import biject_to
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


class _PhysicalAutoDelta(AutoDelta):
    """AutoDelta with current support transforms and native deterministic sites.

    The optimizer parameters are unconstrained, but the guide's Delta sites
    are physical values with zero log-density correction. Trace_ELBO therefore
    minimizes the negative native joint density, without a transform Jacobian.
    Replaying the transforms avoids freezing conditional bounds at initialization.
    """

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)
        self._initial_unconstrained = {
            name: biject_to(self.prototype_trace[name]["fn"].support).inv(value)
            for name, value in self._init_locs.items()
        }

    def __call__(self, *args, **kwargs):
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)
        plates = self._create_plates(*args, **kwargs)
        unconstrained = {}
        for name, value in self._initial_unconstrained.items():
            site = self.prototype_trace[name]
            with ExitStack() as stack:
                for frame in site["cond_indep_stack"]:
                    stack.enter_context(plates[frame.name])
                unconstrained[name] = numpyro.param(
                    f"{name}_{self.prefix}_unconstrained_loc", value,
                    event_dim=self._event_dims[name])
        # This replays model-dependent supports using the current upstream values.
        # The auxiliary replay must not add model sites to the guide trace.
        with handlers.block():
            point = self._postprocess_fn(unconstrained)
        result = {}
        for name in self._init_locs:
            site = self.prototype_trace[name]
            with ExitStack() as stack:
                for frame in site["cond_indep_stack"]:
                    stack.enter_context(plates[frame.name])
                result[name] = numpyro.sample(
                    name, dist.Delta(point[name]).to_event(self._event_dims[name]))
        return result

    def median(self, params):
        unconstrained = {name: params[f"{name}_{self.prefix}_unconstrained_loc"]
                         for name in self._init_locs}
        return self._postprocess_fn(unconstrained)

    def sample_posterior(self, rng_key, params, *args, sample_shape=(), **kwargs):
        return {name: jnp.broadcast_to(value, sample_shape + jnp.shape(value))
                for name, value in self.median(params).items()}


def find_map_svi(model, step_size, num_steps, *, rng_key, p_initial=None,
                 progress_bar=True):
    """Return a local MAP estimate in the model's native sample-site coordinates.

    Optimize log likelihood + log prior with an AutoDelta point guide and
    Trace_ELBO. Bounds are enforced by internal coordinate transforms without
    adding their Jacobians to the objective. Returned deterministic sites are
    recomputed at the fitted sample values. This is not a Laplace approximation.
    """
    initial = init_to_sample if p_initial is None else init_to_value(values=p_initial)
    guide = _PhysicalAutoDelta(model, init_loc_fn=initial)
    svi = SVI(model, guide, numpyro.optim.Adam(step_size), Trace_ELBO())
    result = svi.run(rng_key, num_steps, progress_bar=progress_bar)
    return guide.median(result.params)
