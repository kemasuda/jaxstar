"""Opt-in deterministic parity with immutable saved artifacts/frozen physics.

The full spectra and gradients use frozen_main.npz, with its fixed continuum
removed externally. Intermediate stages and per-region generalizations use
read-only frozen numerical source; no reference generator is run.
"""

import os
from pathlib import Path
import runpy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxstar.specfit import SpecModel
from jaxstar.specfit.broadening import combined_kernel


ROOT = os.environ.get("JAXSPEC_REFERENCE_ROOT")
pytestmark = pytest.mark.skipif(not ROOT, reason="set JAXSPEC_REFERENCE_ROOT for frozen forward-model parity")
HELPER = runpy.run_path(str(Path(__file__).resolve().parents[2] / "dev_notebooks/specfit/_specmodel_reference.py"))


@pytest.fixture(scope="module", autouse=True)
def precision(x64_context):
    with x64_context():
        yield


@pytest.fixture(scope="module", params=("coelho", "bosz", "tlusty"))
def case(request, tmp_path_factory):
    HELPER["verify_reference"](Path(ROOT).resolve())
    result = HELPER["make_case"](ROOT, tmp_path_factory.mktemp("forward"), request.param)
    yield result
    jax.clear_caches()


def test_intrinsic_broadened_full_match_frozen_source_and_saved_full(case):
    model = SpecModel(case.spectra, vmax=case.vmax)
    expected = HELPER["frozen_stages"](case)
    for stage in ("intrinsic", "broadened", "full"):
        evaluate = lambda m, p, w: getattr(m, stage)(p, w)
        for function in (evaluate, jax.jit(evaluate)):
            actual = function(model, case.params, case.wave)
            np.testing.assert_allclose(actual, expected[stage], rtol=5e-6, atol=1e-6)
    # There is no stored BOSZ full-forward artifact: its comparison above uses
    # the actual frozen numerical path on the supplied BOSZ atmosphere grid.
    if case.name != "bosz":
        physical = case.oracle["flux"] / HELPER["continuum_for_oracle"](case.wave)
        np.testing.assert_allclose(model(case.params, case.wave), physical, rtol=5e-6, atol=1e-6)


def test_region_parameters_match_equivalent_frozen_low_level_mapping(case):
    params = {"components": ({**case.params["components"][0],
                "broadening": {key: np.array([value, value * 1.2])[:len(case.wave)]
                               for key, value in case.params["components"][0]["broadening"].items()}},),
              "instrument": case.params["instrument"]}
    expected = HELPER["frozen_stages"](case, params)
    model = SpecModel(case.spectra, vmax=case.vmax)
    for stage in ("broadened", "full"):
        actual = jax.jit(lambda m, p, w: getattr(m, stage)(p, w))(model, params, case.wave)
        np.testing.assert_allclose(actual, expected[stage], rtol=5e-6, atol=1e-6)


def test_single_star_dilution_matches_frozen_model(case):
    component = case.params["components"][0]
    broad = component["broadening"]
    params = {**case.params, "dilution": .23}
    old_params = {**component["atmosphere"], "vsini": broad["vsini"], "zeta": broad["vmacro"],
                  "u1": broad["u1"], "u2": broad["u2"], "rv": component["rv"],
                  "wavres": params["instrument"]["resolving_power"], "dilution": .23,
                  "norm": jnp.ones(len(case.wave)), "slope": jnp.zeros(len(case.wave))}
    expected = case.old.fluxmodel_multiorder(old_params)
    actual = SpecModel(case.spectra, vmax=case.vmax)(params, case.wave)
    np.testing.assert_allclose(actual, expected, rtol=5e-6, atol=1e-6)


def test_saved_objective_physical_gradients(case):
    if case.name == "bosz":
        # No saved full-forward gradient oracle exists for BOSZ. Compare its
        # deterministic weighted-sum gradients directly with frozen source.
        weights = jnp.linspace(.3, 1.1, case.wave.size).reshape(case.wave.shape)
        old = lambda p: jnp.sum(HELPER["frozen_stages"](case, p)["full"] * weights)
        new = lambda m, p: jnp.sum(m(p, case.wave) * weights)
        expected = jax.jit(jax.grad(old))(case.params)
        _, actual = jax.jit(jax.value_and_grad(new, argnums=1))(SpecModel(case.spectra, vmax=case.vmax), case.params)
        for got, want in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
            scale = max(float(np.max(np.abs(want))), 1e-9)
            np.testing.assert_allclose(got / scale, want / scale, rtol=3e-4, atol=3e-4)
        return
    valid = ~case.oracle["mask_obs"] & np.isfinite(case.oracle["error_obs"]) & (case.oracle["error_obs"] > 0)
    flux = jnp.asarray(np.where(valid, case.oracle["flux_obs"], 1.))
    error = jnp.asarray(np.where(valid, case.oracle["error_obs"], 1.))
    base = jnp.asarray(HELPER["continuum_for_oracle"](case.wave))
    def objective(model, params):
        residual = (base * model(params, case.wave) - flux) / error
        return jnp.sum(jnp.where(valid, residual**2, 0.)) / valid.sum()
    value, gradient = jax.jit(jax.value_and_grad(objective, argnums=1))(
        SpecModel(case.spectra, vmax=case.vmax), case.params)
    np.testing.assert_allclose(value, case.oracle["objective"], rtol=2e-5, atol=2e-6)
    component = gradient["components"][0]
    mapped = {**component["atmosphere"], **component["broadening"], "rv": component["rv"],
              "wavres": gradient["instrument"]["resolving_power"]}
    mapped["zeta"] = mapped.pop("vmacro")
    for key, actual in mapped.items():
        expected = case.oracle[f"gradient_{key}"]
        scale = max(float(np.max(np.abs(expected))), 1e-9)
        np.testing.assert_allclose(actual / scale, expected / scale, rtol=3e-4, atol=3e-4, err_msg=key)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_combined_kernel_matches_frozen_numerical_logic(case, dtype):
    velocity = np.linspace(-50., 50., 101).astype(dtype)
    expected = case.modules.kernels.rotmacrokernel(velocity, 3.1, 6.3, .5, .2, 1.7, Nt=500)
    actual = combined_kernel(velocity, 3.1, 6.3, .5, .2, 1.7)
    np.testing.assert_allclose(actual, expected, rtol=5e-6, atol=1e-7)
