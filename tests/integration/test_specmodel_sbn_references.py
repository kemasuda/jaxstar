"""SB2 parity against immutable TLUSTY artifacts and read-only legacy source."""

import os
from pathlib import Path
import runpy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxstar.specfit import SpecModel

ROOT = os.environ.get("JAXSPEC_REFERENCE_ROOT")
pytestmark = pytest.mark.skipif(not ROOT, reason="set JAXSPEC_REFERENCE_ROOT for frozen SB2 parity")
HELPER = runpy.run_path(str(Path(__file__).resolve().parents[2] / "examples/_specmodel_reference.py"))


@pytest.fixture(scope="module", autouse=True)
def precision(x64_context):
    with x64_context():
        yield


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    HELPER["verify_reference"](Path(ROOT).resolve())
    result = HELPER["make_case"](ROOT, tmp_path_factory.mktemp("sb2"), "tlusty",
        points=([31250., 3.85, -.12], [28750., 4.10, -.18]), reference_tag="tlusty_sb2")
    result.params = {
        "components": (
            {"atmosphere": {"teff": 31250., "logg": 3.85, "logZ": -.12},
             "broadening": {"vsini": 105., "vmacro": 7., "u1": .5, "u2": .2}, "rv": jnp.array([-180.82])},
            {"atmosphere": {"teff": 28750., "logg": 4.10, "logZ": -.18},
             "broadening": {"vsini": 85., "vmacro": 5., "u1": .4, "u2": .25}, "rv": jnp.array([180.82])}),
        "instrument": {"resolving_power": jnp.array([2500.])},
        "flux_weights": jnp.array([1., .65]), "dilution": jnp.array(0.)}
    yield result
    jax.clear_caches()


@pytest.mark.parametrize("dilution", [0., .2])
def test_saved_sb2_ratio_mapping_full_components_and_dilution(case, dilution):
    params = {**case.params, "dilution": dilution}
    model = SpecModel(case.spectra, vmax=case.vmax)
    physical = case.oracle["flux"] / HELPER["continuum_for_oracle"](case.wave)
    expected = dilution + (1 - dilution) * physical
    for evaluate in (lambda m, p, w: m(p, w), jax.jit(lambda m, p, w: m(p, w))):
        np.testing.assert_allclose(evaluate(model, params, case.wave), expected, rtol=5e-6, atol=1e-6)
    parts = model.decompose(params, case.wave)
    for index, component in enumerate(params["components"]):
        single = {"components": (component,), "instrument": params["instrument"]}
        old = HELPER["frozen_stages"](case, single)["full"]
        np.testing.assert_allclose(parts.components[index], old, rtol=5e-6, atol=1e-6)
    np.testing.assert_allclose(parts.stellar_fractions[:, 0], [1 / 1.65, .65 / 1.65], atol=1e-14)
    # The old SB2 class did not itself expose dilution. Its saved physical
    # mixture receives the preserved single-star dilution rule externally.


def test_saved_sb2_physical_and_ratio_gradients(case):
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
    mapped = {"f2_f1": gradient["flux_weights"][1], "wavres": gradient["instrument"]["resolving_power"]}
    for i, component in enumerate(gradient["components"], 1):
        mapped.update({f"{name}{i}": value for name, value in component["atmosphere"].items()})
        mapped.update({f"{name}{i}": value for name, value in component["broadening"].items() if name in ("vsini", "vmacro")})
        mapped[f"zeta{i}"] = mapped.pop(f"vmacro{i}")
        mapped[f"u1{i}"] = component["broadening"]["u1"]
        mapped[f"u2{i}"] = component["broadening"]["u2"]
        mapped[f"rv{i}"] = component["rv"]
    for key, actual in mapped.items():
        expected = case.oracle[f"gradient_{key}"]
        scale = max(float(np.max(np.abs(expected))), 1e-9)
        np.testing.assert_allclose(actual / scale, expected / scale, rtol=3e-4, atol=3e-4, err_msg=key)
