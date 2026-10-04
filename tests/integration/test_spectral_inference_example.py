"""Example contracts without a long inference run or external downloads."""

import importlib.util
import os
from pathlib import Path

import jax
import numpy as np
from numpyro.infer.util import log_density
import pytest

from jaxstar.specfit import model_single


def example(name):
    path = Path(__file__).resolve().parents[2] / "dev_notebooks/specfit" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_synthetic_actual_forward_and_offset_pixels(x64_context, dtype):
    with x64_context(dtype == np.float64):
        case = example("_spectral_inference_data").synthetic_case(regions=2, pixels=80, model_pixels=601, dtype=dtype)
        obs, model = case.observation, case.specmodel
        assert case.minimum_grid_separation > 0
        assert obs.shape == (2, 80)
        assert np.any(obs.mask) and np.all(np.isnan(obs.flux[obs.mask]))
        lp = lambda point, obs, model: log_density(model_single, (obs, model), case.priors, point)[0]
        truth = {k: case.truth.get(k, case.reference_sigma_continuum) for k in case.initial}
        value, gradient = jax.jit(jax.value_and_grad(lp))(truth, obs, model)
        assert np.isfinite(value)
        assert all(np.all(np.isfinite(x)) for x in jax.tree.leaves(gradient))
        displaced = dict(truth, rv=truth["rv"] + 3., teff=6300.)
        assert value > lp(displaced, obs, model) + 20
        assert np.all(case.coefficients[:, 3:] == 0)


def test_existing_real_order_setup_and_common_artifact(tmp_path, x64_context):
    root = Path(os.environ.get("JAXSPEC_REFERENCE_ROOT", "../jaxspec"))
    if not (root / "characterization/sample_data_ird.csv").exists():
        pytest.skip("optional frozen sample input unavailable")
    with x64_context():
        helper = example("spectral_inference_real_setup")
        artifact = tmp_path / "ird.npz"
        case = helper.load_ird_order8(root, artifact, rv_initial=12.4, stride=32)
        assert case.observation.region == case.specmodel.spectra.regions == (8,)
        assert artifact.exists()
        assert np.any(case.observation.mask)
        density, _ = log_density(model_single, (case.observation, case.specmodel), case.priors, case.initial)
        assert np.isfinite(density)
        # Future loading uses only the common artifact, without opening the legacy grid.
        before = artifact.stat().st_mtime_ns
        loaded = helper.load_ird_order8(root, artifact, rv_initial=12.4, stride=32)
        assert artifact.stat().st_mtime_ns == before
        np.testing.assert_array_equal(loaded.specmodel.spectra.wavelength, case.specmodel.spectra.wavelength)
