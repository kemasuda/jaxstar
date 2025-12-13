import numpy as np
import pandas as pd
import pytest
import jax.random as random

from jaxstar.mistfit.mistfit import MistFit, check_mistgrid_path


class DummyKernel:
    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs


class DummyMCMC:
    def __init__(self, kernel, num_warmup, num_samples):
        self.kernel = kernel
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.runs = []
        self.summary_called = False

    def run(self, rng_key, **kwargs):
        self.runs.append((rng_key, kwargs))

    def print_summary(self):
        self.summary_called = True

    def get_samples(self):
        return {"age": np.array([1.0]), "kmag": np.array([2.0])}


@pytest.mark.filterwarnings("ignore:Conversion of an array with ndim > 0 to a scalar is deprecated")
def test_setup_and_run_hmc_uses_injected_mcmc(monkeypatch, synthetic_mistgrid):
    cfg = synthetic_mistgrid
    monkeypatch.setattr(
        "jaxstar.mistfit.mistfit.numpyro.infer.NUTS", DummyKernel
    )
    monkeypatch.setattr(
        "jaxstar.mistfit.mistfit.numpyro.infer.MCMC", DummyMCMC
    )

    mf = MistFit(path=str(cfg["path"]))
    mf.set_data(keys=["kmag", "parallax"], vals=[10.0, 0.5], errs=[0.1, 0.05])
    mf.setup_hmc(target_accept_prob=0.9, num_warmup=2, num_samples=3)

    # Kernel/MCMC parameters are propagated
    assert isinstance(mf.mcmc, DummyMCMC)
    assert mf.mcmc.kernel.kwargs["target_accept_prob"] == 0.9
    assert mf.mcmc.num_warmup == 2
    assert mf.mcmc.num_samples == 3

    # Running produces stored samples and triggers summary
    rng_key = random.PRNGKey(0)
    mf.run_hmc(rng_key)
    assert mf.mcmc.summary_called
    assert isinstance(mf.samples, pd.DataFrame)
    assert set(mf.samples.columns) == {"age", "kmag"}
    assert len(mf.mcmc.runs) == 1


def test_check_mistgrid_path_skips_download(monkeypatch, synthetic_mistgrid):
    # Simulate presence of grid file to avoid invoking create_mistgrid
    def fake_exists(path):
        if str(path).endswith("mistgrid_iso.npz"):
            return True
        return False

    def fail_create():
        raise AssertionError("create_mistgrid should not be called when grid exists")

    monkeypatch.setattr("jaxstar.mistfit.mistfit.os.path.exists", fake_exists)
    monkeypatch.setattr("jaxstar.mistfit.mistfit.create_mistgrid", fail_create)

    # Should return the default path without calling the downloader
    path = check_mistgrid_path()
    assert path.endswith("mistgrid_iso.npz")


def test_check_mistgrid_path_triggers_download_when_missing(monkeypatch):
    calls = {}

    def fake_exists(path):
        return False

    def fake_create():
        calls["create"] = True

    monkeypatch.setattr("jaxstar.mistfit.mistfit.os.path.exists", fake_exists)
    monkeypatch.setattr("jaxstar.mistfit.mistfit.create_mistgrid", fake_create)

    path = check_mistgrid_path()
    assert calls.get("create") is True
    assert path.endswith("mistgrid_iso.npz")
