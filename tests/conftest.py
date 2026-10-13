"""Precision context for spectral/payload tests across supported JAX versions."""

from contextlib import contextmanager

import jax
import pytest


@pytest.fixture(scope="session")
def x64_context():
    @contextmanager
    def context(enabled=True):
        if hasattr(jax, "enable_x64"):
            # Public context API used by JAX 0.11.
            with jax.enable_x64(enabled):
                yield
        else:
            # Older supported JAX (including the CPU validation environment).
            # Use only public configuration APIs, not removed experimental ones.
            previous = jax.config.x64_enabled
            jax.config.update("jax_enable_x64", enabled)
            try:
                yield
            finally:
                jax.config.update("jax_enable_x64", previous)

    return context


@pytest.fixture
def arviz_from_dict():
    """Build external posterior data using either installed ArviZ API."""
    import arviz as az

    def convert(*, coords=None, dims=None, **groups):
        if int(az.__version__.split(".", 1)[0]) >= 1:
            return az.from_dict(groups, coords=coords, dims=dims)
        return az.from_dict(**groups, coords=coords, dims=dims)

    return convert


@pytest.fixture
def arviz_data_type():
    import arviz as az
    if int(az.__version__.split(".", 1)[0]) >= 1:
        from xarray import DataTree
        return DataTree
    return az.InferenceData
