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
