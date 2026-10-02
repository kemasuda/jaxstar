"""Low-level combined kernel and Gaussian convention checks."""

import jax
import numpy as np
import pytest
from scipy.special import j0

from jaxstar.specfit.broadening import _j0, combined_kernel, gaussian_sigma


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_bessel_and_kernel_precision_normalization_and_symmetry(dtype, x64_context):
    with x64_context():
        arguments = np.linspace(-30, 30, 501).astype(dtype)
        np.testing.assert_allclose(_j0(arguments), j0(arguments), atol=3e-7 if dtype == np.float32 else 3e-15)
        velocity = np.linspace(-50, 50, 101).astype(dtype)
        kernel = combined_kernel(velocity, 3.1, 6.3, .5, .2, gaussian_sigma(70000.))
        np.testing.assert_allclose(kernel.sum(), 1., atol=2e-7)
        np.testing.assert_allclose(kernel, kernel[::-1], atol=2e-7)
        assert np.all(np.isfinite(kernel))


def test_gaussian_sigma_frozen_convention_and_disabled_ip():
    np.testing.assert_allclose(gaussian_sigma(70000.), 299792.458 / 70000. / 2.354820, rtol=1e-7)
    assert gaussian_sigma(np.inf) == 0
