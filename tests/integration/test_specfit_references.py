"""Read-only characterization of frozen CCF/clipping routines, without importing them."""

import ast
import os
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.interpolate import interp1d
from scipy.signal import correlate, medfilt
from scipy.stats import median_abs_deviation

from jaxstar.grid import Field, RectilinearGrid
from jaxstar.specfit import Observation, SpecFit, SpecModel, single_star_params
from jaxstar.specfit._data import _SpectralLibrary
from jaxstar.specfit.fit import _compute_ccf, _extend_mask

ROOT = os.environ.get("JAXSPEC_REFERENCE_ROOT")
pytestmark = pytest.mark.skipif(not ROOT, reason="set JAXSPEC_REFERENCE_ROOT for frozen SpecFit parity")


def frozen_function(file, name, namespace):
    with warnings.catch_warnings():
        # Unrelated legacy docstring escape warnings arise while parsing the
        # whole frozen module; only the selected function is executed below.
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse((Path(ROOT) / "src/jaxspec" / file).read_text())
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), file, "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_frozen_compute_ccf(dtype):
    namespace = dict(np=np, interp1d=interp1d, correlate=correlate, c_in_kms=299792.458)
    old = frozen_function("utils.py", "compute_ccf", namespace)
    velocity = np.linspace(-250., 250., 701)
    source = (5000 * np.exp(velocity / 299792.458)).astype(dtype)
    template = (1 - .4 * np.exp(-.5 * (velocity / 4.)**2)).astype(dtype)
    wave = (5000 * np.exp(np.linspace(-120., 120., 96) / 299792.458)).astype(dtype)
    flux = (1 - .4 * np.exp(-.5 * ((np.linspace(-120., 120., 96) - 12.) / 4.)**2)).astype(dtype)
    # Select both fixed and fitted masks before CCF, as in frozen check_ccf.
    usable = np.ones(96, bool)
    usable[[10, 11, 48, 89]] = False
    actual = _compute_ccf(wave[usable], flux[usable], source, template, 5)
    expected = old(wave[usable], flux[usable], source, template)
    for new, reference in zip(actual, expected):
        np.testing.assert_array_equal(new, reference)


@pytest.mark.parametrize("factor", [0., .5, 1., 2.])
def test_frozen_extension(factor):
    old = frozen_function("specfit.py", "extend_mask", dict(np=np))
    flags = np.random.default_rng(5).random(53) > .7
    flags[[0, -1]] = True
    np.testing.assert_array_equal(_extend_mask(flags, factor), old(flags, factor, return_float=False))


def test_frozen_check_ccf_and_outlier_mask(x64_context):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    namespace = dict(np=np, interp1d=interp1d, medfilt=medfilt,
                     mad=median_abs_deviation, plt=plt)
    namespace["compute_ccf"] = frozen_function("utils.py", "compute_ccf",
        dict(np=np, interp1d=interp1d, correlate=correlate, c_in_kms=299792.458))
    namespace["extend_mask"] = frozen_function("specfit.py", "extend_mask", dict(np=np))
    old_ccf = frozen_function("specfit.py", "check_ccf", namespace)
    old_outliers = frozen_function("specfit.py", "mask_outliers", namespace)
    with x64_context(True):
        velocity = np.linspace(-250., 250., 601)
        source = np.array([5000., 6000.])[:, None] * np.exp(velocity / 299792.458)
        payload = np.broadcast_to(1 - .4 * np.exp(-.5 * (velocity / 4.)**2), (1, 2, 601)).copy()
        grid = RectilinearGrid(axes={"teff": np.array([5800.])},
            fields={"flux": Field(payload, ("teff",), payload_dims=("region", "pixel"))})
        model = SpecModel(_SpectralLibrary(grid, source, (8, 9), "fixture"), vmax=30.)
        wave = np.array([5000., 6000.])[:, None] * np.exp(np.linspace(-120., 120., 96) / 299792.458)
        params = single_star_params({"teff": 5800.}, vsini=6., vmacro=3., q1=.36,
                                    q2=.3, rv=12., resolving_power=70000.)
        physical = np.asarray(model(params, wave))
        masks = np.zeros_like(physical, bool)
        masks[:, 10] = True
        fit = SpecFit(Observation(wave, physical, np.full_like(physical, .01), masks), model)
        extra = np.zeros_like(masks)
        extra[:, 48] = True
        fit.set_fit_mask(extra)
        template = np.asarray(model.intrinsic({"components": ({"atmosphere": {"teff": 5800.}},)}, source))
        sm = SimpleNamespace(wav_obs=wave, flux_obs=physical, error_obs=np.full_like(physical, .01),
                             mask_obs=masks, mask_fit=extra.copy(), wavgrid=source, Norder=2,
                             sg=SimpleNamespace(model="fixture", values=lambda *args: template))
        legacy = SimpleNamespace(sm=sm, orders=(8, 9), vmax=100.)
        expected = old_ccf(legacy, v_limit=50., ccfvmax=40.)
        np.testing.assert_array_equal(fit.check_ccf({"teff": 5800.}, v_limit=50., ccfvmax=40.), expected)
        np.testing.assert_array_equal(fit.ccf["region_rv"], legacy.ccfrvlist)
        # Finite unmasked ordinary data must reproduce the exact legacy flags.
        flux = physical + .001 * np.random.default_rng(9).normal(size=physical.shape)
        flux[:, [10, 70]] += .2
        sm.flux_obs = flux
        sm.mask_obs = np.zeros_like(masks)
        sm.mask_fit = np.zeros_like(masks)
        old_outliers(legacy, {"fluxmodel": physical, "vsini": 6.})
        fit = SpecFit(Observation(wave, flux, np.full_like(flux, .01)), model)
        np.testing.assert_array_equal(fit.mask_outliers(prediction=physical, mask_v=6.), sm.mask_fit)
        plt.close("all")
