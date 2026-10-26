"""Offline Korg orchestration tests. No Julia import, downloads or synthesis."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from jaxstar.specfit import load_spectral_grid, prepare_korg, save_spectral_grid
from jaxstar.specfit import korg
from jaxstar.specfit.numpyro_model import empirical_vmic


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("prepare_korg_script", ROOT / "scripts/specfit/prepare_grid.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


@pytest.fixture
def cli_builder():
    """Restore the global precision flag after calling the CLI in-process."""
    previous = jax.config.x64_enabled
    try:
        yield builder
    finally:
        # The CLI sets a process-wide default; a thread-local x64 context alone
        # does not undo that update on newer JAX versions.
        jax.config.update("jax_enable_x64", previous)


class FakeRuntime:
    def __init__(self, path):
        self.line_path = path
        path.write_text("fake offline lines")
        self.provenance = {"versions": {"korg": "fake"}, "linelist_hash": "one"}
        self.coverage = np.array([4000., 9000.])
        self.calls, self.preflight = [], []
        self.fail_at = None
        self.bad_atmosphere = None

    def to_vacuum(self, wavelength, medium):
        return np.asarray(wavelength) + (1. if medium == "air" else 0.)

    def atmosphere(self, point):
        self.preflight.append(point)
        if point["teff"] == self.bad_atmosphere:
            raise ValueError("outside MARCS")
        return point

    def load_lines(self):
        pass

    def windows(self, intervals, step):
        return intervals, step

    def synthesize(self, point, windows, vmic, line_buffer):
        self.calls.append(dict(point=point, vmic=vmic, line_buffer=line_buffer))
        if len(self.calls) == self.fail_at:
            raise RuntimeError("interrupted synthesis")
        intervals, step = windows
        wave = np.concatenate([np.arange(lo, hi + step / 10, step) for lo, hi in intervals])
        normalized = .8 + .0001 * (wave - 5000) + .00001 * (point["teff"] - 6000) + .01 * vmic
        return wave, 2 * normalized, np.full_like(wave, 2.)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    runtime = FakeRuntime(tmp_path / "lines.dat")
    monkeypatch.setattr(korg, "_get_runtime", lambda: runtime)
    return runtime


@pytest.fixture
def settings():
    return dict(axes=dict(teff=[5750., 6000.], logg=[4.25], feh=[0.], alpha=[.2]),
                wavelength_ranges=[[5000, 5002], [5001, 5003]], regions=[9, 3],
                wavelength_medium="air", pixels=101)


def test_normalization_medium_empirical_vmic_and_common_storage(runtime, settings, tmp_path, x64_context):
    with x64_context():
        spectra, report = prepare_korg(**settings, return_report=True)
        assert spectra.library == "korg"
        assert spectra.grid.axis_names == ("teff", "logg", "feh", "alpha")
        assert spectra.regions == (9, 3) and spectra.flux_kind == "normalized"
        np.testing.assert_allclose(spectra.wavelength[:, [0, -1]], settings["wavelength_ranges"])
        assert len(report["contract"]["vacuum_intervals_A"]) == 1  # overlapping regions merged
        for call in runtime.calls:
            point = call["point"]
            vmic = float(empirical_vmic(point["teff"], point["logg"], point["feh"]))
            assert call["vmic"] == vmic and vmic != 1.
            expected = (.8 + .0001 * (np.asarray(spectra.wavelength) + 1. - 5000)
                        + .00001 * (point["teff"] - 6000) + .01 * vmic)
            np.testing.assert_allclose(spectra.grid.interpolate(point)["flux"], expected, atol=1e-12)
        save_spectral_grid(tmp_path / "grid.npz", spectra)
        restored = load_spectral_grid(tmp_path / "grid.npz")
        np.testing.assert_array_equal(restored.grid.field("flux").values, spectra.grid.field("flux").values)


def test_preflight_rejects_before_any_synthesis(runtime, settings, tmp_path, x64_context):
    runtime.bad_atmosphere = 6000.
    with x64_context(), pytest.raises(ValueError, match="MARCS interpolation failed.*6000"):
        prepare_korg(**settings, cache_dir=tmp_path / "cache")
    assert not runtime.calls and not (tmp_path / "cache").exists()


def test_interrupted_build_resumes_and_settings_invalidate_cache(runtime, settings, tmp_path, x64_context):
    cache = tmp_path / "cache"
    runtime.fail_at = 2
    with x64_context():
        with pytest.raises(ValueError, match="node failed.*6000.*interrupted synthesis"):
            prepare_korg(**settings, cache_dir=cache)
        assert len(list(cache.glob("*/node_*.npz"))) == 1
        runtime.fail_at = None
        spectra, resumed = prepare_korg(**settings, cache_dir=cache, return_report=True)
        assert [n["cached"] for n in resumed["nodes"]] == [True, False]
        assert len(runtime.calls) == 3
        again, report = prepare_korg(**settings, cache_dir=cache, return_report=True)
        assert all(n["cached"] for n in report["nodes"]) and len(runtime.calls) == 3
        np.testing.assert_array_equal(spectra.grid.field("flux").values, again.grid.field("flux").values)
        for changes in ({"synthesis_step": .00125}, {"line_buffer": 20.}, {"pixels": 102},
                        {"wavelength_medium": "vacuum"},
                        {"axes": {**settings["axes"], "alpha": [0.]}}):
            _, changed = prepare_korg(**{**settings, **changes}, cache_dir=cache, return_report=True)
            assert changed["signature"] != report["signature"]
            assert not any(n["cached"] for n in changed["nodes"])
        runtime.provenance = {**runtime.provenance, "linelist_hash": "changed"}
        _, changed = prepare_korg(**settings, cache_dir=cache, return_report=True)
        assert changed["signature"] != report["signature"]


def test_corrupt_cache_is_an_error(runtime, settings, tmp_path, x64_context):
    with x64_context():
        _, report = prepare_korg(**settings, cache_dir=tmp_path / "cache", return_report=True)
        path = Path(report["cache_path"]) / "node_0000000.npz"
        with np.load(path) as data:
            changed = dict(data)
        changed["flux"] = changed["flux"] + .1
        np.savez(path, **changed)
        with pytest.raises(ValueError, match="Invalid Korg cache node.*checksum"):
            prepare_korg(**settings, cache_dir=tmp_path / "cache")
        assert len(runtime.calls) == 2


@pytest.mark.parametrize("change,match", [
    ({"normalized": False}, "normalized=true"),
    ({"synthesis_step": 0.}, "synthesis_step"),
    ({"line_buffer": float("nan")}, "line_buffer"),
    ({"wavelength_medium": "unknown"}, "explicitly"),
    ({"regions": [0]}, "one string/integer"),
    ({"sampling": "native"}, "sampling"),
    ({"velocity_step": 1.}, "exactly one"),
    ({"pixels": None}, "exactly one"),
    ({"pixels": True}, "pixels must be an integer"),
    ({"sampling": "linear", "velocity_step": 1.}, "requires sampling='log'"),
    ({"axes": dict(teff=[6000], logg=[4], feh=[0], alpha=[0], vmic=[1])}, "extra abundance"),
])
def test_bad_inputs_fail_before_julia(settings, change, match, monkeypatch, x64_context):
    def unexpected():
        pytest.fail("Julia must not be initialized for invalid settings")
    monkeypatch.setattr(korg, "_get_runtime", unexpected)
    with x64_context(), pytest.raises(ValueError, match=match):
        prepare_korg(**{**settings, **change})


def test_vmic_invalid_before_julia(settings, monkeypatch, x64_context):
    monkeypatch.setattr(korg, "empirical_vmic", lambda *args: -1.)
    monkeypatch.setattr(korg, "_get_runtime", lambda: pytest.fail("unexpected Julia"))
    with x64_context(), pytest.raises(ValueError, match="Invalid empirical vmic"):
        prepare_korg(**settings)


def test_line_coverage_includes_padding_and_resampling_does_not_extrapolate(runtime, settings, x64_context):
    runtime.coverage = np.array([5000., 5005.])
    with x64_context(), pytest.raises(ValueError, match="line_buffer exceed GES"):
        prepare_korg(**settings)
    assert not runtime.calls
    with pytest.raises(ValueError, match="no extrapolation"):
        korg._resample(([5001., 5002.], [2., 2.], [2., 2.]), np.array([[5000., 5003.]]), [[4999., 5004.]])


def test_abundance_mapping_and_marcs_arguments():
    backend = object.__new__(korg._Runtime)
    backend.solar = np.zeros(92)
    backend.alpha = np.array([8, 10, 12, 14, 16, 18, 20, 22])
    calls = []

    def abundances(metals_h, alpha_h, *, solar_abundances):
        calls.append((metals_h, alpha_h))
        assert solar_abundances is backend.solar
        result = np.full(92, metals_h)
        result[backend.alpha - 1] = alpha_h
        return result

    def marcs(teff, logg, A_X, **kwargs):
        assert kwargs == dict(clamp_abundances=False, perturb_at_grid_values=False)
        return "atmosphere"

    backend.korg = SimpleNamespace(format_A_X=abundances, interpolate_marcs=marcs)
    atmosphere, A_X = backend.atmosphere(dict(teff=6000., logg=4., feh=-.5, alpha=.2))
    assert atmosphere == "atmosphere" and calls == [(-.5, -.3)]
    np.testing.assert_allclose(A_X[backend.alpha - 1] - A_X[25], .2)


def test_linelist_setup_failure_can_be_retried():
    backend = object.__new__(korg._Runtime)
    backend.lines = None
    calls = []

    def load(**kwargs):
        calls.append(kwargs)
        return object()

    def failed_path(*args):
        raise RuntimeError("provenance read failed")

    backend.korg = SimpleNamespace(get_GES_linelist=load)
    backend.jl = SimpleNamespace(pathof=failed_path)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="provenance read failed"):
            backend.load_lines()
        assert backend.lines is None
    assert len(calls) == 2


def recipe_file(tmp_path):
    table = tmp_path / "ranges.csv"
    table.write_text("region,wavmin,wavmax,unit,medium\n3,5000,5002,angstrom,air\n")
    recipe = dict(library="korg", axes=dict(teff=[5750, 6000], logg=[4.25], feh=[0], alpha=[0]),
                  wavelength_ranges_file="ranges.csv", wav_margin=.1, sampling=dict(pixels=101),
                  normalized=True)
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    return path, recipe


def test_script_dry_run_build_and_resume(runtime, tmp_path, capsys, cli_builder):
    path, recipe = recipe_file(tmp_path)
    output = tmp_path / "bundle"
    cli_builder.main([str(path), "--dry-run"])
    assert json.loads(capsys.readouterr().out)["preparation"]["synthesis_step"] == .0025
    assert not runtime.calls
    runtime.fail_at = 2
    with pytest.raises(ValueError, match="interrupted synthesis"):
        cli_builder.build(path, None, output)
    assert not output.exists() and (tmp_path / "bundle.cache").is_dir()
    runtime.fail_at = None
    cli_builder.build(path, None, output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert [n["cached"] for n in manifest["korg"]["nodes"]] == [True, False]
    assert cli_builder.read_recipe(output / "recipe.json")[2] == manifest["resolved_preparation"]
    for name, record in manifest["outputs"].items():
        assert korg._digest_file(output / name) == record["sha256"]
    with pytest.raises(FileExistsError):
        cli_builder.build(path, None, output)


@pytest.mark.parametrize("changes", [{"normalized": False}, {"korg": {"linelist": "GALAH"}},
                                     {"korg": {"synthesis_step": -1}}, {"missing_nodes": "logg"}])
def test_recipe_rejects_unsupported_physics(tmp_path, changes):
    path, recipe = recipe_file(tmp_path)
    path.write_text(json.dumps({**recipe, **changes}))
    with pytest.raises(ValueError):
        builder.read_recipe(path)


def test_common_grid_loading_and_dry_run_without_juliacall(runtime, settings, tmp_path, x64_context):
    with x64_context():
        spectra = prepare_korg(**settings)
        save_spectral_grid(tmp_path / "grid.npz", spectra)
    path, _ = recipe_file(tmp_path)
    code = """
import sys, runpy
sys.modules['juliacall'] = None  # simulate an installation without the optional extra
import jax
jax.config.update('jax_enable_x64', True)
from jaxstar.specfit import load_spectral_grid, prepare_korg
grid = load_spectral_grid(sys.argv[1])
assert grid.library == 'korg'
script, recipe = sys.argv[2:]
sys.argv = [script, recipe, '--dry-run']
runpy.run_path(script, run_name='__main__')
try:
    prepare_korg(axes={'teff': [6000], 'logg': [4], 'feh': [0], 'alpha': [0]},
                 wavelength_ranges=[[5000, 5002]], wavelength_medium='air', pixels=101)
except ImportError as error:
    assert 'optional dependency' in str(error)
else:
    raise AssertionError('generation must require JuliaCall')
"""
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path / "grid.npz"),
                             str(ROOT / "scripts/specfit/prepare_grid.py"), str(path)],
                            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
