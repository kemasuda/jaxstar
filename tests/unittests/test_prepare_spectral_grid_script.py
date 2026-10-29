"""Exercise the offline recipe CLI with tiny native BOSZ and Coelho inputs."""

import csv
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from astropy.io import fits
import numpy as np
import pytest

from jaxstar.specfit import load_spectral_grid, SpecModel


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/specfit/prepare_grid.py"
spec = importlib.util.spec_from_file_location("prepare_grid_script", SCRIPT)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def run_cli(tmp_path, *args):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"),
               MPLCONFIGDIR=str(tmp_path / "mpl"))
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)


def inputs(tmp_path, library, normalized=True):
    raw = tmp_path / "raw"
    raw.mkdir()
    axes = dict(teff=[6000, 6250], logg=[4.0])
    wave = 4999. + np.arange(601) * .01
    if library == "bosz":
        axes.update(mh=[0.], alpha=[0.], carbon=[0.], vmic=[0, 1, 2])
    else:
        axes.update(feh=[0.], alpha=[0.])
    for t in axes["teff"]:
        for v in axes.get("vmic", [0]):
            flux = .7 + .02 * (wave-5000.) + .05 * (t-6000.) / 250. + .03 * v
            if library == "bosz":
                path = raw / "m+0.00" / f"bosz2024_mp_t{t}_g+4.0_m+0.00_a+0.00_c+0.00_v{v}_rorig_noresam.txt.gz"
                path.parent.mkdir(exist_ok=True)
                with gzip.open(path, "wt") as stream:
                    np.savetxt(stream, np.column_stack((wave, 2*flux, np.full(len(wave), 2.))))
            else:
                hdu = fits.PrimaryHDU(np.stack((flux, 2*flux)))
                hdu.header["CRVAL1"], hdu.header["CD1_1"] = wave[0], .01
                hdu.writeto(raw / f"{t}_40_p00p00.ms.fits")
    table = tmp_path / "ranges.csv"
    table.write_text("region,wavmin,wavmax,unit,medium\n"
                     "4,5000,5001,angstrom,air\n9,5002,5003,angstrom,air\n")
    recipe = dict(library=library, axes=axes, wavelength_ranges_file="ranges.csv",
                  regions=[9, 4], wav_margin=.2, sampling=dict(pixels=128), normalized=normalized)
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    return raw, path, recipe


@pytest.mark.parametrize("library", ["bosz", "coelho"])
@pytest.mark.parametrize("normalized", [True, False])
def test_cli_build_preserves_native_flux_regions_and_provenance(tmp_path, x64_context, library, normalized):
    raw, recipe_path, recipe = inputs(tmp_path, library, normalized)
    # Exercise both forms through the CLI, NPZ and portable recipe snapshot.
    input_recipe = {**recipe, "axes": dict(recipe["axes"])}
    if normalized:
        input_recipe["axes"]["teff"] = {"min": 6000, "max": 6250, "step": 250}
    recipe_path.write_text(json.dumps(input_recipe))
    output = tmp_path / "shared" / "v1"
    process = run_cli(tmp_path, recipe_path, "--data-dir", raw, "--output-dir", output)
    assert process.returncode == 0, process.stdout + process.stderr
    with x64_context():
        spectra = load_spectral_grid(output / "grid.npz")
        assert spectra.regions == (9, 4)
        assert spectra.library == library
        assert spectra.wavelength_medium == "air"
        assert spectra.flux_kind == ("normalized" if normalized else "unnormalized")
        np.testing.assert_allclose(spectra.wavelength[:, [0, -1]], [[5001.8, 5003.2], [4999.8, 5001.2]])
        point = {name: values[-1] for name, values in recipe["axes"].items()}
        if library == "bosz":
            np.testing.assert_array_equal(spectra.grid.axis("vmic"), [0, 1, 2])
            point["vmic"] = 1.
        expected = .75 + .02 * (np.asarray(spectra.wavelength)-5000.) + (.03 if library == "bosz" else 0.)
        if not normalized:
            expected *= 2
        np.testing.assert_allclose(spectra.grid.interpolate(point)["flux"], expected, rtol=1e-6)
        assert SpecModel(spectra, vmax=10.).spectra is spectra

    manifest = json.loads((output / "manifest.json").read_text())
    for name, record in manifest["sources"].items():
        assert hashlib.sha256((raw / name).read_bytes()).hexdigest() == record["sha256"]
    assert len(manifest["sources"]) == (6 if library == "bosz" else 2)
    for name, record in manifest["outputs"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == record["sha256"]
    assert manifest["resolved_preparation"]["regions"] == [9, 4]
    assert manifest["resolved_preparation"]["axes"] == recipe["axes"]
    assert json.loads((output / "recipe.json").read_text())["axes"] == input_recipe["axes"]
    _, saved_csv, saved_kwargs = builder.read_recipe(output / "recipe.json")
    assert saved_csv == output / "wavelength_ranges.csv"
    assert saved_kwargs == manifest["resolved_preparation"]

    # Re-running against an existing bundle refuses even if raw input is absent.
    original = (output / "manifest.json").read_bytes()
    again = run_cli(tmp_path, recipe_path, "--data-dir", tmp_path / "absent", "--output-dir", output)
    assert again.returncode != 0 and "already exists" in again.stderr
    assert (output / "manifest.json").read_bytes() == original


def test_dry_run_and_failed_build_do_not_publish(tmp_path):
    raw, recipe_path, recipe = inputs(tmp_path, "bosz")
    recipe["axes"]["teff"] = [6500]  # Missing native node, no fallback.
    recipe_path.write_text(json.dumps(recipe))
    output = tmp_path / "output"
    dry = run_cli(tmp_path, recipe_path, "--output-dir", output, "--dry-run")
    assert dry.returncode == 0, dry.stderr
    assert json.loads(dry.stdout)["preparation"]["axes"]["teff"] == [6500]
    assert not output.exists()
    failed = run_cli(tmp_path, recipe_path, "--data-dir", raw, "--output-dir", output)
    assert failed.returncode != 0 and "t6500" in failed.stderr
    assert not output.exists()


def test_logg_replacement_preview_build_and_manifest(tmp_path, x64_context):
    raw, recipe_path, recipe = inputs(tmp_path, "bosz")
    recipe["axes"].update(teff=[6000], logg=[3.5, 4., 4.5, 5.], vmic=[1])
    recipe["missing_nodes"] = "logg"
    recipe_path.write_text(json.dumps(recipe))
    wave = 4999. + np.arange(601)*.01
    for g in (3., 4.5):
        geometry = "ms" if g <= 3 else "mp"
        path = raw / "m+0.00" / f"bosz2024_{geometry}_t6000_g+{g:.1f}_m+0.00_a+0.00_c+0.00_v1_rorig_noresam.txt.gz"
        flux = .73 + .1*(g-4.) + .02*(wave-5000.)
        with gzip.open(path, "wt") as stream:
            np.savetxt(stream, np.column_stack((wave, flux*2, np.full(wave.size, 2.))))
    output = tmp_path / "prepared"
    dry = run_cli(tmp_path, recipe_path, "--data-dir", raw, "--output-dir", output, "--dry-run")
    assert dry.returncode == 0, dry.stderr
    assert not output.exists()
    assert "will interpolate linearly in logg from 3 and 4" in dry.stderr
    assert "will use logg=4.5 at the requested upper edge" in dry.stderr
    preview = json.loads(dry.stdout)["input_check"]
    assert preview["ready"] and preview["source_file_count"] == 3
    result = run_cli(tmp_path, recipe_path, "--data-dir", raw, "--output-dir", output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.index("Input check passed") < result.stdout.index("Preparing bosz")
    manifest = json.loads((output / "manifest.json").read_text())
    replacements = manifest["source_plan"]["replacements"]
    assert replacements == preview["replacements"]
    assert [r["method"] for r in replacements] == ["linear_logg", "copy_lower_logg"]
    assert [s["weight"] for s in replacements[0]["sources"]] == [.5, .5]
    assert len(manifest["sources"]) == 3
    for name, record in manifest["sources"].items():
        assert hashlib.sha256((raw / name).read_bytes()).hexdigest() == record["sha256"]
    with x64_context():
        spectra = load_spectral_grid(output / "grid.npz")
        flux = np.asarray(spectra.grid.field("flux").values)[0, :, 0, 0, 0, 0]
        expected = .73 + .02*(np.asarray(spectra.wavelength)-5000.)
        np.testing.assert_allclose(flux[0], expected-.05, rtol=1e-6)
        np.testing.assert_allclose(flux[1], expected, rtol=1e-6)
        np.testing.assert_allclose(flux[2], expected+.05, rtol=1e-6)
        np.testing.assert_array_equal(flux[2], flux[3])
        assert set(spectra.sources) == set(manifest["source_plan"]["sources"])


def test_input_check_reports_all_missing_files_without_preparing(tmp_path):
    raw, path, recipe = inputs(tmp_path, "bosz")
    recipe["axes"].update(teff=[6000, 6500, 6750], vmic=[1])
    recipe["missing_nodes"] = "logg"
    path.write_text(json.dumps(recipe))
    output = tmp_path / "absent"
    result = run_cli(tmp_path, path, "--data-dir", raw, "--output-dir", output)
    assert result.returncode != 0
    assert "t6500" in result.stderr and "t6750" in result.stderr
    assert "no spectra were read" in result.stderr
    assert "Preparing bosz" not in result.stdout
    assert not output.exists()


@pytest.mark.parametrize("axis,values,expected", [
    ("teff", {"min": 4000, "max": 7000, "step": 250}, list(range(4000, 7001, 250))),
    ("mh", {"min": -0.3, "max": 0.3, "step": 0.1}, [-0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3]),
    ("carbon", {"min": 0., "max": 0., "step": 0.25}, [0.]),
])
def test_dry_run_expands_inclusive_ranges(tmp_path, axis, values, expected):
    _, path, recipe = inputs(tmp_path, "bosz")
    recipe["axes"][axis] = values
    path.write_text(json.dumps(recipe))
    dry = run_cli(tmp_path, path, "--dry-run")
    assert dry.returncode == 0, dry.stderr
    assert json.loads(dry.stdout)["preparation"]["axes"][axis] == expected


@pytest.mark.parametrize("values,match", [
    ({"min": 4000, "max": 7000}, "exactly min, max and step"),
    ({"min": 4000, "max": 7000, "step": 250, "stop": 7000}, "exactly min, max and step"),
    ({"min": 4000, "max": 7000, "step": 0}, "step > 0"),
    ({"min": 4000, "max": 7000, "step": -250}, "step > 0"),
    ({"min": 7000, "max": 4000, "step": 250}, "max >= min"),
    ({"min": 4000, "max": 7100, "step": 250}, "integer multiple"),
    ({"min": 4000, "max": 7000, "step": "250"}, "finite numbers"),
    ({"min": 4000, "max": 7000, "step": True}, "finite numbers"),
    ({"min": 4000, "max": float("inf"), "step": 250}, "finite numbers"),
])
def test_axis_range_errors_are_caught_before_raw_io(tmp_path, values, match):
    _, path, recipe = inputs(tmp_path, "bosz")
    recipe["axes"]["teff"] = values
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match=match):
        builder.read_recipe(path)


@pytest.mark.parametrize("change,match", [
    ({"wavelength_medium": "vacuum"}, "conflicts"),
    ({"regions": [9, 9]}, "distinct"),
    ({"regions": [99]}, "CSV region"),
    ({"sampling": {"pixels": 5, "velocity_step": 1}}, "exactly one"),
    ({"normalized": "false"}, "true or false"),
    ({"velocity_step": 1}, "unknown recipe keys"),
    ({"missing_nodes": "nearest"}, "missing_nodes"),
])
def test_recipe_errors_are_caught_before_raw_io(tmp_path, change, match):
    _, path, recipe = inputs(tmp_path, "bosz")
    path.write_text(json.dumps({**recipe, **change}))
    with pytest.raises(ValueError, match=match):
        builder.read_recipe(path)


def test_shared_examples_and_unknown_medium(tmp_path):
    root = SCRIPT.parent
    for path in sorted((root / "recipes").glob("*.json")):
        recipe, csv_path, kwargs = builder.read_recipe(path)
        assert csv_path.parent == root / "wavelength_ranges"
        assert kwargs["wavelength_medium"] in {"air", "vacuum"}
        if recipe["library"] == "bosz":
            assert 1 in kwargs["axes"]["vmic"]
    for path in sorted((root / "wavelength_ranges").glob("*.csv")):
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        assert rows and all(row["unit"] == "angstrom" for row in rows)
        assert len({row["region"] for row in rows}) == len(rows)
    # An ambiguous legacy table must not silently inherit a preparer's default.
    recipe = json.loads((root / "recipes/coelho_gaoes.json").read_text())
    recipe.update(wavelength_ranges_file=str(root / "wavelength_ranges/kpf_red.csv"))
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match="medium is unknown"):
        builder.read_recipe(path)
    recipe["wavelength_medium"] = "vacuum"
    path.write_text(json.dumps(recipe))
    assert builder.read_recipe(path)[2]["wavelength_medium"] == "vacuum"


@pytest.mark.parametrize("normalized", [True, False])
def test_tlusty_ob2025_cli_preflight_build_and_portable_bundle(tmp_path, x64_context, normalized):
    raw = tmp_path / "raw"
    wave = np.linspace(4998., 5005., 701)
    for family, t, value in (("BSTAR", 29000, .7), ("OSTAR", 30000, .9)):
        directory = raw / f"{family}_z100v2"
        directory.mkdir(parents=True)
        for kind, flux in (("spec", 2*value), ("cont", 2.)):
            with gzip.open(directory / f"z100t{t}g400v2.{kind}.gz", "wt") as stream:
                np.savetxt(stream, np.column_stack((wave, np.full(wave.size, flux))))
    table = tmp_path / "ranges.csv"
    table.write_text("region,wavmin,wavmax,unit,medium\n0,5000,5003,angstrom,vacuum\n")
    recipe = dict(library="tlusty", axes=dict(teff=[29000, 30000], logg=[3., 4.], logZ=[0.]),
                  wavelength_ranges_file="ranges.csv", wav_margin=.1,
                  sampling=dict(velocity_step=3.), normalized=normalized,
                  missing_nodes="lowest_logg")
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    output = tmp_path / "bundle"
    dry = run_cli(tmp_path, path, "--data-dir", raw, "--dry-run")
    assert dry.returncode == 0, dry.stderr
    preview = json.loads(dry.stdout)
    assert preview["input_check"]["source_file_count"] == 4
    assert preview["input_check"]["requested_nodes"] == 4
    assert len(preview["input_check"]["replacements"]) == 2
    assert preview["preparation"]["missing_nodes"] == "lowest_logg"
    assert "will use lowest available logg=4" in dry.stderr
    assert preview["preparation"]["source_format"] == "ob2025"
    assert "normalized" not in preview["preparation"]
    assert not output.exists()
    result = run_cli(tmp_path, path, "--data-dir", raw, "--output-dir", output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.index("Input check passed") < result.stdout.index("Preparing tlusty")
    assert result.stdout.index("will use lowest available logg=4") < result.stdout.index("Preparing tlusty")
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["sources"]) == 4
    assert manifest["source_plan"]["replacements"] == preview["input_check"]["replacements"]
    assert "_tlusty.py" in manifest["implementations"]
    assert "_data.py" in manifest["implementations"]
    assert builder.read_recipe(output / "recipe.json")[2] == manifest["resolved_preparation"]
    with x64_context():
        spectra = load_spectral_grid(output / "grid.npz")
        assert spectra.provenance == manifest["provenance"]
        assert spectra.provenance["replacements"] == manifest["source_plan"]["replacements"]
        assert spectra.provenance["atmosphere_vmic_kms"]["OSTAR"] == 10.
        assert spectra.grid.axis_names == ("teff", "logg", "logZ")
        for i, value in enumerate((.7, .9)):
            np.testing.assert_allclose(spectra.grid.field("flux").values[i],
                                       value if normalized else 2*value)

    # Missing both a continuum and another atmosphere: one preflight failure,
    # no flux reads/build message and no partially published bundle.
    (raw / "BSTAR_z100v2/z100t29000g400v2.cont.gz").unlink()
    recipe["axes"]["teff"].append(32500)
    path.write_text(json.dumps(recipe))
    failed = run_cli(tmp_path, path, "--data-dir", raw, "--output-dir", tmp_path / "failed")
    assert failed.returncode != 0
    assert "z100t29000g400v2.cont.gz" in failed.stderr
    assert "z100t32500g400v2.spec.gz" in failed.stderr
    assert "no spectra were read" in failed.stderr
    assert "Preparing tlusty" not in failed.stdout
    assert not (tmp_path / "failed").exists()
