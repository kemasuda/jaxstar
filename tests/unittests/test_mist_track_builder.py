"""Synthetic native archives test builder values, coverage, provenance and I/O."""

import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest

from jaxstar.mistfit import MistModel
from jaxstar.mistfit.mistgrid import cache, create_tracks
from jaxstar.mistfit.mistgrid import _tracks_v12 as v12
from jaxstar.mistfit.mistgrid.sources import bolometric_correction_archive, track_archives


FEHS = (-.25, .25)
MASSES = (.8, 1.)
OPTIONS = dict(mass_range=(.8, 1.), eep_range=(2, 4), feh=FEHS, allow_missing=False)


def write_archive(path, files):
    with tarfile.open(path, "w:xz") as archive:
        for name, text in files.items():
            payload = text.encode("ascii")
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


def track_text(mass, feh, *, rotation=.4, end=5, bad_age=False, bad_temperature=False):
    lines = ["# MIST version number = 1.2", "# Yinit Zinit [Fe/H] [a/Fe] v/vcrit",
             f"# .27 .014 {feh} 0.0 {rotation}",
             "# initial_mass N_pts N_EEP N_col phase type",
             f"# {mass} {end} 2 8 YES low-mass", f"# EEPs: 1 {end}",
             "# star_age star_mass log_Teff log_g log_L log_R log_surf_z surface_h1"]
    for eep in range(1, end+1):
        age = 1e8*(eep+eep**2/5)/mass**2
        if bad_age and eep == 3:
            age = 1.
        temperature = 3.73+.005*eep
        if bad_temperature and eep == 3:
            temperature = 5.
        values = (age, mass-.01, temperature, 4.15, .2*eep+.1*mass,
                  .1*eep, np.log10(.0134)+feh, .7)
        lines.append(" ".join(format(value, ".17g") for value in values))
    return "\n".join(lines)+"\n"


def bc_files():
    files = {}
    for z in (-1., -.5, 0., .5, 1.):
        lines = ["# Teff logg [Fe/H] Av Rv " + " ".join(v12.BANDS)]
        for temperature in (4000., 5000., 6000., 7000., 8000.):
            for gravity in (3., 4., 5.):
                # Globally affine: every supported Steffen/linear combination
                # has this analytic answer, including temperature edge cells.
                bands = z*.4 + np.log10(temperature)*.7 + gravity*.2 + np.arange(14)*.03
                values = (temperature, gravity, z, 0., 3.1, *bands)
                lines.append(" ".join(format(v, ".17g") for v in values))
        files[f"bc_{z}.UBVRIplus"] = "\n".join(lines)+"\n"
    return files


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(cache, "urlopen", lambda *a, **k: pytest.fail("tests must never download"))


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    write_archive(root / bolometric_correction_archive().filename, bc_files())
    for z, source in zip(FEHS, track_archives(feh=FEHS)):
        write_archive(root / source.filename, {
            f"tracks/{round(m*100):05d}M.track.eep": track_text(m, z) for m in MASSES})
    return root


def build(tmp_path, inputs, **options):
    return create_tracks.create_track_grid(tmp_path / "grid.npz", **{
        **OPTIONS, "local_dir": inputs, "raw_cache_dir": tmp_path / "raw", "offline": True, **options})


def test_builder_values_metadata_and_loader(inputs, tmp_path, x64_context):
    path = build(tmp_path, inputs)
    with np.load(path, allow_pickle=False) as grid:
        np.testing.assert_array_equal(grid["initial_mass"], MASSES)
        np.testing.assert_array_equal(grid["initial_feh"], FEHS)
        np.testing.assert_array_equal(grid["eep"], [2., 3., 4.])
        mass, feh, eep = np.meshgrid(MASSES, FEHS, [2., 3., 4.], indexing="ij")
        age = .1*(eep+eep**2/5)/mass**2
        np.testing.assert_allclose(grid["age_gyr"], age, rtol=1e-14)
        np.testing.assert_allclose(grid["logage"], np.log10(age)+9, rtol=1e-14)
        np.testing.assert_allclose(grid["bc_metallicity"], feh, atol=1e-15)
        np.testing.assert_allclose(grid["cmd_surface_feh"], feh-np.log10(.7/.7381), atol=1e-15)
        expected_bc = feh[..., None]*.4 + (.7*(3.73+.005*eep)+4.15*.2)[..., None] + np.arange(14)*.03
        expected_mag = 4.74-2.5*(.2*eep+.1*mass)[..., None]-expected_bc
        np.testing.assert_allclose(grid["absolute_magnitudes"], expected_mag, atol=3e-15)
        info = json.loads(grid["metadata_json"].item())
        assert info["kind"] == "track" and info["mist_version"] == "1.2"
        assert info["native_tracks"] == 4 and info["shape"] == [2, 2, 3]
        assert len(info["inputs"]) == 3
        for record in info["inputs"]:
            assert len(record["sha256"]) == 64
            assert record["size_bytes"] == Path(record["path"]).stat().st_size
        assert set(info["implementation_sha256"]) == {"create_tracks.py", "_tracks_v12.py"}
    with x64_context():
        model = MistModel.from_tracks(path)
        assert "metadata_json" not in model.output_keys
        params = model({"mass_init": 1., "feh_init": .25, "eep": 3.},
                       keys=("age", "dtdeep", "jmag_abs"))
        np.testing.assert_allclose(params["age"], .48, rtol=1e-14)
        np.testing.assert_allclose(params["dtdeep"], .22, rtol=1e-14)
        np.testing.assert_allclose(params["jmag_abs"], expected_mag[1, 1, 1, 5], atol=3e-15)


def test_rebuild_from_cache_without_local_sources_or_network(inputs, tmp_path):
    original = build(tmp_path, inputs, feh=(.25, -.25))
    second = create_tracks.create_track_grid(tmp_path / "second.npz", **OPTIONS,
                                             raw_cache_dir=tmp_path / "raw", offline=True)
    with np.load(original) as a, np.load(second) as b:
        assert set(a.files) == set(b.files)
        for name in a.files:
            np.testing.assert_array_equal(a[name], b[name])


def test_both_rotations_are_parsed(inputs, tmp_path):
    for z, source in zip(FEHS, track_archives(feh=FEHS, rotation=0.)):
        write_archive(inputs / source.filename, {
            f"{round(m*100):05d}M.track.eep": track_text(m, z, rotation=0.) for m in MASSES})
    with np.load(build(tmp_path, inputs, rotation=0.)) as grid:
        assert json.loads(grid["metadata_json"].item())["rotation"] == 0.


@pytest.mark.parametrize("case", ["short", "age", "BC", "mass", "composition", "version", "column", "duplicate", "missing"])
def test_bad_track_inputs_fail_with_context_and_preserve_output(inputs, tmp_path, case):
    output = tmp_path / "grid.npz"
    output.write_bytes(b"existing grid")
    files = {f"{round(m*100):05d}M.track.eep": track_text(m, FEHS[0]) for m in MASSES}
    name = "00100M.track.eep"
    if case == "short":
        files[name] = track_text(1., FEHS[0], end=3)
    elif case == "age":
        files[name] = track_text(1., FEHS[0], bad_age=True)
    elif case == "BC":
        files[name] = track_text(1., FEHS[0], bad_temperature=True)
    elif case == "mass":
        files[name] = track_text(.9, FEHS[0])
    elif case == "composition":
        files[name] = track_text(1., 0.)
    elif case == "version":
        files[name] = files[name].replace("version number = 1.2", "version number = 2.5")
    elif case == "column":
        files[name] = files[name].replace("surface_h1", "wrong_column")
    elif case == "duplicate":
        files["duplicate/"+name] = files[name]
    else:
        del files[name]
    write_archive(inputs / track_archives(feh=FEHS)[0].filename, files)
    with pytest.raises(ValueError) as error:
        build(tmp_path, inputs, overwrite=True)
    assert ("mass=1" in str(error.value) or "Missing native mass/FeH combinations" in str(error.value))
    assert output.read_bytes() == b"existing grid"
    assert not list(tmp_path.glob("*.part"))


def test_interp_alternatives_are_not_mixed_with_native_tracks(inputs, tmp_path):
    source = track_archives(feh=FEHS)[0]
    files = {f"{round(m*100):05d}M.track.eep": track_text(m, FEHS[0]) for m in MASSES}
    files["00090M.track.eep_INTERP"] = "not a native track"
    files["README"] = "Archive notes"
    write_archive(inputs / source.filename, files)
    with np.load(build(tmp_path, inputs)) as grid:
        np.testing.assert_array_equal(grid["initial_mass"], MASSES)


@pytest.mark.parametrize("options", [{"mass_range": (1., .8)}, {"mass_range": (0., 1.)},
                                    {"eep_range": (2., 3.)}, {"eep_range": (2.1, 4.1)},
                                    {"feh": [0.]}, {"feh": [-.25, -.25]},
                                    {"version": "2.5"}, {"rotation": .2}, {"afe": .2}])
def test_invalid_options_fail_before_acquiring_inputs(tmp_path, monkeypatch, options):
    monkeypatch.setattr(create_tracks, "fetch_archive", lambda *a, **k: pytest.fail("unexpected acquisition"))
    with pytest.raises(ValueError):
        create_tracks.create_track_grid(tmp_path / "grid.npz", **{**OPTIONS, **options})
    assert not list(tmp_path.iterdir())


def test_existing_output_is_not_overwritten_or_downloaded(inputs, tmp_path, monkeypatch):
    (tmp_path / "grid.npz").write_bytes(b"old")
    monkeypatch.setattr(create_tracks, "fetch_archive", lambda *a, **k: pytest.fail("unexpected acquisition"))
    with pytest.raises(FileExistsError):
        build(tmp_path, inputs)
    assert (tmp_path / "grid.npz").read_bytes() == b"old"


def test_write_failure_leaves_old_grid_intact(inputs, tmp_path, monkeypatch):
    (tmp_path / "grid.npz").write_bytes(b"old")

    def broken_save(stream, **arrays):
        stream.write(b"partial NPZ")
        raise OSError("disk full")

    monkeypatch.setattr(create_tracks.np, "savez_compressed", broken_save)
    with pytest.raises(OSError, match="disk full"):
        build(tmp_path, inputs, overwrite=True)
    assert (tmp_path / "grid.npz").read_bytes() == b"old"
    assert not list(tmp_path.glob("*.part"))


def test_cli_builds_a_grid(inputs, tmp_path):
    output = tmp_path / "cli.npz"
    create_tracks.main([str(output), "--mass-range", ".8", "1.", "--eep-range", "2", "4",
                        "--feh", "-.25", ".25", "--local-dir", str(inputs),
                        "--raw-cache-dir", str(tmp_path / "raw"), "--offline"])
    with np.load(output) as grid:
        assert grid["logage"].shape == (2, 2, 3)


def test_bc_temperature_edges_and_missing_columns(inputs, tmp_path):
    path = inputs / bolometric_correction_archive().filename
    axes, fields = v12.read_bc_archive(path)
    queries = np.array([[.1, np.log10(t), 4.2] for t in (4000., 4500., 5500., 7500., 8000.)])
    expected = .4*queries[:, :1]+.7*queries[:, 1:2]+.2*queries[:, 2:3]+np.arange(14)*.03
    np.testing.assert_allclose(v12.track_bc(axes, fields, queries), expected, atol=2e-15)
    files = bc_files()
    key = next(iter(files))
    files[key] = files[key].replace("2MASS_J", "unknown_band")
    write_archive(path, files)
    with pytest.raises(ValueError, match="Missing UBVRIplus bands"):
        v12.read_bc_archive(path)


@pytest.mark.parametrize("metadata", ['not JSON', '[]', '{"kind":"isochrone","format_version":1}',
                                     '{"kind":"track","format_version":99}'])
def test_loader_rejects_invalid_metadata(inputs, tmp_path, x64_context, metadata):
    path = build(tmp_path, inputs)
    with np.load(path) as data:
        fields = dict(data)
    np.savez(path, **{**fields, "metadata_json": np.array(metadata)})
    with x64_context(), pytest.raises(ValueError, match="metadata"):
        MistModel.from_tracks(path)


def test_default_full_extent_retains_ragged_tracks_missing_pairs_and_bc(inputs, tmp_path, x64_context):
    sources = track_archives(feh=FEHS)
    write_archive(inputs/sources[0].filename, {
        '00080M.track.eep': track_text(.8, FEHS[0], end=3),
        '00100M.track.eep': track_text(1., FEHS[0], bad_temperature=True),
    })
    write_archive(inputs/sources[1].filename, {'00100M.track.eep': track_text(1., FEHS[1], bad_age=True)})
    path = create_tracks.create_track_grid(tmp_path/'all.npz', feh=FEHS, local_dir=inputs,
                                           raw_cache_dir=tmp_path/'raw', offline=True)
    with np.load(path) as data:
        np.testing.assert_array_equal(data['eep'], np.arange(1., 6.))
        present = data['track_present']
        assert present.dtype == bool and present.shape == (2, 2, 5)
        assert present.sum() == 13
        assert not present[0, 1].any()
        assert np.isnan(data['age_gyr'][0, 0, 3:]).all()
        assert np.isnan(data['absolute_magnitudes'][1, 0, 2]).all()
        assert np.isfinite(data['logTeff'][1, 0, 2])
        info = json.loads(data['metadata_json'].item())
        assert info['format_version'] == 2
        assert info['coverage']['missing_mass_feh'] == [[.8, .25]]
        assert info['coverage']['present_nodes'] == 13
    with x64_context():
        model = MistModel.from_tracks(path)
        p = dict(mass_init=1., feh_init=-.25, eep=3.)
        pred = model(p, keys=('teff', 'kmag_abs', 'dtdeep'))
        assert np.isfinite(pred['teff']) and np.isfinite(pred['dtdeep'])
        assert np.isneginf(pred['kmag_abs'])
        assert np.isneginf(model(dict(p, feh_init=.25), keys='dtdeep')['dtdeep'])


def test_bc_inclusive_edges_and_monotonic_three_point_recipe(inputs):
    axes, fields = v12.read_bc_archive(inputs/bolometric_correction_archive().filename)
    queries = np.array(np.meshgrid([-1., -.9, -.5, .5, .9, 1.],
                                  [axes[1][0], 3.73, axes[1][-1]], [3., 4.2, 5.],
                                  indexing='ij')).reshape(3, -1).T
    expected = .4*queries[:, :1]+.7*queries[:, 1:2]+.2*queries[:, 2:3]+np.arange(14)*.03
    np.testing.assert_allclose(v12.track_bc(axes, fields, queries), expected, atol=3e-15)
    x = np.array([[0., 1., 2.], [0., 1., 2.]])
    y = np.array([[[0.], [1.], [100.]], [[100.], [1.], [0.]]])
    # Limiting gives slope 0 at the flat outer end, 2 at the middle node.
    np.testing.assert_allclose(v12.steffen_three(x, y, np.array([.5, 1.5])), [[.25], [.25]])
    for coordinate in ([-1.001, 3.73, 4.], [0., 3., 4.], [0., 3.73, 5.001]):
        with pytest.raises(ValueError, match='Unsupported BC'):
            v12.track_bc(axes, fields, np.array([coordinate]))


def test_selected_mass_range_may_be_absent_at_one_feh(inputs, tmp_path):
    write_archive(inputs/track_archives(feh=FEHS)[0].filename,
                  {'00200M.track.eep': track_text(2., FEHS[0])})
    path = build(tmp_path, inputs, allow_missing=True)
    with np.load(path) as data:
        assert not data['track_present'][:, 0].any()
        assert data['track_present'][:, 1].all()
        assert np.isnan(data['age_gyr'][:, 0]).all()
