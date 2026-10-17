"""FeH ranges select native archives inclusively before acquiring inputs."""

import json

import numpy as np
import pytest

from jaxstar.mistfit.mistgrid import cache, create_tracks
from jaxstar.mistfit.mistgrid.sources import bolometric_correction_archive, track_archives
from test_mist_track_builder import MASSES, bc_files, track_text, write_archive
from test_mist_v25_tracks import bc_v2_files, native_track


FEHS = (-.75, -.5, -.25, 0., .25, .5)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(cache, 'urlopen', lambda *a, **k: pytest.fail('unexpected download'))


@pytest.mark.parametrize('version', ['1.2', '2.5'])
@pytest.mark.parametrize('cli', [False, True])
@pytest.mark.parametrize('bounds, expected', [
    ((-.75, .5), FEHS),  # both endpoints and every intermediate plane
    ((-.6, .3), (-.5, -.25, 0., .25)),  # bounds need not be native values
])
def test_range_builds_selected_native_planes(tmp_path, version, cli, bounds, expected):
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    text = native_track if version == '2.5' else track_text
    bc = bc_v2_files() if version == '2.5' else bc_files()
    write_archive(inputs / bolometric_correction_archive(version=version).filename, bc)
    for feh, source in zip(FEHS, track_archives(version=version, feh=FEHS)):
        write_archive(inputs / source.filename, {
            f'{round(mass * 100):05d}M.track.eep': text(mass, feh) for mass in MASSES})

    output, raw = tmp_path / 'grid.npz', tmp_path / 'raw'
    if cli:
        create_tracks.main([
            str(output), '--version', version, '--mass-range', '.8', '1.', '--eep-range', '2', '4',
            '--feh-range', *map(str, bounds), '--local-dir', str(inputs),
            '--raw-cache-dir', str(raw), '--offline',
        ])
    else:
        create_tracks.create_track_grid(output, version=version, feh_range=np.asarray(bounds),
                                         mass_range=(.8, 1.), eep_range=(2, 4),
                                         local_dir=inputs, raw_cache_dir=raw, offline=True)
    with np.load(output) as data:
        np.testing.assert_array_equal(data['initial_feh'], expected)
        assert data['logage'].shape == (2, len(expected), 3)
        info = json.loads(data['metadata_json'].item())
        assert info['feh_range'] == list(bounds)
        assert info['feh'] == list(expected)
        assert info['native_tracks'] == 2 * len(expected)
        # Only selected FeH archives plus the shared BC archive were acquired.
        assert len(info['inputs']) == len(expected) + 1
        acquired = {r['source']['filename'] for r in info['inputs'] if r['source']['kind'] == 'tracks'}
        assert acquired == {s.filename for s in track_archives(version=version, feh=expected)}

    # A range and an explicit list must give identical numerical fields.
    explicit = create_tracks.create_track_grid(
        tmp_path / 'explicit.npz', version=version, feh=expected,
        mass_range=(.8, 1.), eep_range=(2, 4), raw_cache_dir=raw, offline=True,
    )
    with np.load(output) as a, np.load(explicit) as b:
        assert set(a.files) == set(b.files)
        for name in a.files:
            if name != 'metadata_json':
                np.testing.assert_array_equal(a[name], b[name])


@pytest.mark.parametrize('bounds', [
    (.5, -.75), (0., 0.), (-np.inf, 0.), (0., np.inf), (np.nan, .5),
    (), (-.75,), (-.75, 0., .5), [[-.75], [.5]],
    (.01, .2),  # no published points
    (-.1, .1),  # only FeH=0, insufficient for a rectangular grid
])
def test_bad_ranges_fail_before_io(tmp_path, monkeypatch, bounds):
    monkeypatch.setattr(create_tracks, 'fetch_archive', lambda *a, **k: pytest.fail('unexpected acquisition'))
    with pytest.raises(ValueError, match='feh_range'):
        create_tracks.create_track_grid(tmp_path / 'grid.npz', feh_range=bounds, version='2.5')
    assert not list(tmp_path.iterdir())


def test_list_and_range_cannot_be_combined_in_api_or_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(create_tracks, 'fetch_archive', lambda *a, **k: pytest.fail('unexpected acquisition'))
    with pytest.raises(ValueError, match='mutually exclusive'):
        create_tracks.create_track_grid(tmp_path / 'grid.npz', feh=[-.25, 0.], feh_range=(-.75, .5))

    monkeypatch.setattr(create_tracks, 'create_track_grid', lambda *a, **k: pytest.fail('unexpected build'))
    with pytest.raises(SystemExit) as error:
        create_tracks.main([str(tmp_path / 'grid.npz'), '--feh', '-.25', '0.',
                            '--feh-range', '-.75', '.5'])
    assert error.value.code == 2
    assert 'not allowed with argument' in capsys.readouterr().err
    assert not list(tmp_path.iterdir())
