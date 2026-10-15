"""Offline acquisition/reuse tests with small real XZ/tar and CMD fixtures."""

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import importlib
import io
import json
import lzma
from pathlib import Path
import tarfile
from threading import Barrier

import numpy as np
import pytest

from jaxstar.mistfit.mistgrid import cache, paths
from jaxstar.mistfit.mistgrid.sources import (
    MistArchive, bolometric_correction_archive, isochrone_archive, track_archives,
)


def tar_bytes(files=None, member=None):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:xz") as archive:
        for name, content in (files or {"data/track.eep": b"native track\n"}).items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        if member is not None:
            archive.addfile(member)
    return stream.getvalue()


class Response(io.BytesIO):
    def __init__(self, data, length=None):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data) if length is None else length)}

    def geturl(self):
        return "https://example.org/final.txz"


@pytest.fixture
def source():
    return track_archives(feh=[0.])[0]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(cache, "urlopen", lambda *a, **k: pytest.fail("unexpected network access"))


def test_catalogue_paths_and_unsupported_releases():
    sources = track_archives()
    assert len(sources) == 15
    assert sources[0].filename == "MIST_v1.2_feh_m4.00_afe_p0.0_vvcrit0.4_EEPS.txz"
    assert sources[-1].filename == "MIST_v1.2_feh_p0.50_afe_p0.0_vvcrit0.4_EEPS.txz"
    assert len({s.cache_key for s in (*sources, *track_archives(rotation=0.))}) == 30
    assert isochrone_archive().url == "https://mist.science/data/tarballs_v1.2/MIST_v1.2_vvcrit0.4_UBVRIplus.txz"
    bc = bolometric_correction_archive()
    assert bc.url == "https://mist.science/BC_tables/v1/UBVRIplus.txz"
    assert bc.cache_key != replace(bc, version="2.5").cache_key
    assert bc.cache_key != replace(bc, kind="isochrones").cache_key
    for options in ({"version": "9.9"}, {"rotation": .2}, {"afe": .2},
                    {"feh": [.1]}, {"feh": []}, {"feh": [0., 0.]}):
        with pytest.raises(ValueError):
            track_archives(**options)


@pytest.mark.parametrize("change", [{"version": "../1"}, {"kind": "/tmp"},
                                   {"filename": "a/b.txz"}, {"url": "file:///tmp/local"}])
def test_source_identity_rejects_invalid_paths(source, change):
    with pytest.raises(ValueError):
        replace(source, **change)


def test_raw_path_is_independent_of_npz_and_package_versions(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "user_cache_path", lambda *a, **k: tmp_path)
    monkeypatch.delenv(paths.MIST_RAW_ENV_VAR, raising=False)
    monkeypatch.setenv(paths.MISTGRID_ENV_VAR, str(tmp_path / "other/grid.npz"))
    assert paths.resolve_mist_raw_dir() == tmp_path / "mist/raw"
    monkeypatch.setenv(paths.MIST_RAW_ENV_VAR, str(tmp_path / "shared"))
    assert paths.resolve_mist_raw_dir() == tmp_path / "shared"
    assert paths.resolve_mist_raw_dir(tmp_path / "explicit") == tmp_path / "explicit"


def test_download_receipt_offline_reuse_and_extraction(source, tmp_path, monkeypatch):
    payload = tar_bytes()
    calls = []

    def download(request, **kwargs):
        calls.append(request.full_url)
        return Response(payload)

    monkeypatch.setattr(cache, "urlopen", download)
    first = cache.fetch_archive(source, cache_dir=tmp_path)
    record = json.loads((tmp_path / source.cache_key / "current.json").read_text())
    assert calls == [source.url]
    assert record["size_bytes"] == len(payload)
    assert record["download_url"] == "https://example.org/final.txz"
    assert record["sha256"] == first.parent.name
    assert first.read_bytes() == payload
    monkeypatch.setattr(cache, "urlopen", lambda *a, **k: pytest.fail("cache reuse must be offline"))
    assert cache.fetch_archive(source, cache_dir=tmp_path, offline=True) == first
    extracted = cache.extract_archive(source, cache_dir=tmp_path, offline=True)
    assert (extracted / "data/track.eep").read_bytes() == b"native track\n"
    monkeypatch.setattr(cache, "_tar_contents", lambda *a, **k: pytest.fail("unnecessary extraction"))
    assert cache.extract_archive(source, cache_dir=tmp_path, offline=True) == extracted
    assert not list(tmp_path.rglob("*.part"))


def test_local_import_and_version_isolation(source, tmp_path):
    local = tmp_path / "existing.txz"
    payload = tar_bytes()
    local.write_bytes(payload)
    root = tmp_path / "cache"
    old = cache.fetch_archive(source, cache_dir=root, local_path=local, offline=True)
    new = cache.fetch_archive(replace(source, version="2.5"), cache_dir=root,
                              local_path=local, offline=True)
    assert old != new
    assert old.read_bytes() == new.read_bytes() == local.read_bytes() == payload
    local.unlink()
    assert cache.fetch_archive(source, cache_dir=root, offline=True) == old


def test_concurrent_downloads_publish_consistent_receipts(source, tmp_path, monkeypatch):
    rendezvous = Barrier(2)

    def download(*args, **kwargs):
        rendezvous.wait(timeout=5)
        return Response(tar_bytes())

    monkeypatch.setattr(cache, "urlopen", download)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: cache.extract_archive(source, cache_dir=tmp_path), range(2)))
    assert results[0] == results[1]
    assert (results[0] / "data/track.eep").read_bytes() == b"native track\n"
    cache.fetch_archive(source, cache_dir=tmp_path, offline=True)
    assert not list(tmp_path.rglob("*.part"))


def test_offline_miss_and_partial_files_are_not_reused(source, tmp_path, monkeypatch):
    entry = tmp_path / source.cache_key
    entry.mkdir(parents=True)
    partial = entry / "interrupted.part"
    partial.write_bytes(b"unfinished")
    with pytest.raises(FileNotFoundError, match="offline"):
        cache.fetch_archive(source, cache_dir=tmp_path, offline=True)
    monkeypatch.setattr(cache, "urlopen", lambda *a, **k: Response(tar_bytes()))
    result = cache.fetch_archive(source, cache_dir=tmp_path)
    assert result.read_bytes() == tar_bytes()
    assert partial.read_bytes() == b"unfinished"  # Never interpreted as completed data.


@pytest.mark.parametrize("failure", ["short", "html", "xz", "disconnect"])
def test_failed_download_never_publishes_an_archive(source, tmp_path, monkeypatch, failure):
    payload = tar_bytes()

    class Interrupted(Response):
        def read(self, size=-1):
            if self.tell():
                raise ConnectionError("connection lost")
            return super().read(20)

    responses = {"short": lambda: Response(payload, len(payload)+10),
                 "html": lambda: Response(b"<html>not an archive</html>"),
                 "xz": lambda: Response(payload[:-16]),
                 "disconnect": lambda: Interrupted(payload)}
    monkeypatch.setattr(cache, "urlopen", lambda *a, **k: responses[failure]())
    with pytest.raises((cache.MistCacheError, lzma.LZMAError, EOFError, ConnectionError)):
        cache.fetch_archive(source, cache_dir=tmp_path)
    assert not (tmp_path / source.cache_key / "current.json").exists()
    assert not list(tmp_path.rglob("*.part"))
    with pytest.raises(FileNotFoundError):
        cache.fetch_archive(source, cache_dir=tmp_path, offline=True)


def test_failed_refresh_keeps_previous_generation(source, tmp_path, monkeypatch):
    local = tmp_path / "input.txz"
    local.write_bytes(tar_bytes())
    first = cache.fetch_archive(source, cache_dir=tmp_path, local_path=local)
    receipt = tmp_path / source.cache_key / "current.json"
    before = receipt.read_bytes()
    monkeypatch.setattr(cache, "urlopen", lambda *a, **k: Response(b"error response"))
    with pytest.raises(lzma.LZMAError):
        cache.fetch_archive(source, cache_dir=tmp_path, force=True)
    assert receipt.read_bytes() == before
    assert cache.fetch_archive(source, cache_dir=tmp_path, offline=True) == first
    local.write_bytes(tar_bytes({"updated.eep": b"updated native data"}))
    original_write = cache._write_json
    monkeypatch.setattr(cache, "_write_json", lambda *a, **k: (_ for _ in ()).throw(OSError("disk failure")))
    with pytest.raises(OSError, match="disk failure"):
        cache.fetch_archive(source, cache_dir=tmp_path, local_path=local, force=True)
    assert receipt.read_bytes() == before
    assert cache.fetch_archive(source, cache_dir=tmp_path, offline=True) == first
    monkeypatch.setattr(cache, "_write_json", original_write)
    second = cache.fetch_archive(source, cache_dir=tmp_path, local_path=local, force=True)
    assert second != first and first.is_file()
    assert cache.fetch_archive(source, cache_dir=tmp_path, offline=True) == second


def test_corruption_detected_without_silent_download_and_explicit_repair(source, tmp_path):
    local = tmp_path / "input.txz"
    payload = tar_bytes()
    local.write_bytes(payload)
    path = cache.fetch_archive(source, cache_dir=tmp_path, local_path=local)
    damaged = bytearray(path.read_bytes())
    damaged[30] ^= 1
    path.write_bytes(damaged)
    with pytest.raises(cache.MistCacheError, match="corrupt"):
        cache.fetch_archive(source, cache_dir=tmp_path)
    repaired = cache.fetch_archive(source, cache_dir=tmp_path, local_path=local, force=True)
    assert repaired.read_bytes() == payload
    with pytest.raises(cache.MistCacheError, match="source mismatch"):
        cache.fetch_archive(replace(source, url="https://example.org/changed"), cache_dir=tmp_path)


@pytest.mark.parametrize("name,kind", [("../escape", tarfile.REGTYPE),
                                      ("/absolute", tarfile.REGTYPE),
                                      (".complete.json", tarfile.REGTYPE),
                                      ("data/link", tarfile.SYMTYPE),
                                      ("data/hard", tarfile.LNKTYPE)])
def test_archive_cannot_escape_or_create_links(source, tmp_path, name, kind):
    member = tarfile.TarInfo(name)
    member.type, member.linkname = kind, "../../escape"
    local = tmp_path / "bad.txz"
    local.write_bytes(tar_bytes(member=member))
    with pytest.raises(cache.MistCacheError, match="Unsupported archive member"):
        cache.fetch_archive(source, cache_dir=tmp_path, local_path=local)
    assert not (tmp_path / source.cache_key / "current.json").exists()


def test_incomplete_extraction_is_not_reused(source, tmp_path, monkeypatch):
    local = tmp_path / "input.txz"
    local.write_bytes(tar_bytes())
    archive = cache.fetch_archive(source, cache_dir=tmp_path, local_path=local)
    original = cache._tar_contents

    def interrupted(path, destination=None):
        if destination is not None:
            (destination / "partial").write_text("incomplete")
            raise OSError("extract failure")
        return original(path)

    monkeypatch.setattr(cache, "_tar_contents", interrupted)
    with pytest.raises(OSError, match="extract failure"):
        cache.extract_archive(source, cache_dir=tmp_path, offline=True)
    assert not (archive.parent / "extracted").exists()
    assert not list(tmp_path.rglob("*.part"))
    monkeypatch.setattr(cache, "_tar_contents", original)
    extracted = cache.extract_archive(source, cache_dir=tmp_path, offline=True)
    (extracted / "data/track.eep").unlink()
    with pytest.raises(cache.MistCacheError, match="Missing or modified"):
        cache.extract_archive(source, cache_dir=tmp_path, offline=True)


def cmd_fixture():
    """Eight metallicities exercise the retained six-metallicity cut."""
    files = {}
    for feh in (-4., -3.5, -3., -2.5, -2., -1.75, -1.5, -1.25):
        rows = []
        for age in (8., 9.):
            for eep in (0, 1, 2):
                mass = 1. + .1*eep + .01*feh
                row = [eep, age, mass, mass-.01, 3.75, 4., .5, feh, feh+.03]
                row += list(np.arange(24, dtype=float)) + [0.]
                rows.append(" ".join(str(v) for v in row))
        name = f"MIST_v1.2_vvcrit0.4_UBVRIplus/MIST_v1.2_feh_m{abs(feh):.2f}_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd"
        files[name] = ("\n".join(rows)+"\n").encode()
    return files


@pytest.fixture
def creator(tmp_path, monkeypatch):
    module = importlib.import_module("jaxstar.mistfit.mistgrid.create_grid")
    monkeypatch.setattr(module, "default_mistgrid_path", lambda: tmp_path / "legacy/grid.npz")
    return module


def test_build_two_npz_outputs_reuses_one_download(creator, tmp_path, monkeypatch):
    calls = []

    def download(*a, **k):
        calls.append(1)
        return Response(tar_bytes(cmd_fixture()))

    monkeypatch.setattr(cache, "urlopen", download)
    first = creator.create_mistgrid(tmp_path / "one/grid.npz", raw_cache_dir=tmp_path / "raw")
    monkeypatch.setattr(cache, "urlopen", lambda *a, **k: pytest.fail("rebuild downloaded again"))
    second = creator.create_mistgrid(tmp_path / "two/grid.npz", raw_cache_dir=tmp_path / "raw", offline=True)
    assert len(calls) == 1
    with np.load(first) as a, np.load(second) as b:
        assert set(a.files) == set(b.files)
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_array_equal(a["fgrid"], [-1.5, -1.25])
        np.testing.assert_array_equal(a["eepgrid"], [0., 1., 2.])
        np.testing.assert_allclose(a["dmdeep"], .1)
        np.testing.assert_allclose(a["teff"], 10**3.75)
        assert a["mass"].shape == (2, 2, 3)


@pytest.mark.parametrize("expanded", [False, True])
def test_legacy_source_files_are_reused_offline(creator, tmp_path, expanded):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    if expanded:
        for name, data in cmd_fixture().items():
            path = legacy / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    else:
        (legacy / isochrone_archive().filename).write_bytes(tar_bytes(cmd_fixture()))
    output = creator.create_mistgrid(tmp_path / "other/grid.npz",
                                    raw_cache_dir=tmp_path / "raw", offline=True)
    with np.load(output) as data:
        np.testing.assert_allclose(data["dmdeep"], .1)
    assert (legacy / ("MIST_v1.2_vvcrit0.4_UBVRIplus" if expanded else isochrone_archive().filename)).exists()
