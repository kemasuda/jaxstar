"""Reusable raw MIST archives, independent of generated NPZ grid schemas.

Downloads/imports are streamed into temporary files, validated and hashed.
An atomic receipt selects a complete content-addressed archive, so interrupted
refreshes cannot invalidate the previous copy. Readers never use .part files.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import lzma
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
from urllib.request import Request, urlopen

from .paths import resolve_mist_raw_dir
from .sources import MistArchive


class MistCacheError(RuntimeError):
    """A cached receipt/archive is invalid; never silently replace local data."""


_CHUNK = 1024 * 1024


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".part")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path):
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise ValueError("expected an object")
        return value
    except (OSError, ValueError) as exc:
        raise MistCacheError(f"Invalid cache record: {path}") from exc


def _tar_contents(path, destination=None):
    """Validate the whole XZ stream and optionally extract ordinary files.

    Avoid extractall, links and special files, including on Python 3.10.
    Draining the decompressor checks the XZ footer after tar's end marker.
    """
    files = {}
    with lzma.open(path, "rb") as compressed:
        with tarfile.open(fileobj=compressed, mode="r|") as archive:
            for member in archive:
                relative = PurePosixPath(member.name)
                if (relative.is_absolute() or ".." in relative.parts or "\\" in member.name
                        or ":" in member.name or str(relative) == ".complete.json"
                        or not (member.isfile() or member.isdir())):
                    raise MistCacheError(f"Unsupported archive member: {member.name}")
                if not relative.parts:
                    if member.isdir():
                        continue
                    raise MistCacheError("Empty archive member name")
                if destination is not None:
                    target = destination.joinpath(*relative.parts)
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst, _CHUNK)
                if member.isfile():
                    if str(relative) in files:
                        raise MistCacheError(f"Duplicate archive file: {member.name}")
                    files[str(relative)] = member.size
        while compressed.read(_CHUNK):
            pass
    if not files:
        raise MistCacheError("Archive contains no regular files")
    return files


def _cached(source, entry):
    receipt = entry / "current.json"
    if not receipt.exists():
        return None
    record = _read_json(receipt)
    if record.get("format_version") != 1 or record.get("source") != asdict(source):
        raise MistCacheError(f"Cache source mismatch: {receipt}; use force=True to refresh")
    digest = record.get("sha256", "")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise MistCacheError(f"Invalid SHA-256 in {receipt}")
    path = entry / digest / source.filename
    if (not path.is_file() or path.stat().st_size != record.get("size_bytes")
            or _digest(path) != digest):
        raise MistCacheError(f"Cached archive is missing or corrupt: {path}; use force=True to refresh")
    return path


def fetch_archive(source, *, cache_dir=None, local_path=None, offline=False,
                  force=False, timeout=60):
    """Return a verified cached archive, importing a local file or downloading.

    A valid cache always wins unless force=True. local_path imports an existing
    archive without changing it or using the network. offline=True forbids
    network access, including on a miss; local imports still work. Reuse checks
    the stored byte size and SHA-256 without contacting the server. SHA-256 is
    a local integrity record, not an upstream authenticity guarantee.

    force=True refreshes explicitly; failed refreshes retain the previous copy.
    Partial transfers restart on retry (no HTTP Range resume). Successful old
    content remains cached when a refresh returns different bytes.
    """
    if not isinstance(source, MistArchive):
        raise TypeError("source must be a MistArchive")
    entry = resolve_mist_raw_dir(cache_dir) / source.cache_key
    if not force:
        cached = _cached(source, entry)
        if cached is not None:
            return cached
    if local_path is None and offline:
        raise FileNotFoundError(f"No reusable archive for {source.filename} (offline=True)")
    entry.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=entry, suffix=".part")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as output:
            if local_path is not None:
                local = Path(local_path).expanduser().resolve()
                with local.open("rb") as stream:
                    shutil.copyfileobj(stream, output, _CHUNK)
                origin = {"imported_from": str(local)}
            else:
                request = Request(source.url, headers={"User-Agent": "jaxstar-mistgrid",
                                                       "Accept-Encoding": "identity"})
                with urlopen(request, timeout=timeout) as stream:
                    shutil.copyfileobj(stream, output, _CHUNK)
                    length = stream.headers.get("Content-Length")
                    if length is not None and output.tell() != int(length):
                        raise MistCacheError("Incomplete archive download (Content-Length mismatch)")
                    origin = {"download_url": stream.geturl()}
            output.flush()
            os.fsync(output.fileno())
        _tar_contents(temporary)
        digest = _digest(temporary)
        target = entry / digest / source.filename
        target.parent.mkdir(parents=True, exist_ok=True)
        size = temporary.stat().st_size
        os.replace(temporary, target)
        _write_json(entry / "current.json", {
            "format_version": 1, "source": asdict(source), "sha256": digest,
            "size_bytes": size, "stored_at": datetime.now(timezone.utc).isoformat(),
            **origin,
        })
        return target
    finally:
        temporary.unlink(missing_ok=True)


def extract_archive(source, *, cache_dir=None, local_path=None, offline=False):
    """Return a completed extraction, reusing it or the cached archive offline.

    Extraction is keyed by archive content and published by directory rename.
    A completion record checks member existence/size on reuse; edited extracted
    data should be kept outside the cache. No archive paths can escape the
    temporary extraction directory.
    """
    archive = fetch_archive(source, cache_dir=cache_dir, local_path=local_path, offline=offline)
    target = archive.parent / "extracted"

    def check():
        record = _read_json(target / ".complete.json")
        files = record.get("files")
        if record.get("sha256") != archive.parent.name or not isinstance(files, dict) or not files:
            raise MistCacheError(f"Incomplete extraction: {target}")
        for name, size in files.items():
            member = target / name
            if not member.is_file() or member.stat().st_size != size:
                raise MistCacheError(f"Missing or modified extracted file: {member}")

    if target.exists():
        check()
        return target
    temporary = Path(tempfile.mkdtemp(dir=archive.parent, suffix=".part"))
    try:
        files = _tar_contents(archive, temporary)
        _write_json(temporary / ".complete.json", {"sha256": archive.parent.name, "files": files})
        try:
            temporary.rename(target)
        except OSError:
            if not target.is_dir():
                raise
            check()  # A concurrent extraction may have completed first.
        return target
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
