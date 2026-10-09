"""Version-specific MIST archive catalogue, independent of transport/parsers.

Catalogue: https://mist.science/model_grids.html (checked 2026-10-07).
Only v1.2 UBVRIplus CMDs/BCs and native EEP archives are registered here.
Adding a release requires an explicit catalogue and separate parser/physics
validation; substituting a version in a URL is not sufficient.
"""

from dataclasses import dataclass
from pathlib import Path
import re
from urllib.parse import urlsplit


@dataclass(frozen=True)
class MistArchive:
    """One physical archive; filename must encode its composition/rotation.

    version names the MIST release, kind separates tracks/isochrones/BC tables.
    BC archives covering all FeH or shared by rotations have one identity.
    The cache stores this descriptor verbatim, including the source URL.
    """

    version: str
    kind: str
    filename: str
    url: str

    def __post_init__(self):
        for name in ("version", "kind", "filename"):
            value = getattr(self, name)
            if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]*", value):
                raise ValueError(f"{name} must be a single safe path component")
        parts = urlsplit(self.url)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ValueError("archive URL must be HTTP(S)")

    @property
    def cache_key(self):
        return Path(f"v{self.version}") / self.kind / self.filename


_BASE = "https://mist.science"
_V12_FEH = (-4., -3.5, -3., -2.5, -2., -1.75, -1.5, -1.25,
            -1., -.75, -.5, -.25, 0., .25, .5)


def _v12(version, afe, rotation=None, photometry=None):
    if version != "1.2":
        raise ValueError("Only MIST version='1.2' is currently registered")
    if afe != 0.:
        raise ValueError("MIST v1.2 packaged archives require afe=0")
    if rotation is not None and rotation not in (0., .4):
        raise ValueError("MIST v1.2 rotation must be 0.0 or 0.4")
    if photometry is not None and photometry != "UBVRIplus":
        raise ValueError("Only UBVRIplus photometry is currently registered")


def isochrone_archive(*, version="1.2", rotation=.4, afe=0., photometry="UBVRIplus"):
    """Synthetic CMD archive containing all available FeH values."""
    _v12(version, afe, rotation, photometry)
    filename = f"MIST_v{version}_vvcrit{rotation:.1f}_{photometry}.txz"
    return MistArchive(version, "isochrones", filename,
                       f"{_BASE}/data/tarballs_v{version}/{filename}")


def track_metallicities(*, version="1.2", rotation=.4, afe=0., feh=None):
    """Validate a native EEP archive selection; None selects the full catalogue."""
    _v12(version, afe, rotation)
    metallicities = _V12_FEH if feh is None else tuple(feh)
    if not metallicities or len(set(metallicities)) != len(metallicities):
        raise ValueError("feh must be a nonempty sequence of distinct metallicities")
    for value in metallicities:
        if value not in _V12_FEH:
            raise ValueError(f"Unavailable MIST v1.2 FeH: {value}; choose from {_V12_FEH}")
    return metallicities


def track_archives(*, version="1.2", rotation=.4, afe=0., feh=None):
    """Describe all native EEP archives, or an explicit iterable of FeH values.

    Each archive contains all published masses. This only returns descriptors;
    no network or disk access occurs until fetch_archive is called.
    """
    metallicities = track_metallicities(version=version, rotation=rotation, afe=afe, feh=feh)
    result = []
    for value in metallicities:
        token = ("m" if value < 0 else "p") + f"{abs(value):.2f}"
        filename = f"MIST_v{version}_feh_{token}_afe_p0.0_vvcrit{rotation:.1f}_EEPS.txz"
        result.append(MistArchive(version, "tracks", filename,
                                  f"{_BASE}/data/tarballs_v{version}/{filename}"))
    return tuple(result)


def bolometric_correction_archive(*, version="1.2", afe=0., photometry="UBVRIplus"):
    """BC table shared by all track masses, FeH values and rotation rates."""
    _v12(version, afe, photometry=photometry)
    return MistArchive(version, "bolometric-corrections", f"{photometry}.txz",
                       f"{_BASE}/BC_tables/v1/{photometry}.txz")
