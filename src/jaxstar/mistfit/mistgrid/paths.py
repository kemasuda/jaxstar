"""Path resolution for the generated MIST grid."""

import os
from pathlib import Path

from platformdirs import user_cache_path


MISTGRID_ENV_VAR = "JAXSTAR_MISTGRID_PATH"
MISTGRID_FILENAME = "mistgrid_iso.npz"
MISTGRID_CACHE_VERSION = "MIST_v1.2_vvcrit0.4_UBVRIplus"


def default_mistgrid_path() -> Path:
    """Return the platform-specific cache path used by default."""
    return (
        user_cache_path("jaxstar", appauthor=False)
        / MISTGRID_CACHE_VERSION
        / MISTGRID_FILENAME
    )


def resolve_mistgrid_path(path=None) -> Path:
    """Resolve an explicit path, environment override, or default cache path."""
    if path is not None:
        return Path(path).expanduser()

    environment_path = os.environ.get(MISTGRID_ENV_VAR)
    if environment_path:
        return Path(environment_path).expanduser()

    return default_mistgrid_path()
