"""Load the existing isochrone NPZ schema into the numerical MIST data carrier."""

import numpy as np

from jaxstar.grid import Field, RectilinearGrid
from .._data import _MistGridData
from .create_grid import create_mistgrid
from .paths import resolve_mistgrid_path


def load_isochrones(path=None):
    """Read every grid field once and close the NPZ before returning.

    An explicit path must exist. With no path, honor the environment override
    or user cache and generate a missing grid there, as in historical MistFit.
    No I/O is performed by subsequent model evaluation or PyTree restoration.
    """
    grid_path = resolve_mistgrid_path(path)
    if path is None and not grid_path.is_file():
        print(f"{grid_path} not found.")
        create_mistgrid(path=grid_path)
    axis_keys = {"age": "logagrid", "feh": "fgrid", "eep": "eepgrid"}
    with np.load(grid_path, allow_pickle=False) as data:
        missing = set(axis_keys.values()) - set(data.files)
        if missing:
            raise ValueError(f"isochrone NPZ is missing axes: {sorted(missing)}")
        grid = RectilinearGrid(
            axes={name: data[key] for name, key in axis_keys.items()},
            fields={
                name: Field(data[name], dims=("age", "feh", "eep"))
                for name in data.files if name not in axis_keys.values()
            },
        )
    return _MistGridData(grid, kind="isochrone", sources=(str(grid_path.resolve()),))
