"""Support checks for partially populated track grids."""

import jax.numpy as jnp

from jaxstar.grid import GridResult
from jaxstar.grid.core import _fractional_coordinate, _payload_stencil


def track_support(data, coordinates, keys):
    """Require every nonzero-weight corner, separately for each output field.

    Use the same coordinate mapping as RectilinearGrid. Checking Boolean
    corners avoids rounding an interpolated mask to one near a missing node.
    Exact nodes/faces remain usable when adjacent zero-weight corners are absent.
    """
    grid = data.grid
    queries = jnp.broadcast_arrays(*(coordinates[k] for k in grid.axis_names))
    fractions, in_bounds = [], jnp.ones_like(queries[0], dtype=bool)
    for name, kind, query in zip(grid.axis_names, grid.axis_kinds, queries):
        fractional, valid = _fractional_coordinate(grid.axis(name), kind, query)
        fractions.append(fractional)
        in_bounds &= valid
    stencil = _payload_stencil(fractions, [grid.axis(k).size for k in grid.axis_names])
    outputs = []
    for key in keys:
        mask = data.validity[grid.field_names.index(key)]
        valid = in_bounds
        for indices, weight in stencil:
            valid = valid & ((weight == 0) | mask[indices])
        outputs.append(valid)
    return GridResult(keys, tuple(outputs))
