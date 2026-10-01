"""Pure numerical building blocks for named rectilinear grids.

This module deliberately has no MIST, spectrum, file-I/O, or NumPyro
dependencies.  The first implementation supports scalar fields defined over
all grid axes.  Additional field layouts can be added without changing the
coordinate or selection API.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.ndimage import map_coordinates


ArrayLike = Any


@dataclass(frozen=True, eq=False)
class Axis:
    """One named grid axis supplied to :class:`RectilinearGrid`.

    Args:
        values: Finite, strictly increasing one-dimensional coordinates.
        kind: ``"auto"`` (default), ``"regular"``, ``"nonuniform"``, or
            ``"singleton"``.  ``"auto"`` detects the appropriate kernel once
            when the grid is constructed.
    """

    values: ArrayLike
    kind: str = "auto"


@dataclass(frozen=True, eq=False)
class Field:
    """A scalar field and the named grid dimensions of its array."""

    values: ArrayLike
    dims: tuple[str, ...]

    def __init__(self, values: ArrayLike, dims: Sequence[str]):
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "dims", tuple(dims))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class GridResult(Mapping[str, Any]):
    """Immutable, ordered mapping returned by grid interpolation."""

    names: tuple[str, ...]
    arrays: tuple[Any, ...]

    def __post_init__(self):
        names = tuple(self.names)
        arrays = tuple(self.arrays)
        if len(names) != len(arrays):
            raise ValueError("result names and arrays must have the same length")
        _validate_names(names, "result")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "arrays", arrays)

    def __getitem__(self, name: str) -> Any:
        try:
            index = self.names.index(name)
        except ValueError as error:
            raise KeyError(name) from error
        return self.arrays[index]

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.names)

    def tree_flatten(self):
        return self.arrays, self.names

    @classmethod
    def tree_unflatten(cls, names, arrays):
        return cls(names=names, arrays=tuple(arrays))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False, init=False)
class RectilinearGrid:
    """Immutable collection of named axes and scalar fields.

    Construct a grid from mappings so their insertion order defines the schema
    order.  Field arrays may use a different dimension order as long as
    ``Field.dims`` declares it explicitly.

    The grid is a JAX PyTree: numerical arrays are dynamic leaves, while names,
    dimension metadata, axis kinds, and the boundary policy are static.  A grid
    can therefore be passed as an argument to a jitted function without making
    the mutable-object/static-argument mistake present in the legacy grid.
    """

    axis_names: tuple[str, ...]
    axis_kinds: tuple[str, ...]
    field_names: tuple[str, ...]
    field_dims: tuple[tuple[str, ...], ...]
    boundary: str
    _axis_values: tuple[Any, ...]
    _field_values: tuple[Any, ...]
    fill_value: Any

    def __init__(
        self,
        *,
        axes: Mapping[str, Axis | ArrayLike],
        fields: Mapping[str, Field],
        fill_value: ArrayLike = -jnp.inf,
        boundary: str = "constant",
    ):
        normalized = _normalize_grid(
            axes=axes,
            fields=fields,
            fill_value=fill_value,
            boundary=boundary,
        )
        self._set_components(**normalized)

    def _set_components(
        self,
        *,
        axis_names,
        axis_kinds,
        field_names,
        field_dims,
        boundary,
        axis_values,
        field_values,
        fill_value,
    ):
        object.__setattr__(self, "axis_names", tuple(axis_names))
        object.__setattr__(self, "axis_kinds", tuple(axis_kinds))
        object.__setattr__(self, "field_names", tuple(field_names))
        object.__setattr__(self, "field_dims", tuple(field_dims))
        object.__setattr__(self, "boundary", boundary)
        object.__setattr__(self, "_axis_values", tuple(axis_values))
        object.__setattr__(self, "_field_values", tuple(field_values))
        object.__setattr__(self, "fill_value", fill_value)

    def axis(self, name: str):
        """Return the coordinate array for one named axis."""
        try:
            index = self.axis_names.index(name)
        except ValueError as error:
            raise KeyError(name) from error
        return self._axis_values[index]

    def field(self, name: str) -> Field:
        """Return one field together with its dimension metadata."""
        try:
            index = self.field_names.index(name)
        except ValueError as error:
            raise KeyError(name) from error
        return Field(self._field_values[index], self.field_dims[index])

    def interpolate(
        self,
        coordinates: Mapping[str, ArrayLike],
        *,
        keys: str | Sequence[str] | None = None,
    ) -> GridResult:
        """Linearly interpolate selected fields at physical coordinates.

        Args:
            coordinates: One scalar or array per named axis.  Coordinate arrays
                are broadcast together; mapping insertion order is irrelevant.
            keys: Field name, ordered sequence of field names, or ``None`` for
                all fields in schema order.  This selection is immutable and is
                normally static for a compiled model configuration.

        Returns:
            An immutable ordered mapping from selected names to JAX arrays.
        """
        selected_names = _normalize_selected_names(keys, self.field_names)
        query_values = _coordinates_in_schema_order(coordinates, self.axis_names)

        try:
            broadcast_queries = jnp.broadcast_arrays(
                *(jnp.asarray(value) for value in query_values)
            )
        except ValueError as error:
            raise ValueError(
                "coordinate arrays must be broadcast-compatible"
            ) from error

        fractional_by_name = {}
        valid_by_name = {}
        for name, kind, axis_values, query in zip(
            self.axis_names,
            self.axis_kinds,
            self._axis_values,
            broadcast_queries,
        ):
            fractional, valid = _fractional_coordinate(axis_values, kind, query)
            fractional_by_name[name] = fractional
            valid_by_name[name] = valid

        all_valid = valid_by_name[self.axis_names[0]]
        for name in self.axis_names[1:]:
            all_valid = jnp.logical_and(all_valid, valid_by_name[name])

        outputs = []
        for name in selected_names:
            index = self.field_names.index(name)
            dims = self.field_dims[index]
            field_coordinates = [fractional_by_name[dim] for dim in dims]
            interpolated = map_coordinates(
                self._field_values[index],
                field_coordinates,
                order=1,
                mode="nearest",
            )
            fill = jnp.asarray(self.fill_value, dtype=interpolated.dtype)
            outputs.append(jnp.where(all_valid, interpolated, fill))

        return GridResult(names=selected_names, arrays=tuple(outputs))

    def tree_flatten(self):
        children = (self._axis_values, self._field_values, self.fill_value)
        metadata = (
            self.axis_names,
            self.axis_kinds,
            self.field_names,
            self.field_dims,
            self.boundary,
        )
        return children, metadata

    @classmethod
    def tree_unflatten(cls, metadata, children):
        axis_names, axis_kinds, field_names, field_dims, boundary = metadata
        axis_values, field_values, fill_value = children
        grid = object.__new__(cls)
        grid._set_components(
            axis_names=axis_names,
            axis_kinds=axis_kinds,
            field_names=field_names,
            field_dims=field_dims,
            boundary=boundary,
            axis_values=axis_values,
            field_values=field_values,
            fill_value=fill_value,
        )
        return grid


def _normalize_grid(*, axes, fields, fill_value, boundary):
    if not isinstance(axes, Mapping) or not axes:
        raise ValueError("axes must be a non-empty mapping")
    if not isinstance(fields, Mapping) or not fields:
        raise ValueError("fields must be a non-empty mapping")
    if boundary != "constant":
        raise ValueError("only boundary='constant' is currently supported")

    axis_names = tuple(axes)
    _validate_names(axis_names, "axis")

    axis_values = []
    axis_kinds = []
    axis_sizes = {}
    for name, specification in axes.items():
        axis = specification if isinstance(specification, Axis) else Axis(specification)
        values, kind = _normalize_axis(name, axis)
        axis_values.append(values)
        axis_kinds.append(kind)
        axis_sizes[name] = values.shape[0]

    field_names = tuple(fields)
    _validate_names(field_names, "field")

    field_values = []
    field_dims = []
    expected_axis_set = set(axis_names)
    for name, field in fields.items():
        if not isinstance(field, Field):
            raise TypeError(f"field {name!r} must be a Field instance")
        dims = tuple(field.dims)
        _validate_names(dims, f"dimensions for field {name!r}")
        unknown_dims = set(dims) - expected_axis_set
        if unknown_dims:
            raise ValueError(
                f"field {name!r} has unknown dimensions: {sorted(unknown_dims)!r}"
            )
        if set(dims) != expected_axis_set or len(dims) != len(axis_names):
            raise ValueError(
                f"field {name!r} must contain every grid axis exactly once"
            )

        input_values = np.array(field.values, copy=True)
        jax_values = jnp.asarray(input_values)
        host_values = np.asarray(jax_values)
        expected_shape = tuple(axis_sizes[dim] for dim in dims)
        if host_values.shape != expected_shape:
            raise ValueError(
                f"field {name!r} has shape {host_values.shape}, "
                f"expected {expected_shape} for dimensions {dims}"
            )
        if not np.issubdtype(host_values.dtype, np.floating):
            raise TypeError(f"field {name!r} must have a floating-point dtype")

        field_values.append(jax_values)
        field_dims.append(dims)

    input_fill = np.array(fill_value, copy=True)
    is_real_fill = np.issubdtype(
        input_fill.dtype, np.integer
    ) or np.issubdtype(input_fill.dtype, np.floating)
    if not is_real_fill:
        raise TypeError("fill_value must be a real number")
    jax_fill = jnp.asarray(input_fill)
    host_fill = np.asarray(jax_fill)
    if host_fill.shape != ():
        raise ValueError("fill_value must be a scalar")

    return {
        "axis_names": axis_names,
        "axis_kinds": tuple(axis_kinds),
        "field_names": field_names,
        "field_dims": tuple(field_dims),
        "boundary": boundary,
        "axis_values": tuple(axis_values),
        "field_values": tuple(field_values),
        "fill_value": jax_fill,
    }


def _normalize_axis(name: str, axis: Axis):
    input_values = np.array(axis.values, copy=True)
    if not (
        np.issubdtype(input_values.dtype, np.integer)
        or np.issubdtype(input_values.dtype, np.floating)
    ):
        raise TypeError(f"axis {name!r} must have a real numeric dtype")
    if np.issubdtype(input_values.dtype, np.integer):
        input_values = input_values.astype(np.float64)

    jax_values = jnp.asarray(input_values)
    host_values = np.asarray(jax_values)
    if host_values.ndim != 1:
        raise ValueError(f"axis {name!r} must be one-dimensional")
    if host_values.size == 0:
        raise ValueError(f"axis {name!r} must not be empty")
    if not np.issubdtype(host_values.dtype, np.floating):
        raise TypeError(f"axis {name!r} must have a floating-point JAX dtype")
    if not np.all(np.isfinite(host_values)):
        raise ValueError(f"axis {name!r} must contain only finite values")
    if host_values.size > 1 and not np.all(np.diff(host_values) > 0):
        raise ValueError(f"axis {name!r} must be strictly increasing")

    allowed_kinds = {"auto", "regular", "nonuniform", "singleton"}
    if axis.kind not in allowed_kinds:
        raise ValueError(
            f"axis {name!r} kind must be one of {sorted(allowed_kinds)!r}"
        )

    if host_values.size == 1:
        if axis.kind not in {"auto", "singleton"}:
            raise ValueError(f"axis {name!r} with one value must be singleton")
        kind = "singleton"
    else:
        if axis.kind == "singleton":
            raise ValueError(f"axis {name!r} declared singleton has multiple values")
        is_regular = _is_regular_axis(host_values)
        if axis.kind == "regular" and not is_regular:
            raise ValueError(f"axis {name!r} declared regular is not equally spaced")
        if axis.kind == "auto":
            kind = "regular" if is_regular else "nonuniform"
        else:
            kind = axis.kind

    return jax_values, kind


def _is_regular_axis(values):
    """Conservatively recognize axes represented accurately by an affine map."""
    if np.issubdtype(values.dtype, np.integer):
        differences = np.diff(values)
        return bool(np.all(differences == differences[0]))

    # Test the same affine operation used by the regular-axis kernel.  Measuring
    # error in index units avoids a tolerance that becomes too permissive for
    # axes with a large physical offset and fine spacing.  Ambiguous axes fall
    # back to the nonuniform kernel, which is slower but exact at stored nodes.
    dtype = values.dtype
    count = np.asarray(values.size - 1, dtype=dtype)
    fractional = (values - values[0]) * count / (values[-1] - values[0])
    expected = np.arange(values.size, dtype=dtype)
    roundoff_tolerance = (
        64.0
        * np.finfo(dtype).eps
        * np.maximum(np.abs(expected), np.asarray(1.0, dtype=dtype))
    )
    # Never let an axis become "more regular" merely because it has many
    # nodes.  The roundoff estimate above grows with the index, so without a
    # cap a visibly displaced node in a long wavelength grid can be accepted
    # as regular.  Falling back to searchsorted is preferable whenever the
    # affine map misses a stored node by a meaningful fraction of a cell.
    tolerance = np.minimum(
        roundoff_tolerance,
        np.asarray(1.0e-3, dtype=dtype),
    )
    return bool(np.all(np.abs(fractional - expected) <= tolerance))


def _validate_names(names: tuple[str, ...], description: str):
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"{description} names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError(f"{description} names must be unique")


def _normalize_selected_names(keys, field_names):
    if keys is None:
        selected = field_names
    elif isinstance(keys, str):
        selected = (keys,)
    else:
        selected = tuple(keys)

    if not selected:
        raise ValueError("at least one field key must be selected")
    if len(set(selected)) != len(selected):
        raise ValueError("field keys must not contain duplicates")

    unknown = tuple(name for name in selected if name not in field_names)
    if unknown:
        raise KeyError(f"unknown field keys: {unknown!r}")
    return selected


def _coordinates_in_schema_order(coordinates, axis_names):
    if not isinstance(coordinates, Mapping):
        raise TypeError("coordinates must be a mapping from axis names to values")

    provided = set(coordinates)
    expected = set(axis_names)
    def sort_key(value):
        return type(value).__name__, repr(value)

    missing = sorted(expected - provided, key=sort_key)
    extra = sorted(provided - expected, key=sort_key)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing!r}")
        if extra:
            details.append(f"extra={extra!r}")
        detail_text = ", ".join(details)
        raise KeyError(f"coordinate keys do not match grid axes ({detail_text})")

    return tuple(coordinates[name] for name in axis_names)


def _fractional_coordinate(axis_values, kind, query):
    """Map physical coordinates to fractional array indices and a valid mask."""
    query = jnp.asarray(query)
    lower = axis_values[0]
    upper = axis_values[-1]
    valid = jnp.logical_and(jnp.isfinite(query), query >= lower)
    valid = jnp.logical_and(valid, query <= upper)
    if kind == "singleton":
        valid = jnp.logical_and(valid, query == lower)

    # Sanitizing before arithmetic prevents NaN/inf query values from leaking
    # through gradients of an unselected ``where`` branch.
    safe_query = jnp.where(valid, query, lower)

    if kind == "singleton":
        fractional = jnp.zeros_like(safe_query, dtype=jnp.result_type(safe_query, 1.0))
    elif kind == "regular":
        fractional = (safe_query - lower) * (axis_values.size - 1) / (upper - lower)
    elif kind == "nonuniform":
        lower_index = jnp.searchsorted(axis_values, safe_query, side="right") - 1
        lower_index = jnp.clip(lower_index, 0, axis_values.size - 2)
        interval_lower = axis_values[lower_index]
        interval_upper = axis_values[lower_index + 1]
        weight = (safe_query - interval_lower) / (interval_upper - interval_lower)
        fractional = lower_index.astype(weight.dtype) + weight
    else:  # Static metadata is validated at construction.
        raise ValueError(f"unknown axis kind {kind!r}")

    return fractional, valid
