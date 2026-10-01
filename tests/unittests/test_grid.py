from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxstar.grid import Axis, Field, GridResult, RectilinearGrid


@pytest.fixture
def named_grid():
    """A small grid with one regular and one visibly nonuniform axis."""
    x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    y = np.array([-1.0, 0.0, 2.0, 5.0], dtype=np.float32)
    x_values, y_values = np.meshgrid(x, y, indexing="ij")

    linear = 10.0 + 2.0 * x_values + 3.0 * y_values
    return RectilinearGrid(
        axes={"x": Axis(x), "y": Axis(y)},
        fields={
            "linear": Field(linear, dims=("x", "y")),
            # A different dimension order demonstrates why dims are explicit.
            "twice": Field((2.0 * linear).T, dims=("y", "x")),
        },
    )


def test_interpolate_returns_all_fields_in_schema_order(named_grid):
    result = named_grid.interpolate({"y": 0.5, "x": 0.5})

    assert isinstance(result, GridResult)
    assert tuple(result) == ("linear", "twice")
    np.testing.assert_allclose(result["linear"], 12.5)
    np.testing.assert_allclose(result["twice"], 25.0)


def test_keys_select_fields_in_requested_order_without_mutating_grid(named_grid):
    @jax.jit
    def select(grid):
        return grid.interpolate(
            {"x": 0.5, "y": 0.5},
            keys=("twice", "linear"),
        )

    result = select(named_grid)

    assert tuple(result) == ("twice", "linear")
    assert named_grid.field_names == ("linear", "twice")
    np.testing.assert_allclose(result["twice"], 25.0)
    np.testing.assert_allclose(result["linear"], 12.5)


def test_invalid_key_selection_is_rejected(named_grid):
    with pytest.raises(KeyError, match="unknown"):
        named_grid.interpolate({"x": 0.5, "y": 0.5}, keys=("missing",))

    with pytest.raises(ValueError, match="duplicates"):
        named_grid.interpolate(
            {"x": 0.5, "y": 0.5},
            keys=("linear", "linear"),
        )


def test_custom_constant_fill_value_is_cast_to_the_field_dtype(named_grid):
    custom_grid = RectilinearGrid(
        axes={name: named_grid.axis(name) for name in named_grid.axis_names},
        fields={
            name: named_grid.field(name) for name in named_grid.field_names
        },
        fill_value=-999,
    )

    result = custom_grid.interpolate({"x": -0.1, "y": 0.5}, keys="linear")

    np.testing.assert_allclose(result["linear"], -999.0)
    assert result["linear"].dtype == jnp.float32


def test_unsupported_boundary_and_incorrect_axis_kind_are_rejected():
    values = np.array([0.0, 1.0, 3.0], dtype=np.float32)

    with pytest.raises(ValueError, match="boundary"):
        RectilinearGrid(
            axes={"x": values},
            fields={"value": Field(values, dims=("x",))},
            boundary="nearest",
        )

    with pytest.raises(ValueError, match="not equally spaced"):
        RectilinearGrid(
            axes={"x": Axis(values, kind="regular")},
            fields={"value": Field(values, dims=("x",))},
        )


def test_regular_and_nonuniform_axes_use_physical_coordinates(named_grid):
    result = named_grid.interpolate({"x": 0.5, "y": 0.5}, keys="linear")

    assert named_grid.axis_kinds == ("regular", "nonuniform")
    # y=0.5 is one quarter of the way from the y nodes 0 to 2.
    np.testing.assert_allclose(result["linear"], 12.5)


def test_nonuniform_axis_has_piecewise_values_and_gradients():
    axis = np.array([0.0, 1.0, 3.0], dtype=np.float32)
    values = np.array([0.0, 1.0, 5.0], dtype=np.float32)
    grid = RectilinearGrid(
        axes={"x": axis},
        fields={"value": Field(values, dims=("x",))},
    )

    def evaluate(x):
        return grid.interpolate({"x": x}, keys="value")["value"]

    np.testing.assert_allclose(evaluate(0.5), 0.5)
    np.testing.assert_allclose(evaluate(2.0), 3.0)
    np.testing.assert_allclose(jax.grad(evaluate)(0.5), 1.0)
    np.testing.assert_allclose(jax.grad(evaluate)(2.0), 2.0)


def test_float32_linspace_is_detected_as_regular_despite_roundoff():
    axis = np.linspace(8.0, 10.0, 48, dtype=np.float32)
    grid = RectilinearGrid(
        axes={"x": axis},
        fields={"value": Field(axis.copy(), dims=("x",))},
    )

    assert grid.axis_kinds == ("regular",)


def test_high_offset_axis_falls_back_to_exact_nonuniform_mapping():
    axis = np.array([1.0e8, 1.0e8 + 8.0, 1.0e8 + 24.0], dtype=np.float32)
    values = np.array([0.0, 100.0, 200.0], dtype=np.float32)
    grid = RectilinearGrid(
        axes={"x": axis},
        fields={"value": Field(values, dims=("x",))},
    )

    result = grid.interpolate({"x": axis[1]}, keys="value")

    assert grid.axis_kinds == ("nonuniform",)
    np.testing.assert_allclose(result["value"], 100.0)


def test_long_axis_does_not_hide_a_locally_nonuniform_node():
    axis = np.arange(10_000, dtype=np.float32)
    axis[-2] += np.float32(0.05)
    values = np.arange(axis.size, dtype=np.float32)
    grid = RectilinearGrid(
        axes={"wavelength": axis},
        fields={"value": Field(values, dims=("wavelength",))},
    )

    result = grid.interpolate({"wavelength": axis[-2]}, keys="value")

    assert grid.axis_kinds == ("nonuniform",)
    np.testing.assert_allclose(result["value"], values[-2])


def test_axis_is_revalidated_after_jax_dtype_canonicalization():
    # With x64 disabled, JAX stores these distinct float64 values as equal
    # float32 values.  The grid must reject the effective coordinates rather
    # than accepting an axis that later divides by zero.
    axis = np.array([1.0, 1.0 + 1.0e-9], dtype=np.float64)

    if jax.config.read("jax_enable_x64"):
        pytest.skip("the values remain distinct when JAX x64 is enabled")

    with pytest.raises(ValueError, match="strictly increasing"):
        RectilinearGrid(
            axes={"x": axis},
            fields={
                "value": Field(np.array([0.0, 1.0]), dims=("x",)),
            },
        )


@pytest.mark.parametrize(
    ("coordinates", "expected"),
    [
        ({"x": 0.0, "y": -1.0}, 7.0),
        ({"x": 2.0, "y": 5.0}, 29.0),
    ],
    ids=["lower", "upper"],
)
def test_exact_domain_boundaries_are_included(named_grid, coordinates, expected):
    result = named_grid.interpolate(coordinates, keys="linear")

    np.testing.assert_allclose(result["linear"], expected)


@pytest.mark.parametrize(
    "coordinates",
    [
        {"x": -0.1, "y": 0.5},
        {"x": 2.1, "y": 0.5},
        {"x": 0.5, "y": -1.1},
        {"x": 0.5, "y": 5.1},
        {"x": np.nan, "y": 0.5},
        {"x": np.inf, "y": 0.5},
        {"x": -np.inf, "y": 0.5},
    ],
    ids=[
        "x-below",
        "x-above",
        "y-below",
        "y-above",
        "nan",
        "positive-infinity",
        "negative-infinity",
    ],
)
def test_out_of_domain_coordinates_use_constant_fill(named_grid, coordinates):
    result = named_grid.interpolate(coordinates, keys="linear")

    assert np.isneginf(result["linear"])


def test_validity_mask_is_applied_per_broadcast_query(named_grid):
    result = named_grid.interpolate(
        {"x": jnp.array([0.5, 2.1]), "y": 0.5},
        keys="linear",
    )

    np.testing.assert_allclose(result["linear"][0], 12.5)
    assert np.isneginf(result["linear"][1])


def test_coordinate_arrays_broadcast_to_the_output_shape(named_grid):
    x = jnp.array([[0.5], [1.5]])
    y = jnp.array([[-0.5, 0.5, 1.5]])

    result = named_grid.interpolate(
        {"x": x, "y": y},
        keys=("linear", "twice"),
    )

    expected = np.array(
        [
            [9.5, 12.5, 15.5],
            [11.5, 14.5, 17.5],
        ]
    )
    assert result["linear"].shape == (2, 3)
    np.testing.assert_allclose(result["linear"], expected)
    np.testing.assert_allclose(result["twice"], 2.0 * expected)


def test_singleton_axis_accepts_only_its_fixed_coordinate():
    x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    fixed = np.array([4.0], dtype=np.float32)
    values = 10.0 + 2.0 * x[:, None] + 3.0 * fixed[None, :]
    grid = RectilinearGrid(
        axes={"x": x, "fixed": fixed},
        fields={"value": Field(values, dims=("x", "fixed"))},
    )

    inside = grid.interpolate({"x": 0.5, "fixed": 4.0})["value"]
    outside = grid.interpolate({"x": 0.5, "fixed": 4.1})["value"]

    assert grid.axis_kinds == ("regular", "singleton")
    np.testing.assert_allclose(inside, 23.0)
    assert np.isneginf(outside)


def test_grid_is_a_jittable_differentiable_pytree(named_grid):
    @jax.jit
    def evaluate(grid, point):
        return grid.interpolate(
            {"x": point[0], "y": point[1]},
            keys="linear",
        )["linear"]

    point = jnp.array([0.5, 0.5], dtype=jnp.float32)

    np.testing.assert_allclose(evaluate(named_grid, point), 12.5)
    np.testing.assert_allclose(
        jax.grad(lambda query: evaluate(named_grid, query))(point),
        jnp.array([2.0, 3.0], dtype=jnp.float32),
    )


def test_interpolation_works_under_vmap(named_grid):
    points = jnp.array([[0.5, 0.5], [1.5, 1.5]], dtype=jnp.float32)

    values = jax.vmap(
        lambda point: named_grid.interpolate(
            {"x": point[0], "y": point[1]},
            keys="linear",
        )["linear"]
    )(points)

    np.testing.assert_allclose(values, [12.5, 17.5])


def test_float32_field_dtype_is_preserved(named_grid):
    result = named_grid.interpolate({"x": 0.5, "y": 0.5}, keys="linear")

    assert result["linear"].dtype == jnp.float32


@pytest.mark.parametrize(
    ("axis", "message"),
    [
        (np.empty((0,), dtype=np.float32), "empty"),
        (np.zeros((2, 2), dtype=np.float32), "one-dimensional"),
        (np.array([0.0, np.nan], dtype=np.float32), "finite"),
        (np.array([0.0, 0.0], dtype=np.float32), "strictly increasing"),
        (np.array([1.0, 0.0], dtype=np.float32), "strictly increasing"),
    ],
    ids=["empty", "two-dimensional", "nonfinite", "duplicate", "descending"],
)
def test_invalid_axes_are_rejected(axis, message):
    with pytest.raises(ValueError, match=message):
        RectilinearGrid(
            axes={"x": axis},
            fields={"value": Field(np.zeros(axis.size, dtype=np.float32), ("x",))},
        )


def test_field_shape_and_dimensions_are_validated():
    x = np.array([0.0, 1.0], dtype=np.float32)

    with pytest.raises(ValueError, match="unknown dimensions"):
        RectilinearGrid(
            axes={"x": x},
            fields={"value": Field(np.zeros(2, dtype=np.float32), ("z",))},
        )

    with pytest.raises(ValueError, match="shape"):
        RectilinearGrid(
            axes={"x": x},
            fields={"value": Field(np.zeros(3, dtype=np.float32), ("x",))},
        )


def test_coordinate_keys_and_shapes_are_validated(named_grid):
    with pytest.raises(KeyError, match="missing"):
        named_grid.interpolate({"x": 0.5})

    with pytest.raises(KeyError, match="extra"):
        named_grid.interpolate({"x": 0.5, "y": 0.5, "z": 0.5})

    with pytest.raises(ValueError, match="broadcast-compatible"):
        named_grid.interpolate(
            {"x": jnp.ones(2), "y": jnp.ones(3)},
        )


def test_grid_schema_is_immutable_and_copied_from_input_mappings():
    axis_values = np.array([0.0, 1.0], dtype=np.float32)
    field_values = np.array([0.0, 1.0], dtype=np.float32)
    axes = {"x": axis_values}
    fields = {
        "value": Field(field_values, dims=("x",))
    }
    grid = RectilinearGrid(axes=axes, fields=fields)

    axes["other"] = np.array([0.0], dtype=np.float32)
    fields.clear()
    axis_values[:] = [10.0, 20.0]
    field_values[:] = [10.0, 20.0]

    assert grid.axis_names == ("x",)
    assert grid.field_names == ("value",)
    np.testing.assert_array_equal(grid.axis("x"), [0.0, 1.0])
    np.testing.assert_array_equal(grid.field("value").values, [0.0, 1.0])
    with pytest.raises(FrozenInstanceError):
        grid.axis_names = ("other",)


def test_nonfinite_field_values_are_not_rejected_at_construction():
    # Interpolation semantics next to nonfinite cells are intentionally deferred
    # until the MIST adapter work; this only records that such grids can load.
    grid = RectilinearGrid(
        axes={"x": np.array([0.0, 1.0], dtype=np.float32)},
        fields={
            "value": Field(
                np.array([1.0, -np.inf], dtype=np.float32),
                dims=("x",),
            )
        },
    )

    assert np.isneginf(grid.field("value").values[-1])
