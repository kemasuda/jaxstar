"""Small analytic checks for interpolation over axes, carrying trailing blocks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxstar.grid import Field, RectilinearGrid


def coefficients(payload_shape):
    offset = np.arange(np.prod(payload_shape), dtype=np.float32).reshape(payload_shape) + 2
    return offset, 1 + 0.2 * offset, -0.5 + 0.1 * offset, 0.1 * offset


def expected_payload(x, y, payload_shape):
    x, y = np.broadcast_arrays(x, y)
    x = x.reshape(x.shape + (1,) * len(payload_shape))
    y = y.reshape(y.shape + (1,) * len(payload_shape))
    offset, cx, cy, cxy = coefficients(payload_shape)
    return offset + cx * x + cy * y + cxy * x * y


@pytest.fixture(params=[(3,), (2, 3)], ids=["vector", "matrix"])
def payload_grid(request):
    shape = request.param
    payload_dims = ("pixel",) if len(shape) == 1 else ("region", "pixel")
    x = np.array([0, 1, 2], dtype=np.float32)
    y = np.array([-1, 0, 2], dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    values = expected_payload(xx, yy, shape)
    return RectilinearGrid(
        axes={"x": x, "y": y},
        fields={
            "scalar": Field(xx + 2 * yy, dims=("x", "y")),
            "block": Field(values, dims=("x", "y"), payload_dims=payload_dims),
            "reordered": Field(
                2 * values.swapaxes(0, 1), dims=("y", "x"), payload_dims=payload_dims
            ),
        },
    )


@pytest.mark.parametrize("point", [(1.0, 0.0), (0.25, 0.5), (0.0, -1.0), (2.0, 2.0)],
                         ids=["node", "interior", "lower", "upper"])
def test_scalar_query_keeps_payload_and_axis_order(payload_grid, point):
    result = payload_grid.interpolate(dict(zip(("x", "y"), point)))
    shape = payload_grid.field("block").values.shape[2:]
    expected = expected_payload(*point, shape)

    assert tuple(result) == ("scalar", "block", "reordered")
    assert result["scalar"].shape == ()
    assert result["block"].shape == shape
    np.testing.assert_allclose(result["scalar"], point[0] + 2 * point[1])
    np.testing.assert_allclose(result["block"], expected, rtol=1e-6)
    np.testing.assert_allclose(result["reordered"], 2 * expected, rtol=1e-6)


def test_batch_and_broadcast_query_shapes_eager_and_jit(payload_grid):
    @jax.jit
    def evaluate(grid, x, y):
        return grid.interpolate({"y": y, "x": x}, keys=("reordered", "block", "scalar"))

    shape = payload_grid.field("block").values.shape[2:]
    for x, y in [
        (jnp.array([0.25, 1.5]), jnp.array([0.5, -0.5])),
        (jnp.array([[0.25], [1.5]]), jnp.array([[-0.5, 0.5, 1.5]])),
    ]:
        expected = expected_payload(x, y, shape)
        eager = payload_grid.interpolate({"x": x, "y": y})
        compiled = evaluate(payload_grid, x, y)
        assert tuple(compiled) == ("reordered", "block", "scalar")
        assert compiled["block"].shape == np.broadcast_shapes(x.shape, y.shape) + shape
        for result in (eager, compiled):
            np.testing.assert_allclose(result["block"], expected, rtol=2e-6)
            np.testing.assert_allclose(result["reordered"], 2 * expected, rtol=2e-6)
            np.testing.assert_allclose(result["scalar"], np.asarray(x) + 2 * np.asarray(y))


def test_payload_value_and_coordinate_gradients(payload_grid):
    def objective(grid, point):
        value = grid.interpolate({"x": point[0], "y": point[1]}, keys="block")["block"]
        return jnp.sum(value**2)

    point = jnp.array([0.25, 0.5], dtype=jnp.float32)
    shape = payload_grid.field("block").values.shape[2:]
    values = expected_payload(*point, shape)
    _, cx, cy, cxy = coefficients(shape)
    expected_grad = [np.sum(2 * values * (cx + cxy * point[1])),
                     np.sum(2 * values * (cy + cxy * point[0]))]

    for evaluate in (jax.value_and_grad(objective, argnums=1),
                     jax.jit(jax.value_and_grad(objective, argnums=1))):
        value, grad = evaluate(payload_grid, point)
        np.testing.assert_allclose(value, np.sum(values**2), rtol=2e-6)
        np.testing.assert_allclose(grad, expected_grad, rtol=2e-6)


def test_broadcast_gradients_reduce_over_queries_and_payload(payload_grid):
    y = jnp.array([[-0.5, 0.5, 1.5]], dtype=jnp.float32)

    def objective(x):
        return jnp.sum(payload_grid.interpolate({"x": x, "y": y}, keys="block")["block"])

    shape = payload_grid.field("block").values.shape[2:]
    _, cx, _, cxy = coefficients(shape)
    expected = sum(np.sum(cx + cxy * yy) for yy in np.asarray(y[0]))
    actual = jax.jit(jax.grad(objective))(jnp.array([[0.25], [1.5]], dtype=jnp.float32))
    np.testing.assert_allclose(actual, np.full((2, 1), expected), rtol=2e-6)


def test_payload_vmap_matches_native_batching(payload_grid):
    points = jnp.array([[0.25, 0.5], [1.5, -0.5]], dtype=jnp.float32)
    mapped = jax.jit(jax.vmap(lambda point: payload_grid.interpolate(
        {"x": point[0], "y": point[1]}, keys="block"
    )["block"]))(points)
    native = payload_grid.interpolate({"x": points[:, 0], "y": points[:, 1]}, keys="block")
    np.testing.assert_allclose(mapped, native["block"], rtol=2e-6)


@pytest.mark.parametrize("fill", [-np.inf, -99.0], ids=["default", "custom"])
def test_payload_outside_mask_is_per_query_and_gradients_are_safe(payload_grid, fill):
    grid = RectilinearGrid(
        axes={name: payload_grid.axis(name) for name in payload_grid.axis_names},
        fields={"block": payload_grid.field("block")}, fill_value=fill,
    )
    x = jnp.array([0.25, -0.1, 2.1, jnp.nan, jnp.inf, -jnp.inf, 1.0])
    y = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2.1])
    result = jax.jit(lambda g, xx, yy: g.interpolate({"x": xx, "y": yy}))(grid, x, y)
    shape = grid.field("block").values.shape[2:]
    np.testing.assert_allclose(result["block"][0], expected_payload(0.25, 0.5, shape), rtol=1e-6)
    np.testing.assert_array_equal(result["block"][1:], np.full((6,) + shape, fill))

    def objective(point):
        return jnp.sum(grid.interpolate({"x": point[0], "y": point[1]})["block"])

    gradients = jax.jit(jax.vmap(jax.grad(objective)))(jnp.stack((x[1:], y[1:]), axis=1))
    np.testing.assert_array_equal(gradients, np.zeros((6, 2)))


def test_grid_payload_metadata_is_static_and_values_are_dynamic(payload_grid):
    field = payload_grid.field("block")
    assert isinstance(field.payload_dims, tuple)
    assert field.payload_dims == payload_grid.field_payload_dims[1]
    assert payload_grid.field("scalar").payload_dims == ()
    leaves, structure = jax.tree_util.tree_flatten(payload_grid)
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)
    restored = jax.tree_util.tree_unflatten(structure, leaves)
    assert restored.field_payload_dims == payload_grid.field_payload_dims
    changed = RectilinearGrid(
        axes={name: restored.axis(name) for name in restored.axis_names},
        fields={name: Field(
            restored.field(name).values + 5,
            restored.field(name).dims,
            payload_dims=restored.field(name).payload_dims,
        ) for name in restored.field_names},
    )
    assert jax.tree_util.tree_structure(changed) == structure
    traces = []

    @jax.jit
    def evaluate(grid):
        traces.append(True)
        return grid.interpolate({"x": 0.25, "y": 0.5}, keys="block")["block"]

    old, new = evaluate(restored), evaluate(changed)
    np.testing.assert_allclose(new, old + 5, rtol=2e-6)
    assert len(traces) == 1


@pytest.mark.parametrize("singleton_only", [False, True])
def test_payload_singleton_axis(singleton_only):
    axes = {"fixed": np.array([4], dtype=np.float32)}
    dims = ("fixed",)
    values = np.array([[2, 3, 5]], dtype=np.float32)
    coordinates = {"fixed": 4.0}
    if not singleton_only:
        axes["x"] = np.array([0, 1, 2], dtype=np.float32)
        dims += ("x",)
        values = values[:, None, :] + np.arange(3, dtype=np.float32)[None, :, None]
        coordinates["x"] = 0.25
    grid = RectilinearGrid(axes=axes, fields={"block": Field(values, dims, payload_dims=("pixel",))})

    @jax.jit
    def evaluate(fixed):
        return grid.interpolate({**coordinates, "fixed": fixed})["block"]

    np.testing.assert_allclose(evaluate(4.0), [2, 3, 5] if singleton_only else [2.25, 3.25, 5.25])
    assert np.all(np.isneginf(evaluate(4.1)))
    np.testing.assert_allclose(jax.grad(lambda fixed: evaluate(fixed).sum())(4.0), 0)


@pytest.mark.parametrize("axis", [[0, 1, 2], [0, 1, 3]], ids=["regular", "nonuniform"])
def test_payload_piecewise_gradients_and_endpoints_match_scalar_path(axis):
    values = np.array([0, 1, 5], dtype=np.float32)
    grid = RectilinearGrid(axes={"x": np.array(axis, dtype=np.float32)}, fields={
        "scalar": Field(values, ("x",)),
        "block": Field(values[:, None], ("x",), payload_dims=("pixel",)),
    })
    points = jnp.array([axis[0], 0.5, axis[1], (axis[1] + axis[2]) / 2, axis[2]])

    def evaluate(x, key):
        return grid.interpolate({"x": x}, keys=key)[key].sum()

    for transform in (lambda f: f, jax.grad):
        scalar = jax.jit(jax.vmap(transform(lambda x: evaluate(x, "scalar"))))(points)
        block = jax.jit(jax.vmap(transform(lambda x: evaluate(x, "block"))))(points)
        np.testing.assert_allclose(block, scalar, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("field_dtype,coordinate_dtype", [
    (np.float32, np.float64), (np.float64, np.float32), (np.float64, np.float64),
], ids=["field32-query64", "field64-query32", "field64-query64"])
@pytest.mark.parametrize("query", [2.0, 2.123456789], ids=["midpoint", "non-dyadic"])
def test_payload_mixed_dtypes_retain_field_precision(field_dtype, coordinate_dtype, query):
    with jax.experimental.enable_x64():
        axis = np.array([0, 1, 3], dtype=coordinate_dtype)
        values = np.array([[1.000000001, 2], [4.000000003, 8], [10.000000009, 20]], dtype=field_dtype)
        grid = RectilinearGrid(axes={"x": axis}, fields={
            "block": Field(values, ("x",), payload_dims=("pixel",)),
            "scalar": Field(values[:, 0], ("x",)),
        })
        x = jnp.asarray(query, dtype=coordinate_dtype)
        weight = (float(x) - 1.0) / 2.0
        expected = (1.0 - weight) * values[1] + weight * values[2]
        tolerance = 2e-6 if np.float32 in (field_dtype, coordinate_dtype) else 1e-12

        @jax.jit
        def evaluate(g, query):
            return g.interpolate({"x": query})

        for result in (grid.interpolate({"x": x}), evaluate(grid, x)):
            assert result["block"].dtype == field_dtype
            assert result["scalar"].dtype == field_dtype
            np.testing.assert_allclose(result["block"], expected, rtol=tolerance)
            np.testing.assert_allclose(result["block"][0], result["scalar"], rtol=tolerance)
        assert grid.field("block").values.dtype == field_dtype
        grad = jax.jit(jax.grad(lambda query: evaluate(grid, query)["block"].sum()))(x)
        assert grad.dtype == coordinate_dtype
        expected_grad = np.asarray(np.sum((values[2] - values[1]) / 2), dtype=coordinate_dtype)
        np.testing.assert_allclose(grad, expected_grad, rtol=tolerance)


@pytest.mark.parametrize("shape,dims,payload_dims,message", [
    ((2, 3), ("x",), (), "shape"),
    ((2,), ("x",), ("pixel",), "shape"),
    ((2, 3, 4), ("x",), ("pixel",), "shape"),
    ((3, 2), ("x",), ("pixel",), "shape"),
    ((2, 3), ("x", "pixel"), (), "unknown dimensions"),
    ((2, 3), ("other",), ("pixel",), "unknown dimensions"),
    ((2, 3), (), ("pixel",), "every grid axis"),
    ((2, 3), ("x",), ("x",), "distinct from grid axes"),
    ((2, 3, 4), ("x",), ("pixel", "pixel"), "unique"),
    ((2, 3), ("x",), ("",), "non-empty strings"),
    ((2, 3), ("x",), (123,), "non-empty strings"),
])
def test_payload_schema_validation(shape, dims, payload_dims, message):
    with pytest.raises(ValueError, match=message):
        RectilinearGrid(axes={"x": np.array([0, 1], dtype=np.float32)}, fields={
            "block": Field(np.zeros(shape, dtype=np.float32), dims, payload_dims=payload_dims),
        })


def test_nonfinite_payload_can_load_without_changing_scalar_invalid_cell_behavior():
    values = np.array([1, -np.inf], dtype=np.float32)
    grid = RectilinearGrid(axes={"x": np.array([0, 1], dtype=np.float32)}, fields={
        "scalar": Field(values, ("x",)),
        "block": Field(values[:, None], ("x",), payload_dims=("pixel",)),
    })
    assert np.isneginf(grid.field("block").values[-1, 0])
    result = grid.interpolate({"x": 0.0})
    # No new mask or invalid-cell repair: the existing 0 * -inf result remains.
    assert np.isnan(result["scalar"])
    assert np.isnan(result["block"][0])
