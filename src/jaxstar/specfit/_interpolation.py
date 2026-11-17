"""Local, shape-preserving Teff interpolation of spectral payloads in JAX.

Other atmosphere axes are interpolated by the existing multilinear kernel
*before* computing PCHIP slopes. Precomputing slopes at atmosphere corners
would give a different result because the slope limiter is nonlinear.
"""

import jax.numpy as jnp


def _interior_slope(h0, h1, d0, d1):
    """Fritsch--Butland weighted harmonic mean, safe at flat/extremal pixels."""
    same_sign = ((d0 > 0) & (d1 > 0)) | ((d0 < 0) & (d1 < 0))
    # Sanitize before division: a where around an unsafe reciprocal can still
    # produce NaN in reverse-mode AD. Ratios also avoid squaring tiny secants
    # in the derivative of 1/d, especially for nearly flat float32 continua.
    a, b = jnp.where(same_sign, d0, 1), jnp.where(same_sign, d1, 1)
    first_small = jnp.abs(a) <= jnp.abs(b)
    small, large = jnp.where(first_small, a, b), jnp.where(first_small, b, a)
    w0 = (2 * h1 + h0) / (3 * (h0 + h1))
    wsmall = jnp.where(first_small, w0, 1 - w0)
    slope = small / (wsmall + (1 - wsmall) * (small / large))
    return jnp.where(same_sign, slope, 0)


def _edge_slope(h0, h1, d0, d1):
    """One-sided three-point slope with the PCHIP shape limiter."""
    slope = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    slope = jnp.where(jnp.sign(slope) != jnp.sign(d0), 0, slope)
    limit = (jnp.sign(d0) != jnp.sign(d1)) & (jnp.abs(slope) > 3 * jnp.abs(d0))
    return jnp.where(limit, 3 * d0, slope)


def interpolate_flux(grid, coordinates, *, teff_interp="cubic"):
    """Return flux with shape broadcast(query) + (region, pixel).

    ``linear`` delegates unchanged to RectilinearGrid. ``cubic`` (default) uses
    actual Teff coordinates and at most four native slices, retains the grid's
    fill policy, and reduces to linear/constant for two/one Teff nodes. Grids
    without a named teff axis retain multilinear interpolation. The mode is
    static under JIT. All numerical evaluation, including slopes, is JAX-native.
    PCHIP is C1 in Teff at fixed other coordinates, not generally C2; slope
    limiting can introduce kinks with respect to other atmosphere coordinates.
    """
    if teff_interp == "linear":
        return grid.interpolate(coordinates)["flux"]
    if teff_interp != "cubic":
        raise ValueError("teff_interp must be 'linear' or 'cubic'")
    if "teff" not in grid.axis_names:
        return grid.interpolate(coordinates)["flux"]
    if set(coordinates) != set(grid.axis_names):
        raise ValueError("coordinates must supply exactly the named grid axes")
    temperatures = grid.axis("teff")
    if temperatures.size < 3:
        return grid.interpolate(coordinates)["flux"]

    queries = jnp.broadcast_arrays(*(jnp.asarray(coordinates[n]) for n in grid.axis_names))
    safe, valid = {}, jnp.asarray(True)
    for name, query in zip(grid.axis_names, queries):
        axis = grid.axis(name)
        inside = jnp.isfinite(query) & (query >= axis[0]) & (query <= axis[-1])
        valid = valid & inside
        safe[name] = jnp.where(inside, query, axis[0])
    teff = safe["teff"]
    lower = jnp.clip(jnp.searchsorted(temperatures, teff, side="right") - 1,
                     0, temperatures.size - 2)
    indices = jnp.clip(lower[..., None] + jnp.arange(-1, 3), 0, temperatures.size - 1)
    nodes = temperatures[indices]
    local = {name: value[..., None] for name, value in safe.items()}
    local["teff"] = nodes
    values = grid.interpolate(local)["flux"]
    y0, y1, y2, y3 = (values[..., i, :, :] for i in range(4))

    def payload(value):
        return value.astype(values.dtype)[..., None, None]

    h = nodes[..., 2] - nodes[..., 1]
    # Repeated edge indices are placeholders; keep every evaluated branch safe.
    h0 = payload(jnp.where(lower == 0, h, nodes[..., 1] - nodes[..., 0]))
    h1 = payload(h)
    h2 = payload(jnp.where(lower == temperatures.size - 2, h, nodes[..., 3] - nodes[..., 2]))
    d0, d1, d2 = (y1 - y0) / h0, (y2 - y1) / h1, (y3 - y2) / h2
    left = jnp.where((lower == 0)[..., None, None],
                     _edge_slope(h1, h2, d1, d2), _interior_slope(h0, h1, d0, d1))
    right = jnp.where((lower == temperatures.size - 2)[..., None, None],
                      _edge_slope(h1, h0, d1, d0), _interior_slope(h1, h2, d1, d2))
    u = payload((teff - nodes[..., 1]) / h)
    blend = u * u * (3 - 2 * u)
    flux = ((1 - blend) * y1 + blend * y2
            + h1 * u * (1 - u) * ((1 - u) * left - u * right))
    return jnp.where(valid[..., None, None], flux, grid.fill_value.astype(flux.dtype))
