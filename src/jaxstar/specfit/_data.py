"""Private association of numerical flux grids with their wavelength payload."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from jaxstar.grid import RectilinearGrid


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class _SpectralLibrary:
    """Loader result; interpolate ``grid`` directly, without another wrapper.

    ``wavelength`` is in Angstrom, once per (region, pixel), never per atmosphere
    node. Region identifiers may repeat (e.g. two segments of the same order).
    This type stays private until a downstream consumer establishes its API.
    """

    grid: RectilinearGrid
    wavelength: object
    regions: tuple
    library: str
    wavelength_medium: str = "unknown"
    flux_kind: str = "as_stored"
    sources: tuple[str, ...] = ()
    wavelength_unit: str = "angstrom"

    def __post_init__(self):
        if not isinstance(self.grid, RectilinearGrid):
            raise TypeError("grid must be a RectilinearGrid")
        if self.grid.field_names != ("flux",):
            raise ValueError("a spectral library must contain exactly the flux field")
        field = self.grid.field("flux")
        if field.payload_dims != ("region", "pixel"):
            raise ValueError("flux payload dimensions must be ('region', 'pixel')")
        input_wavelength = np.array(self.wavelength, copy=True)
        if not np.issubdtype(input_wavelength.dtype, np.floating):
            raise TypeError("wavelength must have a floating-point dtype")
        wavelength = jnp.asarray(input_wavelength)
        _validate_wavelength(np.asarray(wavelength), ndim=2)
        if wavelength.shape != field.values.shape[len(field.dims):]:
            raise ValueError("wavelength shape must match the flux (region, pixel) payload")
        regions = tuple(self.regions)
        if len(regions) != wavelength.shape[0]:
            raise ValueError("one region identifier is required per wavelength row")
        if any(not isinstance(region, (str, int)) for region in regions):
            raise ValueError("region identifiers must be strings or integers")
        if not isinstance(self.library, str) or not self.library:
            raise ValueError("library must be a non-empty provenance label")
        if self.wavelength_unit != "angstrom":
            raise ValueError("spectral wavelengths must be in angstrom")
        if self.wavelength_medium not in {"unknown", "air", "vacuum"}:
            raise ValueError("wavelength_medium must be 'unknown', 'air' or 'vacuum'")
        if self.flux_kind not in {"as_stored", "normalized", "unnormalized", "median_scaled"}:
            raise ValueError("unsupported flux_kind")
        object.__setattr__(self, "wavelength", wavelength)
        object.__setattr__(self, "regions", regions)
        object.__setattr__(self, "sources", tuple(str(source) for source in self.sources))

    def tree_flatten(self):
        children = (self.grid, self.wavelength)
        metadata = (self.regions, self.library, self.wavelength_medium,
                    self.flux_kind, self.sources, self.wavelength_unit)
        return children, metadata

    @classmethod
    def tree_unflatten(cls, metadata, children):
        # JAX may supply tracers or abstract placeholders; validation is only
        # performed by loaders, before entering compiled numerical code.
        regions, library, medium, flux_kind, sources, unit = metadata
        grid, wavelength = children
        result = object.__new__(cls)
        for name, value in (
            ("grid", grid), ("wavelength", wavelength), ("regions", regions),
            ("library", library), ("wavelength_medium", medium),
            ("flux_kind", flux_kind), ("sources", sources), ("wavelength_unit", unit),
        ):
            object.__setattr__(result, name, value)
        return result


def _validate_wavelength(wavelength, *, ndim):
    if wavelength.ndim != ndim or any(size == 0 for size in wavelength.shape):
        raise ValueError(f"wavelength must be a non-empty {ndim}-dimensional array")
    if not np.all(np.isfinite(wavelength)) or np.any(wavelength <= 0):
        raise ValueError("wavelength must contain finite positive values")
    if np.any(np.diff(wavelength, axis=-1) <= 0):
        raise ValueError("wavelength must be strictly increasing within each region")
