"""Private association of a numerical MIST grid with its kind and provenance."""

from dataclasses import dataclass

import jax

from jaxstar.grid import RectilinearGrid


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class _MistGridData:
    """Loader result; interpolation stays in RectilinearGrid.

    Only isochrones are implemented. Their axes are log10(age/yr), initial
    FeH (dex), and EEP, named age/feh/eep in the numerical grid. Numerical
    arrays are dynamic leaves; kind and source paths are static provenance.
    """

    grid: RectilinearGrid
    kind: str = "isochrone"
    sources: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.grid, RectilinearGrid):
            raise TypeError("grid must be a RectilinearGrid")
        if self.kind != "isochrone":
            raise ValueError("only kind='isochrone' is currently supported")
        if set(self.grid.axis_names) != {"age", "feh", "eep"}:
            raise ValueError("isochrone grid axes must be age, feh, and eep")
        if any(self.grid.field_payload_dims):
            raise ValueError("isochrone fields must be scalar at each grid node")
        if isinstance(self.sources, str) or any(not isinstance(s, str) for s in self.sources):
            raise TypeError("sources must be a sequence of path strings")
        object.__setattr__(self, "sources", tuple(self.sources))

    def tree_flatten(self):
        return (self.grid,), (self.kind, self.sources)

    @classmethod
    def tree_unflatten(cls, metadata, children):
        # JAX may supply abstract placeholders: validate only on construction.
        data = object.__new__(cls)
        object.__setattr__(data, "grid", children[0])
        object.__setattr__(data, "kind", metadata[0])
        object.__setattr__(data, "sources", metadata[1])
        return data
