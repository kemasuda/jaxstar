"""Private association of a numerical MIST grid with its kind and provenance."""

from dataclasses import dataclass

import jax

from jaxstar.grid import RectilinearGrid


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class _MistGridData:
    """Loader result; interpolation stays in RectilinearGrid.

    Isochrone axes are age/feh/eep; track axes are initial_mass/initial_feh/eep.
    Numerical arrays are dynamic leaves; kind and source paths are static
    provenance.
    """

    grid: RectilinearGrid
    kind: str = "isochrone"
    sources: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.grid, RectilinearGrid):
            raise TypeError("grid must be a RectilinearGrid")
        axes = {"isochrone": ("age", "feh", "eep"),
                "track": ("initial_mass", "initial_feh", "eep")}
        if self.kind not in axes:
            raise ValueError("kind must be 'isochrone' or 'track'")
        if set(self.grid.axis_names) != set(axes[self.kind]):
            raise ValueError(f"{self.kind} grid axes must be {', '.join(axes[self.kind])}")
        if any(self.grid.field_payload_dims):
            raise ValueError(f"{self.kind} fields must be scalar at each grid node")
        if isinstance(self.sources, str) or any(not isinstance(s, str) for s in self.sources):
            raise TypeError("sources must be a sequence of path strings")
        object.__setattr__(self, "sources", tuple(self.sources))
        if self.kind == "track":
            age = self.grid.field("logage")
            if age.dims != axes["track"]:
                raise ValueError("track logage dimensions must be initial_mass, initial_feh, eep")

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
