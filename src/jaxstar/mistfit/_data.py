"""Private association of a numerical MIST grid with its kind and provenance."""

from dataclasses import dataclass
import json

import jax
import jax.numpy as jnp

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
    # Optional per-field support, in grid.field_names order. Invalid values in
    # the numerical grid are finite placeholders; only MistModel applies masks.
    validity: tuple | None = None
    # Sorted union of native primary EEP labels; plotting metadata only.
    primary_eeps: tuple[int, ...] = ()
    # Small immutable provenance record, without the per-track coverage lists.
    metadata_json: str = "{}"

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
        object.__setattr__(self, "primary_eeps", tuple(self.primary_eeps))
        if not isinstance(self.metadata_json, str) or not isinstance(json.loads(self.metadata_json), dict):
            raise ValueError("MIST metadata must be a JSON object string")
        if self.kind == "track":
            age = self.grid.field("logage")
            if age.dims != axes["track"]:
                raise ValueError("track logage dimensions must be initial_mass, initial_feh, eep")
        if self.validity is not None:
            if self.kind != "track" or len(self.validity) != len(self.grid.field_names):
                raise ValueError("validity must contain one mask per track field")
            masks = tuple(jnp.asarray(mask, dtype=bool) for mask in self.validity)
            if any(mask.shape != self.grid.field(name).values.shape
                   for name, mask in zip(self.grid.field_names, masks)):
                raise ValueError("track validity masks must match their fields")
            object.__setattr__(self, "validity", masks)

    def tree_flatten(self):
        return (self.grid, self.validity), (self.kind, self.sources, self.primary_eeps, self.metadata_json)

    @classmethod
    def tree_unflatten(cls, metadata, children):
        # JAX may supply abstract placeholders: validate only on construction.
        data = object.__new__(cls)
        object.__setattr__(data, "grid", children[0])
        object.__setattr__(data, "kind", metadata[0])
        object.__setattr__(data, "sources", metadata[1])
        object.__setattr__(data, "primary_eeps", metadata[2])
        object.__setattr__(data, "metadata_json", metadata[3])
        object.__setattr__(data, "validity", children[1])
        return data
