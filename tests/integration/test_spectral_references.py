"""Opt-in, read-only comparisons with the supplied frozen jaxspec artifacts.

JAXSPEC_REFERENCE_ROOT=/path/to/jaxspec PYTHONPATH=src python -m pytest \
    tests/integration/test_spectral_references.py -q

The normal suite has no sibling-repository dependency. No old source module is
imported and no reference generator is run. Fractional-pixel evaluation below
is test-only algebra to compare the same physical wavelengths as the oracle.
"""

import hashlib
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxstar.specfit import load_bosz, load_coelho, load_tlusty


ROOT = os.environ.get("JAXSPEC_REFERENCE_ROOT")
pytestmark = pytest.mark.skipif(not ROOT, reason="set JAXSPEC_REFERENCE_ROOT for frozen spectral artifacts")
CASES = {
    "coelho": (load_coelho, (2100, 4100), (8, 3, 2, 0),
               [5812.5, 4.2, -0.15, 0.12]),
    "bosz": (load_bosz, (1680, 3280), (3, 1, 2, 0, 0, 1),
             [5812.5, 4.2, -0.15, 0.10, -0.10, 1.3]),
    "tlusty": (load_tlusty, (9940, 6860), (15, 3, 2),
               [31250, 3.85, -0.12]),
}


@pytest.fixture(scope="module", autouse=True)
def precision():
    with jax.experimental.enable_x64():
        yield


@pytest.fixture(scope="module", params=tuple(CASES))
def real_library(request):
    name = request.param
    loader, windows, node, interior = CASES[name]
    root = Path(ROOT).resolve()
    path = next((root / "characterization" / f"sample_grid_{name}").glob("*.npz"))
    manifest = json.loads((root / "characterization/reference_outputs/manifest.json").read_text())
    for source in (path, root / "src/jaxspec/specgrid.py"):
        with source.open("rb") as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            assert digest.hexdigest() == manifest["sha256"][str(source.relative_to(root))]
    library = loader([path, path], pixel_slices=[slice(start, start + 64) for start in windows])
    with np.load(path, allow_pickle=False) as data:
        raw = data["flux"]
        nodes = np.stack([raw[(*node, slice(start, start + 64))] for start in windows])
        upper = np.stack([raw[(*(size - 1 for size in raw.shape[:-1]), slice(start, start + 64))]
                          for start in windows])
    with np.load(root / "characterization/reference_outputs/frozen_main.npz", allow_pickle=False) as reference:
        outputs = {key: reference[key] for key in reference.files if key.startswith(f"grid_{name}_")}
    yield name, library, np.asarray(interior), outputs, nodes, upper
    jax.clear_caches()


def sample_pixels(flux):
    fractions = jnp.array([[4., 7.3, 14.6, 20.4], [5.2, 8., 17.4, 24.7]])
    lower = jnp.floor(fractions).astype(jnp.int32)
    weight = fractions - lower
    left = jnp.take_along_axis(flux, lower, axis=-1)
    right = jnp.take_along_axis(flux, lower + 1, axis=-1)
    return ((1 - weight) * left + weight * right).astype(flux.dtype)


def evaluate(library, parameters):
    return library.grid.interpolate({name: parameters[i]
                                     for i, name in enumerate(library.grid.axis_names)})["flux"]


def test_real_schema_and_exact_stored_nodes(real_library):
    name, library, _, reference, nodes, _ = real_library
    result = jax.jit(evaluate)(library, reference[f"grid_{name}_node_parameters"])
    assert result.shape == library.wavelength.shape == (2, 64)
    assert result.dtype == (np.float64 if name == "coelho" else np.float32)
    np.testing.assert_allclose(result, nodes, rtol=3e-6, atol=3e-7)


def test_real_interior_and_atmosphere_batch_match_frozen_outputs(real_library):
    name, library, point, reference, _, _ = real_library
    function = lambda lib, p: sample_pixels(evaluate(lib, p))
    for f in (function, jax.jit(function)):
        np.testing.assert_allclose(f(library, point), reference[f"grid_{name}_interior"],
                                   rtol=3e-6, atol=3e-7)
    points = jnp.stack((jnp.asarray(point), jnp.asarray(reference[f"grid_{name}_node_parameters"])))
    result = jax.jit(jax.vmap(function, in_axes=(None, 0)))(library, points)
    np.testing.assert_allclose(result, reference[f"grid_{name}_batch"], rtol=3e-6, atol=3e-7)


def test_real_atmosphere_gradients_match_frozen_outputs(real_library):
    name, library, point, reference, _, _ = real_library
    weights = jnp.array([[0.2, -0.7, 1.1, 0.4], [-0.3, 0.6, 0.9, -0.5]])

    def objective(lib, p):
        return jnp.sum(sample_pixels(evaluate(lib, p)) * weights)

    actual = jax.jit(jax.grad(objective, argnums=1))(library, jnp.asarray(point))
    scales = np.array([library.grid.axis(name)[1] - library.grid.axis(name)[0]
                       for name in library.grid.axis_names])
    np.testing.assert_allclose(actual * scales, reference[f"grid_{name}_gradient"] * scales,
                               rtol=2e-5, atol=3e-7)


def test_real_endpoints_and_domain_use_the_new_grid_contract(real_library):
    _, library, point, _, _, upper = real_library
    upper_point = jnp.array([library.grid.axis(name)[-1] for name in library.grid.axis_names])
    np.testing.assert_allclose(evaluate(library, upper_point), upper, rtol=3e-6, atol=3e-7)
    outside = jnp.asarray(point).at[0].set(library.grid.axis("teff")[0] - 1)
    assert np.all(np.isneginf(evaluate(library, outside)))
