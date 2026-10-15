"""MIST v1.2 EEP parsing and validated UBVRIplus bolometric corrections.

Numerics retained from the independent benchmark implementation. BCs use
Steffen interpolation in logT, linear interpolation in logg, then Steffen in
metallicity, with linear temperature-edge cells and the MESA three-point
monotone recipe at metallicity edges. This version supports Av=0, Rv=3.1.
Coordinates outside the BC table are unsupported (no clipping/extrapolation).

Recipe source: https://github.com/aarondotter/iso/blob/
ed6ef0588664de62df39564fb02e3e0e32d5b2eb/src/BC_tables.f90
Composition: the matching iso_eep_color.f90 uses Zsun=0.0134, Xsun=0.7381.
Keep these release-specific choices here when adding other MIST versions.
"""

import io
import tarfile

import numpy as np

ZSUN = .0134
XSUN = .7381
MBOL_SUN = 4.74

BANDS = (
    "Bessell_U", "Bessell_B", "Bessell_V", "Bessell_R", "Bessell_I",
    "2MASS_J", "2MASS_H", "2MASS_Ks",
    "Gaia_G_DR2Rev", "Gaia_BP_DR2Rev", "Gaia_RP_DR2Rev",
    "Gaia_G_EDR3", "Gaia_BP_EDR3", "Gaia_RP_EDR3",
)


def read_track(stream):
    """Read one native EEP member; reconstruct EEP labels from its header."""
    return _read_track(stream, version="1.2", composition_columns=("log_surf_z", "surface_h1"))


def _read_track(stream, *, version, composition_columns):
    """Shared EEP container validation; each release supplies its chemistry columns."""
    text = stream.read().decode("ascii")
    header = []
    for line in text.splitlines():
        header.append(line.strip())
        words = line.lstrip("#").split()
        if words[:1] == ["star_age"]:
            columns = words
            break
    else:
        raise ValueError("Missing track column header")
    if not any("MIST version number" in line and line.split("=")[-1].strip() == version
               for line in header):
        raise ValueError(f"Track parser requires MIST v{version}")
    metadata = composition = primary = None
    try:
        for i, line in enumerate(header):
            words = line.lstrip("#").split()
            if words[:2] == ["Yinit", "Zinit"]:
                composition = dict(zip(words, map(float, header[i+1].lstrip("#").split())))
            elif words[:2] == ["initial_mass", "N_pts"]:
                metadata = dict(zip(words, header[i+1].lstrip("#").split()))
            elif words[:1] == ["EEPs:"]:
                primary = np.array(list(map(int, words[1:])))
        if metadata is None or composition is None or primary is None:
            raise ValueError("Missing track composition, dimensions or EEP metadata")
        n_points, n_columns = int(metadata["N_pts"]), int(metadata["N_col"])
        mass = float(metadata["initial_mass"])
        if (len(primary) != int(metadata["N_EEP"]) or len(primary) < 2
                or primary[0] < 1 or not np.all(np.diff(primary) > 0)):
            raise ValueError("Primary EEP labels must be positive and increasing")
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError("Initial mass must be finite and positive")
        if not all(np.isfinite(composition[k]) for k in ("[Fe/H]", "[a/Fe]", "v/vcrit")):
            raise ValueError("Nonfinite track composition")
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Missing or malformed track metadata") from exc
    values = np.loadtxt(io.StringIO(text), ndmin=2)
    if (values.shape != (n_points, n_columns) or len(columns) != n_columns
            or len(set(columns)) != n_columns):
        raise ValueError("Track dimensions disagree with the column header")
    eep = np.arange(primary[0], primary[-1]+1)
    if len(eep) != n_points:
        raise ValueError("Track rows do not match the declared EEP range")
    data = {name: values[:, i] for i, name in enumerate(columns)}
    required = ("star_age", "star_mass", "log_Teff", "log_g", "log_L", "log_R",
                *composition_columns)
    for name in required:
        if name not in data:
            raise ValueError(f"Missing required track column: {name}")
    # Numerical validity is checked on the requested EEP slice by the builder.
    return data, eep, {"initial_mass": mass, "composition": composition,
                       "n_points": n_points, "primary_eeps": primary.tolist()}


def read_bc_archive(path):
    """Read the Av=0, Rv=3.1 slices without extracting an archive to disk."""
    planes = []
    with tarfile.open(path, "r|xz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".UBVRIplus"):
                continue
            with archive.extractfile(member) as stream:
                for raw in stream:
                    line = raw.decode("ascii")
                    if line.startswith("#") and line[1:].split()[:5] == [
                        "Teff", "logg", "[Fe/H]", "Av", "Rv"
                    ]:
                        names = line[1:].split()
                        break
                else:
                    raise ValueError(f"BC column header not found: {member.name}")
                values = np.loadtxt(stream, ndmin=2)
            if values.shape[1] != len(names) or len(set(names)) != len(names):
                raise ValueError(f"BC column count mismatch: {member.name}")
            if not set(BANDS).issubset(names):
                raise ValueError(f"Missing UBVRIplus bands in {member.name}")
            values = values[(values[:, 3] == 0) & (values[:, 4] == 3.1)]
            if not len(values) or not np.all(np.isfinite(values[:, :5])) or np.any(values[:, 0] <= 0):
                raise ValueError(f"Invalid Av=0, Rv=3.1 BC axes: {member.name}")
            temperatures = np.unique(values[:, 0])
            gravities = np.unique(values[:, 1])
            metallicities = np.unique(values[:, 2])
            if len(metallicities) != 1:
                raise ValueError("Expected one metallicity per BC file")
            order = np.lexsort((values[:, 1], values[:, 0]))
            ordered = values[order]
            expected = np.stack(np.meshgrid(temperatures, gravities, indexing="ij"), -1)
            if not np.array_equal(ordered[:, :2], expected.reshape(-1, 2)):
                raise ValueError("BC temperature/gravity grid has missing or duplicate nodes")
            fields = ordered[:, [names.index(band) for band in BANDS]]
            if not np.all(np.isfinite(fields)):
                raise ValueError("Nonfinite BC table values")
            planes.append((metallicities[0], temperatures, gravities,
                           fields.reshape(len(temperatures), len(gravities), -1)))
    if len(planes) < 4:
        raise ValueError("At least four BC metallicities are needed")
    planes.sort(key=lambda item: item[0])
    z = np.array([p[0] for p in planes])
    t, g = planes[0][1:3]
    if len(t) < 4 or len(g) < 2:
        raise ValueError("BC interpolation needs at least four temperatures and two gravities")
    if not np.all(np.diff(z) > 0):
        raise ValueError("Duplicate BC metallicities")
    if any(not np.array_equal(t, p[1]) or not np.array_equal(g, p[2]) for p in planes):
        raise ValueError("BC atmosphere axes differ between metallicities")
    return (z, np.log10(t), g), np.stack([p[3] for p in planes])

def steffen_middle(x, y, query):
    """Four-point monotone Hermite interpolation on the middle interval.

    x: (N, 4); y: (N, 4, bands); query: (N,). Slopes use Steffen's
    limiter, as in MESA interp_4pt_pm_sg / interp_4_to_1_sg. This is not
    an unconstrained four-node polynomial or scipy's PCHIP prescription.
    """
    h = np.diff(x, axis=1)[..., None]
    slopes = np.diff(y, axis=1) / h
    parabola = (slopes[:, :-1] * h[:, 1:] + slopes[:, 1:] * h[:, :-1]) / (
        h[:, :-1] + h[:, 1:]
    )
    derivatives = (np.sign(slopes[:, :-1]) + np.sign(slopes[:, 1:])) * np.minimum(
        np.minimum(np.abs(slopes[:, :-1]), np.abs(slopes[:, 1:])),
        0.5 * np.abs(parabola),
    )
    dx = (query - x[:, 1])[:, None]
    quadratic = (3 * slopes[:, 1] - 2 * derivatives[:, 0] - derivatives[:, 1]) / h[:, 1]
    cubic = (derivatives[:, 0] + derivatives[:, 1] - 2 * slopes[:, 1]) / h[:, 1] ** 2
    return y[:, 1] + dx * (derivatives[:, 0] + dx * (quadratic + dx * cubic))

def source_recipe_bc(axes, fields, queries):
    """Interior Av=0 recipe from the public iso/MESA source, using float64.

    Interpolate logT first with Steffen, logg linearly, then metallicity
    with Steffen. This helper covers interior stencils only.
    """
    z, t, g = axes
    indices = [np.searchsorted(a, q, side="right") - 1 for a, q in zip(axes, queries.T)]
    iz, it, ig = indices
    valid = (np.all(np.isfinite(queries), axis=1)
             & (iz >= 1) & (iz < len(z) - 2)
             & (it >= 1) & (it < len(t) - 2)
             & (ig >= 0) & (ig < len(g) - 1))
    if not np.all(valid):
        raise ValueError(f"{np.count_nonzero(~valid)} queries outside supported interior stencils")
    offsets = np.arange(-1, 3)
    tidx = it[:, None] + offsets

    def at_metallicity_and_gravity(zi, gi):
        values = fields[zi[:, None], tidx, gi[:, None]]
        return steffen_middle(t[tidx], values, queries[:, 1])

    weight = ((queries[:, 2] - g[ig]) / (g[ig + 1] - g[ig]))[:, None]
    planes = []
    for dz in offsets:
        lo = at_metallicity_and_gravity(iz + dz, ig)
        hi = at_metallicity_and_gravity(iz + dz, ig + 1)
        planes.append((1 - weight) * lo + weight * hi)
    return steffen_middle(z[iz[:, None] + offsets], np.stack(planes, axis=1), queries[:, 0])

def bc_coverage(axes, queries):
    """Inclusive BC table support and the four-point interior subset."""
    in_table = np.all(np.isfinite(queries), axis=1)
    for axis, coordinate in zip(axes, queries.T):
        in_table &= (coordinate >= axis[0]) & (coordinate <= axis[-1])
    iz, it, ig = [np.searchsorted(a, q, side="right") - 1 for a, q in zip(axes, queries.T)]
    interior = (in_table & (iz >= 1) & (iz < len(axes[0])-2)
                & (it >= 1) & (it < len(axes[1])-2) & (ig < len(axes[2])-1))
    return in_table, in_table.copy(), interior


def steffen_three(x, y, query):
    """MESA interp_3_to_1_sg / mk_pmquad_sg, including endpoint limiting.

    Despite the Fortran routine's name this is piecewise Hermite, not an
    unconstrained quadratic through three nodes. Shapes match steffen_middle
    with three nodes instead of four.
    """
    h = np.diff(x, axis=1)[..., None]
    slopes = np.diff(y, axis=1)/h
    parabola = (slopes[:, 0]*h[:, 1]+slopes[:, 1]*h[:, 0])/(h[:, 0]+h[:, 1])
    middle = (np.sign(slopes[:, 0])+np.sign(slopes[:, 1]))*np.minimum(
        np.minimum(abs(slopes[:, 0]), abs(slopes[:, 1])), .5*abs(parabola))
    ends = []
    for i, j in ((0, 1), (1, 0)):
        p = slopes[:, i]*(1+h[:, i]/(h[:, i]+h[:, j])) - slopes[:, j]*h[:, i]/(h[:, i]+h[:, j])
        ends.append(np.where(p*slopes[:, i] <= 0, 0.,
                             np.sign(p)*np.minimum(abs(p), 2*abs(slopes[:, i]))))
    derivatives = np.stack((ends[0], middle, ends[1]), axis=1)
    interval = (query >= x[:, 1]).astype(int)
    rows = np.arange(len(query))
    dx = (query-x[rows, interval])[:, None]
    d0, d1 = derivatives[rows, interval], derivatives[rows, interval+1]
    slope, width = slopes[rows, interval], h[rows, interval]
    return y[rows, interval]+dx*(d0+dx*((3*slope-2*d0-d1)/width
                                      +dx*(d0+d1-2*slope)/width**2))


def track_bc(axes, fields, queries):
    """Public v1.2 recipe on the entire inclusive BC table.

    Temperature edge intervals are linear; metallicity edge intervals use
    MESA's limited three-point recipe. Interior intervals retain four-point
    Steffen interpolation. Gravity is linear, including its last endpoint.
    The public generator clips out-of-table atmospheres; here those are
    rejected so the builder can mark their photometry as missing explicitly.
    """
    _, supported, interior = bc_coverage(axes, queries)
    if not np.all(supported):
        raise ValueError("Unsupported BC coordinates")
    result = np.empty((len(queries), fields.shape[-1]))
    # Preserve the validated interior arithmetic exactly.
    result[interior] = source_recipe_bc(axes, fields, queries[interior])
    q = queries[~interior]
    if not len(q):
        return result
    z, t, g = axes
    iz, it, ig = [np.minimum(np.searchsorted(a, c, side="right")-1, len(a)-2)
                  for a, c in zip(axes, q.T)]
    gw = ((q[:, 2]-g[ig])/(g[ig+1]-g[ig]))[:, None]
    tw = ((q[:, 1]-t[it])/(t[it+1]-t[it]))[:, None]
    interior_t = (it >= 1) & (it < len(t)-2)

    def plane(zi):
        def at_gravity(gi):
            value = (1-tw)*fields[zi, it, gi]+tw*fields[zi, it+1, gi]
            ti = it[interior_t, None]+np.arange(-1, 3)
            value[interior_t] = steffen_middle(
                t[ti], fields[zi[interior_t, None], ti, gi[interior_t, None]], q[interior_t, 1])
            return value
        return (1-gw)*at_gravity(ig)+gw*at_gravity(ig+1)

    edge_z = (iz == 0) | (iz == len(z)-2)
    # Clipping only selects a fully in-table stencil, never query coordinates.
    start = np.clip(iz-1, 0, len(z)-4)
    planes = np.stack([plane(start+k) for k in range(4)], axis=1)
    values = np.empty((len(q), fields.shape[-1]))
    central = ~edge_z
    values[central] = steffen_middle(z[start[central, None]+np.arange(4)],
                                    planes[central], q[central, 0])
    for mask, offset in ((iz == 0, 0), (iz == len(z)-2, 1)):
        nodes = start[mask, None]+np.arange(offset, offset+3)
        values[mask] = steffen_three(z[nodes], planes[mask, offset:offset+3], q[mask, 0])
    result[~interior] = values
    return result
