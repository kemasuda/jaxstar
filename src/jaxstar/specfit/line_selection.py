"""Data-only screening and construction of native-pixel line SpecFits."""
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .sampling import _C_KMS


@dataclass(frozen=True)
class LineSelectionCriteria:
    """Velocities in km/s; prominence/depth/wing levels in normalized flux.

    No synthetic spectrum is evaluated. Input flux must already be normalized;
    no fitted continuum or local wing renormalization is applied. Masked pixels
    are excluded. Gaps larger than max_gap_pixels native pixel spacings split
    the profile; NaN rectangular padding does not create artificial breaks.
    """
    nearest_line_min_kms: float | None = 20.
    center_window_kms: float = 3.
    center_prominence_min: float | None = .1
    center_depth_max: float | None = None
    check_secondary: bool = True
    secondary_window_kms: float = 10.
    data_secondary_prominence: float = .05
    data_core_halfwidth_kms: float = 5.
    data_smoothing_sigma_pixels: float = 1.2
    max_gap_pixels: float = 2.5
    check_wings: bool = True
    wing_inner_kms: float = 15.
    wing_outer_kms: float = 20.
    wing_min_flux: float = .9

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key.startswith('check_'):
                if not isinstance(value, (bool, np.bool_)):
                    raise TypeError(f'{key} must be Boolean')
            elif value is None:
                if key not in ('nearest_line_min_kms', 'center_prominence_min', 'center_depth_max'):
                    raise ValueError(f'{key} cannot be None')
            elif not np.isscalar(value) or not np.isfinite(value) or value < 0:
                raise ValueError(f'{key} must be finite and nonnegative')
        if self.max_gap_pixels <= 0:
            raise ValueError('max_gap_pixels must be positive')
        if not 0 < self.center_window_kms < self.secondary_window_kms:
            raise ValueError('Require 0 < center_window_kms < secondary_window_kms')
        if not 0 <= self.data_core_halfwidth_kms < self.secondary_window_kms:
            raise ValueError('Require 0 <= data_core_halfwidth_kms < secondary_window_kms')
        if not 0 < self.wing_inner_kms < self.wing_outer_kms:
            raise ValueError('Require 0 < wing_inner_kms < wing_outer_kms')


def _doppler(rv):
    return np.sqrt((1 + rv / _C_KMS) / (1 - rv / _C_KMS))


def select_lines(observation, line_list, *, rv, criteria=None,
                 wavelength_column='wavelength', line_id_column='line_id',
                 fit_half_width_kms=10.):
    """Return all candidates, observed-frame windows and explicit rejection reasons.

    line_list is a table (metadata columns preserved) or a 1D wavelength array.
    Its wavelengths are stellar-rest wavelengths in the observation's medium
    and unit. RV is scalar or one value per input region, positive for redshift.
    No implicit air/vacuum or barycentric conversion is performed. Overlapping
    orders use the one with the largest edge margin. The full supplied list is
    used for neighbor distances. ``selected`` is the conjunction of the rules;
    ``wavelength_min/max`` use the observation frame. Returns a DataFrame even
    when no candidates pass. Selection never changes observations or masks.
    """
    c = criteria if isinstance(criteria, LineSelectionCriteria) else LineSelectionCriteria(**(criteria or {}))
    if not np.isscalar(fit_half_width_kms) or not 0 < fit_half_width_kms < _C_KMS:
        raise ValueError('fit_half_width_kms must be positive and below c')
    if isinstance(line_list, pd.DataFrame) or isinstance(line_list, dict):
        lines = pd.DataFrame(line_list).copy().reset_index(drop=True)
    else:
        array = np.asarray(line_list)
        lines = pd.DataFrame({wavelength_column: array}) if array.ndim == 1 and array.dtype.names is None else pd.DataFrame(line_list)
    centers = lines[wavelength_column].to_numpy(dtype=float)
    if not len(centers) or not np.all(np.isfinite(centers) & (centers > 0)):
        raise ValueError('line wavelengths must be nonempty, finite and positive')
    if line_id_column not in lines:
        lines[line_id_column] = [f'line_{i}' for i in range(len(lines))]
    if lines[line_id_column].isna().any() or lines[line_id_column].duplicated().any():
        raise ValueError('line IDs must be non-null and unique')
    velocities = np.asarray(rv, dtype=float)
    if velocities.shape not in ((), (observation.n_regions,)) or not np.all(np.isfinite(velocities) & (np.abs(velocities) < _C_KMS)):
        raise ValueError('rv must be finite scalar or one value per input region, with abs(rv) < c')
    velocities = np.broadcast_to(velocities, (observation.n_regions,))
    factors = _doppler(velocities)
    wave = np.atleast_2d(np.asarray(observation.wavelength))
    flux = np.atleast_2d(np.asarray(observation.flux))
    valid = np.atleast_2d(np.asarray(observation.valid))
    labels = observation.region or tuple(range(observation.n_regions))
    rest = wave / factors[:, None]
    scan = max(30., c.secondary_window_kms + 10., c.wing_outer_kms + 5.)
    output = []
    for i, line in lines.iterrows():
        center = centers[i]
        distances = np.abs(_C_KMS * np.tanh(np.log(centers / center)))
        distances[i] = np.inf
        nearest = distances.min()
        rows = np.flatnonzero((rest[:, 0] <= center) & (rest[:, -1] >= center))
        record = line.to_dict()
        record.update(covered=bool(len(rows)), selected=False, nearest_line_kms=float(nearest))
        if not len(rows):
            record['rejected_by'] = 'outside_observed_orders'
            output.append(record)
            continue
        row = max(rows, key=lambda r: min(center-rest[r, 0], rest[r, -1]-center))
        v = _C_KMS * np.tanh(np.log(rest[row] / center))
        use = valid[row] & (np.abs(v) <= scan)
        pixels = np.flatnonzero(use)
        native = wave[row, np.isfinite(flux[row])]
        spacing = np.median(np.diff(native)) if len(native) > 1 else np.inf
        runs = np.split(pixels, np.flatnonzero(np.diff(wave[row, pixels]) > c.max_gap_pixels*spacing) + 1)
        peaks = []
        for run in runs:
            if len(run) < 3:
                continue
            y = flux[row, run]
            smooth = gaussian_filter1d(y, c.data_smoothing_sigma_pixels) if c.data_smoothing_sigma_pixels else y
            minima, props = find_peaks(-smooth, prominence=0.)
            peaks.extend((int(run[p]), float(smooth[p]), float(prom)) for p, prom in zip(minima, props['prominences']))
        near = [p for p in peaks if abs(v[p[0]]) <= c.center_window_kms]
        main = min(near, key=lambda p: p[1]) if near else None
        prominence = np.nan if main is None else main[2]
        depth = np.nan if main is None else 1-main[1]
        center_ok = main is not None and (c.center_prominence_min is None or prominence > c.center_prominence_min)
        depth_ok = c.center_depth_max is None or (main is not None and depth <= c.center_depth_max)
        secondary = [p for p in peaks if c.data_core_halfwidth_kms < abs(v[p[0]]) <= c.secondary_window_kms and p[2] >= c.data_secondary_prominence]
        wings = [flux[row, use & (sign*v >= c.wing_inner_kms) & (sign*v <= c.wing_outer_kms)] for sign in (-1, 1)]
        levels = [float(np.median(band)) if len(band) else np.nan for band in wings]
        observed_center = center * factors[row]
        lo, hi = observed_center * _doppler(np.array([-fit_half_width_kms, fit_half_width_kms]))
        coverage_ok = wave[row, 0] <= lo < hi <= wave[row, -1] and np.any(valid[row] & (wave[row] >= lo) & (wave[row] <= hi))
        checks = dict(spacing=c.nearest_line_min_kms is None or nearest >= c.nearest_line_min_kms,
                      central_minimum_prominence=center_ok, depth=depth_ok,
                      secondary=not c.check_secondary or not secondary,
                      wings=not c.check_wings or (np.isfinite(levels).all() and min(levels) >= c.wing_min_flux),
                      fit_window_coverage=coverage_ok)
        record.update(source_index=int(row), source_region=labels[row], initial_rv=float(velocities[row]),
                      wavelength_min=float(lo), wavelength_max=float(hi), center_prominence=float(prominence),
                      center_depth=float(depth), data_secondary_count=len(secondary),
                      data_blue_wing=levels[0], data_red_wing=levels[1],
                      selected=bool(all(checks.values())), rejected_by='; '.join(k for k, good in checks.items() if not good))
        record.update({f'{key}_ok': bool(value) for key, value in checks.items()})
        output.append(record)
    return pd.DataFrame(output)


class _SelectedIP:
    """Adapt an order-row callable without changing its source-row convention."""
    def __init__(self, function, source_wave, indices):
        self.function, self.source_wave, self.indices = function, source_wave, np.asarray(indices)

    def __call__(self, wavelength):
        from .ip_calibration import _fixed_ip_values
        wave = np.asarray(wavelength)
        # Each call keeps every original row in its original position. Repeated
        # source orders are handled in successive batches; unused rows remain
        # inside their own calibration domain, never in a different order.
        base = np.exp(np.linspace(np.log(self.source_wave[:, 0]), np.log(self.source_wave[:, -1]), wave.shape[1])).T
        pending = list(range(len(wave)))
        output = None
        while pending:
            batch, used = [], set()
            for r in pending:
                src = int(self.indices[r])
                if src not in used:
                    batch.append(r)
                    used.add(src)
            request = base.copy()
            for r in batch:
                request[self.indices[r]] = wave[r]
            request.setflags(write=False)
            values = _fixed_ip_values(self.function(request), len(request), pixels=request.shape[1])
            if output is None:
                output = {name: np.empty(wave.shape) for name in values}
            for name, values_at_wave in values.items():
                values_at_wave = np.asarray(values_at_wave)
                for r in batch:
                    source = int(self.indices[r])
                    output[name][r] = values_at_wave if values_at_wave.ndim == 0 else values_at_wave[source]
            pending = [r for r in pending if r not in batch]
        return output


def select_line_specfit(sf, line_list, *, rv=None, criteria=None,
                        wavelength_column='wavelength', line_id_column='line_id',
                        fit_half_width_kms=10., wav_margin=3., crop=True,
                        inherit_fit_mask=False, overlap="error"):
    """Return (line_sf, selection_table); line_sf is None if nothing passes.

    Keeps input masks and source-order physical settings. By default previous
    fit-derived masks are NOT reused: residuals are the evidence for the region
    mixture. Set inherit_fit_mask=True to explicitly carry them over. Model
    arrays are cropped on their native sampling, retaining every atmosphere
    node; source order, pixel mapping and RV initialization are attached to the
    returned fit. IP callables are re-evaluated with their original row mapping.
    overlap='nearest' retains all candidates/windows and assigns each shared
    source pixel to the closest observed-frame line center (log wavelength).
    Other copies are fit-masked, preserving the original input mask. The
    assignment is fixed before fitting; equal distances favor the first line.
    overlap='error' rejects reused pixels. region_overlap_mask records assignment.
    """
    if overlap not in ('error', 'nearest'):
        raise ValueError("overlap must be 'error' or 'nearest'")
    from ._crop import _crop_payload
    from ._selection import _wavelength_request
    from ..grid import Field, RectilinearGrid
    from .model import SpecModel
    from .regions import extract_regions
    from .specfit import SpecFit

    if rv is None:
        if sf.ccf is None:
            raise ValueError('supply rv or run check_ccf before selecting lines')
        rv = sf.ccf.get('region_rv', sf.ccf['rv'])
    table = select_lines(sf.observation, line_list, rv=rv, criteria=criteria,
                         wavelength_column=wavelength_column, line_id_column=line_id_column,
                         fit_half_width_kms=fit_half_width_kms)
    selected = table.loc[table.selected]
    if selected.empty:
        return None, table
    sources = [sf.regions[int(i)] for i in selected.source_index]
    obs, mapping = extract_regions(sf.observation,
        wavelength_ranges=selected[['wavelength_min', 'wavelength_max']].to_numpy(),
        regions=selected[line_id_column].tolist(), source_regions=sources)
    overlap_mask = np.zeros(obs.shape, dtype=bool)
    centers = np.sqrt(selected.wavelength_min.to_numpy()*selected.wavelength_max.to_numpy())
    owners = {}
    for r, (source, pixels) in enumerate(zip(mapping.source_indices, mapping.pixel_indices)):
        for j, pixel in enumerate(pixels):
            if pixel < 0:
                continue
            key = (int(source), int(pixel))
            distance = abs(np.log(obs.wavelength[r, j]/centers[r]))
            if key in owners:
                if overlap == 'error':
                    raise ValueError("selected fit windows reuse observed pixels; use overlap='nearest' to assign each pixel once")
                previous_r, previous_j, previous_distance = owners[key]
                if distance < previous_distance:
                    overlap_mask[previous_r, previous_j] = True
                    owners[key] = (r, j, distance)
                else:
                    overlap_mask[r, j] = True
            else:
                owners[key] = (r, j, distance)
    table.loc[selected.index, 'overlap_excluded_pixels'] = np.sum(overlap_mask & np.asarray(obs.valid), axis=-1)
    parent = sf.specmodel
    spectra = parent.spectra
    indices = mapping.source_indices
    wave = np.asarray(spectra.wavelength)
    flux = np.asarray(spectra.grid.field('flux').values)
    request = _wavelength_request(obs.wavelength, obs.region, wav_margin)
    if np.any(obs.wavelength[:, 0]-wav_margin < wave[indices, 0]) or np.any(obs.wavelength[:, -1]+wav_margin > wave[indices, -1]):
        raise ValueError('line model margin exceeds source grid coverage; prepare a wider parent grid')
    if crop:
        target, payload = _crop_payload(wave, flux, indices, request)
    else:
        target, payload = wave[indices], flux[..., indices, :]
    cropped_grid = RectilinearGrid(
        axes={name: spectra.grid.axis(name) for name in spectra.grid.axis_names},
        fields={'flux': Field(payload, dims=spectra.grid.axis_names, payload_dims=('region', 'pixel'))},
        boundary=spectra.grid.boundary, fill_value=spectra.grid.fill_value)
    grid = replace(spectra, grid=cropped_grid, wavelength=target, regions=obs.region)
    ip = parent.fixed_ip
    if parent._ip_callable is not None:
        ip = _SelectedIP(parent._ip_callable, wave, indices)
    elif ip is not None:
        ip = {name: np.asarray(value)[indices] for name, value in ip.items()}
    model = SpecModel(grid, fixed_ip=ip, vmax=parent.vmax, macro_model=parent.macro_model,
                      teff_interp=parent.teff_interp, broadening_operator=parent.broadening_operator)
    guard = None if sf.custom_guard is None else mapping.take(sf.custom_guard, fill_value=False)
    # Resolve a configured Mg guard before transfer; no new CCF is needed.
    guard_rv = sf.guard_rv
    if sf.mg_guard and guard_rv is None:
        _ = sf.mask_guard
        guard_rv = sf.guard_rv
    result = SpecFit(obs, model, mg_guard=sf.mg_guard, custom_guard=guard, guard_rv=guard_rv)
    fit_mask = overlap_mask.copy()
    if inherit_fit_mask:
        fit_mask |= mapping.take(sf.fit_mask, fill_value=True)
    result.set_fit_mask(fit_mask)
    overlap_mask.setflags(write=False)
    result.region_overlap_mask = overlap_mask
    result.region_mapping = mapping
    result.line_selection = table.copy()
    result.initial_rv = selected.initial_rv.to_numpy(dtype=float)
    if sf.ccf is not None:
        result.ccf = dict(rv=float(np.median(result.initial_rv)), width=sf.ccf['width'], region_rv=result.initial_rv.copy())
    return result, table
