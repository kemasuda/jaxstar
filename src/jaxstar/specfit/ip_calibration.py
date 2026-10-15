"""Host-side loading of fixed, per-order two-Gaussian instrumental profiles."""

from collections.abc import Mapping
import csv
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from types import MappingProxyType

import numpy as np


_IP_PARAMETERS = ("mu1_kms", "mu2_kms", "sigma1_kms", "sigma2_kms", "fraction2")


def _fixed_ip_values(values, count):
    """Copy, expand and validate concrete IP parameters outside JIT."""
    if not isinstance(values, Mapping) or set(values) != set(_IP_PARAMETERS):
        raise ValueError(f"fixed_ip must supply exactly {', '.join(_IP_PARAMETERS)}")
    result = {}
    for name in _IP_PARAMETERS:
        value = np.asarray(values[name])
        if value.dtype.kind not in "fiu" or value.shape not in ((), (count,)):
            raise ValueError(f"fixed_ip {name} must be numeric, scalar or shape ({count},)")
        value = np.broadcast_to(value, (count,)).astype(np.float64, copy=True)
        if not np.all(np.isfinite(value)):
            raise ValueError(f"fixed_ip {name} must be finite")
        if name.startswith("sigma") and np.any(value <= 0):
            raise ValueError(f"fixed_ip {name} must be positive")
        if name == "fraction2" and np.any((value < 0) | (value > 1)):
            raise ValueError("fixed_ip fraction2 must be in [0, 1]")
        value.setflags(write=False)
        result[name] = value
    return result


@dataclass(frozen=True, eq=False)
class IPCalibration:
    """Selected calibration medians, in the explicitly requested region order.

    ``median`` maps the five IP parameter names to read-only NumPy arrays.
    ``orders`` retains the requested order identifiers, including repetitions;
    ``source`` is the resolved parameters.csv path. Construct via
    :func:`read_ip_calibration`. Quantiles and derived CSV rows are not used.
    """

    orders: tuple
    median: Mapping
    source: Path


def read_ip_calibration(dir_name, *, orders):
    """Read ``dir_name/parameters.csv`` and align medians with spectral regions.

    Required CSV columns are ``order, parameter, median``. For each requested
    order, require exactly one row for each of mu1_kms, mu2_kms, sigma1_kms,
    sigma2_kms and fraction2. Shifts and sigmas are in km/s; fraction2 is the
    integrated fraction of component 2. Labels are matched as strings (integer
    104 matches CSV label '104'), independent of CSV row order. Repeated orders
    share the same calibration. Missing or duplicate selected parameters fail.

    Extra columns (including p16/p84) and other parameter rows (including R,
    delta_kms, centroid_kms and intrinsic_fwhm_kms) are ignored. No priors are
    inferred. The caller supplies orders in observation/library region order;
    order identifiers are not inferred from wavelength bounds or region labels.
    """
    if isinstance(orders, (str, bytes)):
        raise ValueError("orders must be a nonempty sequence of integer or string identifiers")
    orders = tuple(orders)
    if not orders or any(isinstance(order, (bool, np.bool_))
                         or not isinstance(order, (Integral, str))
                         or not str(order).strip() for order in orders):
        raise ValueError("orders must be a nonempty sequence of integer or string identifiers")
    labels = tuple(str(order).strip() for order in orders)
    selected = set(labels)
    source = (Path(dir_name).expanduser() / "parameters.csv").resolve()
    rows = {}
    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{source}: missing CSV header")
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        if not {"order", "parameter", "median"}.issubset(reader.fieldnames):
            raise ValueError(f"{source}: required CSV columns are order, parameter, median")
        for row in reader:
            order = (row["order"] or "").strip()
            name = (row["parameter"] or "").strip()
            if order not in selected or name not in _IP_PARAMETERS:
                continue
            key = (order, name)
            if key in rows:
                raise ValueError(f"{source}: duplicate order {order!r}, parameter {name!r}")
            try:
                rows[key] = float(row["median"])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{source}: invalid median for order {order!r}, parameter {name!r}") from exc
    missing = [(order, name) for order in dict.fromkeys(labels)
               for name in _IP_PARAMETERS if (order, name) not in rows]
    if missing:
        raise ValueError(f"{source}: missing IP parameters for order(s): {missing}")
    median = _fixed_ip_values({name: [rows[order, name] for order in labels]
                               for name in _IP_PARAMETERS}, len(orders))
    return IPCalibration(orders, MappingProxyType(median), source)
