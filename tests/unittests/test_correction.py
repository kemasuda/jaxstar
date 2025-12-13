import numpy as np
import pandas as pd
import astropy.units as units
from astropy.coordinates import Angle

from jaxstar.utils import correction


def test_ensure_quantity_numeric_input():
    values = [1.0, 2.0, 3.5]
    quantity = correction._ensure_quantity(values, units.pc, units)
    assert quantity.unit == units.pc
    assert np.allclose(quantity.value, values)


def test_ensure_quantity_series_of_quantities():
    series = pd.Series(units.Quantity([10, 20, 30], unit=units.deg))
    quantity = correction._ensure_quantity(series, units.deg, units)
    assert quantity.unit == units.deg
    assert np.allclose(quantity.value, [10, 20, 30])


def test_ensure_quantity_object_series_with_angle():
    angles = pd.Series([Angle(1, unit=units.deg), np.nan, Angle(2, unit=units.deg)], dtype="object")
    quantity = correction._ensure_quantity(angles, units.deg, units)
    assert quantity.unit == units.deg
    assert np.isnan(quantity.value[1])
    assert np.allclose(
        quantity.value[[0, 2]],
        [1.0, 2.0],
    )
