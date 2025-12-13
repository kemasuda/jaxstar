import numpy as np
import pandas as pd
import astropy.units as units
from astropy.coordinates import Angle
import jaxstar.utils.correction as correction
import sys

class DummyZPT:
    def __init__(self, offset):
        self.offset = offset

    def load_tables(self):
        return None

    def zpt_wrapper(self, row):
        return self.offset


class FakeSkyCoord:
    def __init__(self, l, b, distance, frame):
        self.l = l
        self.b = b
        self.distance = distance
        self.frame = frame


class FakeBayestarQuery:
    def __init__(self, version):
        self.version = version
        self.calls = []

    def __call__(self, coords, mode="median"):
        self.calls.append((coords, mode))
        # Return small reddening matching the shape of distance array
        dist = getattr(coords.distance, "value", coords.distance)
        return units.Quantity(np.ones_like(dist) * 0.1)


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


def test_correct_gedr3_parallax_zeropoint(monkeypatch):
    offset = 0.05
    dummy = DummyZPT(offset)
    mod = type(sys)("zero_point")
    mod.zpt = dummy
    monkeypatch.setitem(sys.modules, "zero_point", mod)

    df = pd.DataFrame({"parallax": [1.0, 2.0]})
    corrected = correction.correct_gedr3_parallax_zeropoint(df.copy())
    assert np.allclose(corrected.values, df["parallax"].values - offset)


def test_correct_kmag_handles_negative_parallax_and_units(monkeypatch):
    fake_query = FakeBayestarQuery(version="bayestar2019")
    monkeypatch.setattr("dustmaps.bayestar.BayestarQuery", lambda version: fake_query)
    monkeypatch.setattr("astropy.coordinates.SkyCoord", FakeSkyCoord)

    df = pd.DataFrame(
        {
            "kmag": [10.0, 11.0],
            "kmag_err": [np.nan, 0.2],
            "parallax": [-0.5, 0.4],  # one negative, one positive
            "l": units.Quantity([1.0, 2.0], unit=units.deg),
            "b": units.Quantity([3.0, 4.0], unit=units.deg),
        }
    )

    corrected = correction.correct_kmag(df.copy())
    # Negative parallax replaced with 8e3 pc, positive with 1/parallax*1e3
    assert corrected.loc[0, "distpc"] == 8000
    assert np.isclose(corrected.loc[1, "distpc"], 1e3 / 0.4)
    # kmag_err NaN filled with median of finite values
    assert np.isfinite(corrected.loc[0, "kmag_err"])
    # ak computed with reddening=0.1 and Rvect[-1] for version 2019 (0.3026)
    expected_ak = 0.1 * correction.Rvect_b19[-1]
    assert np.allclose(corrected["ak"], expected_ak)
