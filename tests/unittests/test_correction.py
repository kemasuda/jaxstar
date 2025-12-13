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
    def __init__(self, version, reddening=0.1, vector=False):
        self.version = version
        self.calls = []
        self.reddening = reddening
        self.vector = vector

    def __call__(self, coords, mode="median"):
        self.calls.append((coords, mode))
        dist = getattr(coords.distance, "value", coords.distance)
        dist = np.asarray(dist)
        if self.vector:
            # Shape (N, 1) so it broadcasts with Rvect (len=8)
            reddening = np.ones(dist.shape + (1,)) * self.reddening
        else:
            reddening = np.ones(dist.shape) * self.reddening
        return units.Quantity(reddening)


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
    fake_query = FakeBayestarQuery(version="bayestar2019", vector=False)
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
    # Dustmaps called once with median mode
    assert len(fake_query.calls) == 1
    _, mode = fake_query.calls[0]
    assert mode == "median"


def test_extinction_mag_vector_uses_units_and_reddening(monkeypatch):
    fake_query = FakeBayestarQuery(version="bayestar2019", reddening=0.2, vector=True)
    monkeypatch.setattr("dustmaps.bayestar.BayestarQuery", lambda version: fake_query)
    monkeypatch.setattr("astropy.coordinates.SkyCoord", FakeSkyCoord)

    l = pd.Series([1.0, 2.0])
    b = units.Quantity([3.0, 4.0], unit=units.deg)
    distpc = [100.0, 200.0]

    ak, ak_err = correction.extinction_mag_vector(l, b, distpc, version="2019")
    # Returned shapes align with input
    assert ak.shape[1] == len(correction.Rvect_b19)
    assert ak.shape[0] == 2
    # Reddening 0.2 * Rvect_b19
    assert np.allclose(ak.value, 0.2 * correction.Rvect_b19)
    # Error is 30% of ak
    assert np.allclose(ak_err.value, 0.3 * ak.value)
