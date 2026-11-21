"""Unit tests for station validation (synthetic data)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from rasterio.transform import from_origin

from hybridclim import validation as val


class Metrics(unittest.TestCase):
    def test_known_errors(self):
        df = pd.DataFrame(dict(name=list("abcd"), variable="P", product="Hybrid",
                               observed=[100, 200, 300, 400], predicted=[110, 190, 330, 400]))
        m = val.metrics(df).iloc[0]
        self.assertAlmostEqual(m.ME, 7.5)
        self.assertAlmostEqual(m.MAE, 12.5)
        self.assertAlmostEqual(m.RMSE, np.sqrt((100 + 100 + 900) / 4))
        self.assertAlmostEqual(m.PBIAS_pct, 100 * 30 / 1000)

    def test_temperature_has_no_pbias(self):
        df = pd.DataFrame(dict(name=list("abc"), variable="T", product="ERA5 raw",
                               observed=[15.0, 16, 17], predicted=[14.0, 15, 16]))
        m = val.metrics(df).iloc[0]
        self.assertTrue(np.isnan(m.PBIAS_pct))
        self.assertAlmostEqual(m.ME, -1.0)


class ReadStations(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        pd.DataFrame(dict(name=["A", "B"], lat=[38.1, 38.2], lon=[27.3, 27.4],
                          T_annual=[17.5, 16.9], P24_mean_annual_max=[np.nan, 60.0])
                     ).to_csv(self.d / "st.csv", index=False)
        yrs = np.arange(1990, 2025)
        pd.DataFrame(dict(name=["A"] * len(yrs), year=yrs, P_max=np.full(len(yrs), 50.0))
                     ).to_csv(self.d / "ams.csv", index=False)

    def test_ams_fills_p24_and_converts(self):
        st, wide, conv = val.read_stations(self.d / "st.csv", self.d / "ams.csv", (1995, 2024),
                                           "calendar-day", f24=1.13)
        self.assertAlmostEqual(conv, 1.13)
        self.assertAlmostEqual(st.loc[st["name"] == "A", "P24_mean_annual_max"].item(), 50 * 1.13)
        self.assertAlmostEqual(st.loc[st["name"] == "B", "P24_mean_annual_max"].item(), 60 * 1.13)
        self.assertEqual(list(wide.columns), ["A"])

    def test_24h_kind_not_converted(self):
        st, _, conv = val.read_stations(self.d / "st.csv", None, None, "24h", f24=1.13)
        self.assertEqual(conv, 1.0)
        self.assertAlmostEqual(st.loc[st["name"] == "B", "P24_mean_annual_max"].item(), 60.0)

    def test_unknown_ams_station(self):
        pd.DataFrame(dict(name=["Z"], year=[2000], P_max=[1.0])).to_csv(self.d / "bad.csv", index=False)
        with self.assertRaises(SystemExit):
            val.read_stations(self.d / "st.csv", self.d / "bad.csv", (1995, 2024))


class SampleProducts(unittest.TestCase):
    def test_sampling(self):
        to_utm = Transformer.from_crs(4326, 32635, always_xy=True)
        x0, y0 = to_utm.transform(27.3, 38.1)
        tr = from_origin(x0 - 300, y0 + 300, 30, 30)            # 20 x 20 px, station in the middle
        hyb = {"T": np.full((20, 20), 17.0), "P": np.full((20, 20), 700.0),
               "P24": np.full((20, 20), 65.0)}
        ch_tr = from_origin(27.0, 38.5, 1 / 120, 1 / 120)
        chelsa = {"T": (np.full((100, 100), 16.0), ch_tr), "P": (np.full((100, 100), 650.0), ch_tr)}
        tp = pd.DataFrame(dict(lat=[38.1, 39.0], lon=[27.3, 28.0], T_target=[16.5, 10.0]))
        pp = pd.DataFrame(dict(lat=[38.0, 39.0], lon=[27.25, 28.0], P_target=[600.0, 1.0],
                               Pmax_day_mean=[40.0, 1.0]))
        st = pd.DataFrame(dict(name=["A"], lat=[38.1], lon=[27.3], T_annual=[17.2],
                               P_annual=[720.0], P24_mean_annual_max=[70.0]))
        df = val.sample_products(st, tp, pp, chelsa, hyb, tr, to_utm, f24=1.1)
        got = df.set_index(["variable", "product"]).predicted
        self.assertEqual(got["T", "ERA5 raw"], 16.5)
        self.assertEqual(got["T", "CHELSA raw (1981-2010)"], 16.0)
        self.assertEqual(got["T", "Hybrid"], 17.0)
        self.assertAlmostEqual(got["P24", "ERA5 raw"], 44.0)
        self.assertNotIn(("P24", "CHELSA raw (1981-2010)"), got.index)
        self.assertEqual(len(df), 3 + 3 + 2)


if __name__ == "__main__":
    unittest.main()
