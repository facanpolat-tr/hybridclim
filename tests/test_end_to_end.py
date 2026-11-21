"""
Offline end-to-end test: runs the full `hybridclim` workflow on the bundled fixture
(tests/data/fixture, built by tests/data/make_fixtures.py) with network access disabled.

Checks that every output is written, that the results are physically consistent, and that the
key numbers match a stored snapshot (tests/data/expected_meta.json). After an intended change of
results, regenerate the snapshot with:  HYBRIDCLIM_UPDATE_SNAPSHOT=1 python -m unittest tests.test_end_to_end
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

HERE = Path(__file__).resolve().parent
FIX = HERE / "data" / "fixture"
SNAPSHOT = HERE / "data" / "expected_meta.json"
RETURN_PERIODS = [2, 5, 10, 25, 50, 100, 500]


def synthetic_stations(folder):
    """Deterministic, clearly synthetic stations inside the test area (not observations)."""
    rng = np.random.default_rng(7)
    n = 5
    st = pd.DataFrame(dict(name=[f"SYN{i}" for i in range(n)], lat=rng.uniform(38.11, 38.19, n),
                           lon=rng.uniform(27.31, 27.39, n), T_annual=rng.normal(17.5, 0.5, n),
                           P_annual=rng.normal(720, 40, n)))
    st.to_csv(folder / "stations.csv", index=False)
    rows = [dict(name=s, year=y, P_max=round(50 * (1 + 0.3 * (rng.gumbel() - 0.5772)), 1))
            for s in st.name for y in range(1985, 2025)]
    pd.DataFrame(rows).to_csv(folder / "ams.csv", index=False)


def key_numbers(meta):
    b = meta["within_boundary"]
    return {
        "T_mean": b["T"]["mean"], "P_mean": b["P"]["mean"], "P24_mean": b["P24_mean"]["mean"],
        "P24_T100_mean": b["P24_T100"]["mean"], "f24": meta["f24"]["value"],
        "dT_mean": meta["change_factors"]["dT_mean"], "CF_mean": meta["change_factors"]["CF_mean"],
        "lapse_per_100m": meta["chelsa_downscaling"]["T"]["slope_per_100m"],
        "x_T100_era5": [g for g in meta["growth_era5"] if g["T_years"] == 100][0]["x_T"],
        "x_T100_stations": [g for g in meta["growth_stations"] if g["T_years"] == 100][0]["x_T"],
        "H1_stations": meta["hosking_wallis"]["stations"]["H1"],
    }


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        shutil.copytree(FIX / "data", cls.tmp / "data")
        cls.out = cls.tmp / "out"
        cls.out.mkdir()
        for f in (FIX / "chelsa").glob("*.tif"):                # pre-seeded CHELSA cache
            shutil.copy(f, cls.out / f.name)
        synthetic_stations(cls.tmp)
        env = dict(os.environ)
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "GDAL_HTTP_PROXY"):   # any network call fails
            env[k] = "http://127.0.0.1:9"
        cmd = [sys.executable, "-m", "hybridclim.maps", "--data-dir", str(cls.tmp / "data"),
               "--dem", str(FIX / "dem_150m.tif"), "--res", "150", "--n-sim", "30",
               "--stations", str(cls.tmp / "stations.csv"), "--stations-ams", str(cls.tmp / "ams.csv"),
               "--out", str(cls.out)]
        cls.proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600,
                                  encoding="utf-8", errors="replace")
        cls.meta = json.loads((cls.out / "meta.json").read_text(encoding="utf-8")) \
            if (cls.out / "meta.json").exists() else None

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_runs_offline(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stderr[-3000:])

    def test_outputs_written(self):
        names = ["temperature_annual_mean_C.tif", "precipitation_annual_total_mm.tif",
                 "precip_24h_mean_annual_max_mm.tif", "dem_grid_m.tif", "meta.json",
                 "diagnostics.png", "map_temperature_precipitation.png",
                 "map_design_precipitation.png", "validation_stations.png", "validation_summary.csv",
                 "hosking_wallis_era5.csv", "hosking_wallis_stations.csv", "growth_curve.csv"]
        names += [f"precip_24h_T{T}_mm.tif" for T in RETURN_PERIODS]
        missing = [n for n in names if not (self.out / n).exists()]
        self.assertEqual(missing, [])

    def test_rasters_share_grid_and_are_plausible(self):
        with rasterio.open(self.out / "temperature_annual_mean_C.tif") as t, \
                rasterio.open(self.out / "precip_24h_T100_mm.tif") as p100, \
                rasterio.open(self.out / "precip_24h_T10_mm.tif") as p10:
            self.assertEqual(t.transform, p100.transform)
            self.assertEqual(t.crs.to_epsg(), 32635)
            T = t.read(1, masked=True)
            a100, a10 = p100.read(1, masked=True), p10.read(1, masked=True)
        self.assertTrue(5 < T.min() and T.max() < 25)
        self.assertTrue((a100 > a10).all())                      # growth curve is increasing

    def test_physical_consistency(self):
        m = self.meta
        self.assertLess(m["chelsa_downscaling"]["T"]["slope_per_100m"], 0)   # cooler uphill
        self.assertTrue(1.0 <= m["f24"]["value"] <= 1.25)
        xs = [g["x_T"] for g in m["growth"]]
        self.assertEqual(xs, sorted(xs))
        prods = {r["product"] for r in m["validation"]}
        self.assertEqual(prods, {"ERA5 raw", "CHELSA raw (1981-2010)", "Hybrid",
                                 "Hybrid+stations (LOOCV)"})

    def test_matches_snapshot(self):
        got = key_numbers(self.meta)
        if os.environ.get("HYBRIDCLIM_UPDATE_SNAPSHOT") == "1":
            SNAPSHOT.write_text(json.dumps(got, indent=2) + "\n", encoding="utf-8")
            self.skipTest("snapshot updated")
        exp = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        for k, v in exp.items():
            self.assertAlmostEqual(got[k], v, delta=1e-3 * max(1.0, abs(v)), msg=k)


if __name__ == "__main__":
    unittest.main()
