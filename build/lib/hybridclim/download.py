"""
ERA5 / ERA5-Land long-series downloader (Open-Meteo Historical API), step 1 of the workflow.

Daily series 1981-2024:
  * Temperature : ERA5-Land (0.1 deg) nodes covering the boundary      -> <data-dir>/daily_era5land/
  * Precipitation: ERA5 (0.25 deg) nodes covering the boundary + 1 cell -> <data-dir>/daily_era5/
                  (Open-Meteo provides no daily ERA5-Land precipitation, and 0.1 deg points would
                   repeat the 0.25 deg ERA5 value, so precipitation is taken at ERA5 cell centres.)
  * Hourly precipitation: ERA5, last 10 years (--hourly-years) -> <data-dir>/hourly_era5/ (for f24)

Why long series?
  * Change factors: the absolute ERA5 bias largely cancels when two periods of the same model are
    differenced or divided (CHELSA period 1981-2010 vs the target period).
  * A long annual-maximum series for design rainfall.

Models are requested explicitly: "best_match" switches to IFS after 2017 and is not homogeneous.

Usage:
    hybridclim-download --data-dir data/example --boundary example/example_boundary.geojson
Re-runnable: finished files are skipped; waits on minute/hour quotas, stops on the daily quota.
Cost: about 115 weighted API calls per point for 44 years of one daily variable.
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point, shape

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

API_URL = "https://archive-api.open-meteo.com/v1/archive"
START, END = "1981-01-01", "2024-12-31"
ERA5_STEP = 0.25
ERA5L_STEP = 0.1
JOBS = {  # folder: (model, variable, frequency)
    "daily_era5land": ("era5_land", "temperature_2m_mean", "daily"),
    "daily_era5": ("era5", "precipitation_sum", "daily"),
}
HOURLY_JOB = ("hourly_era5", ("era5", "precipitation", "hourly"))
BOUNDARY_SUFFIX = "_boundary.geojson"


def load_geom(data_dir: Path):
    files = sorted(data_dir.glob(f"*{BOUNDARY_SUFFIX}"))
    if not files:
        raise SystemExit(f"No boundary: {data_dir}/*{BOUNDARY_SUFFIX} (pass --boundary)")
    return shape(json.loads(files[0].read_text(encoding="utf-8"))["features"][0]["geometry"])


def grid_points(geom, step, buffer):
    """Grid nodes at `step` degrees covering the boundary buffered by `buffer` degrees."""
    area = geom.buffer(buffer)
    l, b, r, t = area.bounds
    lats = np.arange(np.floor(b / step) * step, np.ceil(t / step) * step + step / 2, step)
    lons = np.arange(np.floor(l / step) * step, np.ceil(r / step) * step + step / 2, step)
    return [(round(float(la), 2), round(float(lo), 2)) for la in lats for lo in lons
            if area.contains(Point(lo, la))]


def era5land_points(data_dir: Path):
    """Point names of an existing hourly/ folder if present, else 0.1 deg ERA5-Land nodes."""
    pts = []
    for f in sorted((data_dir / "hourly").glob("lat*_lon*.parquet")):
        m = re.match(r"lat(-?\d+\.\d+)_lon(-?\d+\.\d+)", f.stem)
        if m:
            pts.append((float(m[1]), float(m[2])))
    return pts or grid_points(load_geom(data_dir), ERA5L_STEP, ERA5L_STEP)


def era5_points(data_dir: Path):
    """0.25 deg ERA5 nodes covering the boundary buffered by one cell."""
    return grid_points(load_geom(data_dir), ERA5_STEP, ERA5_STEP)


def fetch(session, lat, lon, model, var, start, end, freq="daily"):
    params = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
              freq: var, "models": model, "timezone": "auto"}
    r = session.get(API_URL, params=params, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    j = r.json()
    df = pd.DataFrame(j[freq]).rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df[var] = pd.to_numeric(df[var], errors="coerce")
    if df[var].notna().mean() < 0.9:
        raise RuntimeError(f"{model}/{var}: data mostly missing")
    df.insert(0, "lat", round(j["latitude"], 4))
    df.insert(1, "lon", round(j["longitude"], 4))
    df.insert(2, "elevation_m", float(j.get("elevation", np.nan)))
    return df


def wait_for_limit(msg: str) -> bool:
    m = msg.lower()
    if "daily" in m and "limit" in m:
        raise SystemExit("Daily API quota reached. Re-run the same command tomorrow "
                         "(finished files are skipped).")
    if "hourly" in m and "limit" in m:
        log.info("Hourly quota reached, waiting 15 min..."); time.sleep(900); return True
    if "minutely" in m or "429" in m or "too many" in m:
        log.info("Minute quota reached, waiting 65 s..."); time.sleep(65); return True
    return False


def main():
    ap = argparse.ArgumentParser(description="ERA5 / ERA5-Land long-series downloader (Open-Meteo)")
    ap.add_argument("--data-dir", required=True, help="Data folder (created if missing)")
    ap.add_argument("--boundary", help="Region boundary GeoJSON (first feature is used; optional "
                                       "properties.name). Copied to <data-dir>/<stem>_boundary.geojson")
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    ap.add_argument("--hourly-years", type=int, default=10, metavar="N",
                    help="Also download the last N years of hourly ERA5 precipitation for f24 (0 = skip)")
    ap.add_argument("--force", action="store_true", help="Re-download existing files")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    if args.boundary:
        src = Path(args.boundary)
        slug = src.stem.removesuffix("_boundary")
        (data_dir / f"{slug}{BOUNDARY_SUFFIX}").write_text(src.read_text(encoding="utf-8"),
                                                          encoding="utf-8")
    s = requests.Session()
    jobs = list(JOBS.items())
    if args.hourly_years > 0:
        jobs.append(HOURLY_JOB)
    era5l_pts = era5land_points(data_dir)
    for sub, (model, var, freq) in jobs:
        out_dir = data_dir / sub
        out_dir.mkdir(exist_ok=True)
        pts = era5l_pts if model == "era5_land" else era5_points(data_dir)
        start = args.start
        if freq == "hourly":
            start = f"{int(args.end[:4]) - args.hourly_years + 1}-01-01"
        log.info(f"{sub}: {len(pts)} points, {model}/{var}, {start} -> {args.end}")
        for i, (lat, lon) in enumerate(pts, 1):
            f = out_dir / f"lat{lat:.2f}_lon{lon:.2f}.parquet"
            if f.exists() and not args.force:
                try:
                    if pd.read_parquet(f, columns=[var])[var].notna().mean() >= 0.9:
                        continue
                except Exception:
                    pass                                    # old or incomplete file: download again
            tries = 3
            while tries:
                try:
                    df = fetch(s, lat, lon, model, var, start, args.end, freq)
                    df.to_parquet(f, compression="zstd")
                    log.info(f"  [{i}/{len(pts)}] {f.name}: {len(df)} {'hours' if freq == 'hourly' else 'days'}, "
                             f"cell ({df['lat'].iloc[0]}, {df['lon'].iloc[0]}), z={df['elevation_m'].iloc[0]:.0f} m")
                    time.sleep(12.0)                        # stay within the free-tier quotas
                    break
                except SystemExit:
                    raise
                except Exception as e:
                    log.error(f"  [{i}/{len(pts)}] ({lat}, {lon}): {e}")
                    if not wait_for_limit(str(e)):
                        tries -= 1
                        time.sleep(5)
        log.info(f"{sub}: {len(list(out_dir.glob('*.parquet')))} files -> {out_dir}")


if __name__ == "__main__":
    main()
