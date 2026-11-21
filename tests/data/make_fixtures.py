"""
Build the small offline fixture used by tests/test_end_to_end.py.

Source: a normal run of the example (data/example, output/example), i.e.
    hybridclim-download --data-dir data/example --boundary example/example_boundary.geojson
    hybridclim --data-dir data/example
The fixture keeps a few grid nodes, a short hourly record and small CHELSA / DEM clips so the
whole workflow runs without network access in a few seconds.

Data licences: ERA5 / ERA5-Land via Open-Meteo (CC BY 4.0), CHELSA v2.1 (CC BY 4.0),
Copernicus DEM GLO-30 (Copernicus DEM licence, free use with attribution).

Usage (from the repository root):  python tests/data/make_fixtures.py
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[2]
SRC_DATA = ROOT / "data" / "example"
SRC_OUT = ROOT / "output" / "example"
DST = Path(__file__).resolve().parent / "fixture"
BOX = (27.30, 38.10, 27.40, 38.20)              # lon_min, lat_min, lon_max, lat_max
ERA5L_NODES = [(38.1, 27.3), (38.1, 27.4), (38.2, 27.3), (38.2, 27.4), (38.0, 27.3), (38.3, 27.4)]
ERA5_NODES = [(38.0, 27.25), (38.0, 27.5), (38.25, 27.25), (38.25, 27.5)]
HOURLY_NODE = (38.25, 27.25)
HOURLY_YEARS = (2019, 2024)


def node_file(folder, lat, lon):
    return SRC_DATA / folder / f"lat{lat:.2f}_lon{lon:.2f}.parquet"


def copy_series(folder, nodes, var, years=None):
    (DST / "data" / folder).mkdir(parents=True, exist_ok=True)
    for lat, lon in nodes:
        f = node_file(folder, lat, lon)
        d = pd.read_parquet(f)
        if years:
            d = d[(d.index.year >= years[0]) & (d.index.year <= years[1])]
        d[var] = d[var].round(1).astype("float32")
        d.index = d.index.astype("datetime64[s]")
        d.to_parquet(DST / "data" / folder / f.name, compression="zstd", compression_level=19,
                     use_byte_stream_split=[var])


def clip_raster(src_path, dst_path, bounds_ll, scale=1):
    with rasterio.open(src_path) as s:
        if s.crs.to_epsg() == 4326:
            win = from_bounds(*bounds_ll, s.transform).round_offsets().round_lengths()
        else:
            from rasterio.warp import transform_bounds
            win = from_bounds(*transform_bounds(4326, s.crs, *bounds_ll), s.transform)
            win = win.round_offsets().round_lengths()
        h, w = int(win.height // scale), int(win.width // scale)
        a = s.read(1, window=win, out_shape=(h, w), resampling=Resampling.average)
        tr = rasterio.transform.Affine(s.transform.a * win.width / w, 0, s.window_transform(win).c,
                                       0, s.transform.e * win.height / h, s.window_transform(win).f)
        prof = dict(driver="GTiff", height=h, width=w, count=1, dtype="float32", crs=s.crs,
                    transform=tr, nodata=s.nodata, compress="deflate")
        with rasterio.open(dst_path, "w", **prof) as d:
            d.write(a.astype("float32"), 1)


def main():
    if DST.exists():
        shutil.rmtree(DST)
    (DST / "data").mkdir(parents=True)
    (DST / "chelsa").mkdir()
    l, b, r, t = BOX
    geo = {"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {"name": "Offline test area"},
        "geometry": {"type": "Polygon", "coordinates": [[[l, b], [r, b], [r, t], [l, t], [l, b]]]}}]}
    (DST / "data" / "test_boundary.geojson").write_text(json.dumps(geo), encoding="utf-8")
    copy_series("daily_era5land", ERA5L_NODES, "temperature_2m_mean")
    copy_series("daily_era5", ERA5_NODES, "precipitation_sum")
    copy_series("hourly_era5", [HOURLY_NODE], "precipitation", HOURLY_YEARS)
    pad = (l - 0.6, b - 0.6, r + 0.6, t + 0.6)
    for v in ("bio1", "bio12"):
        clip_raster(SRC_OUT / f"chelsa_{v}.tif", DST / "chelsa" / f"chelsa_{v}.tif", pad)
    clip_raster(SRC_OUT / "dem_grid_m.tif", DST / "dem_150m.tif", (l - 0.12, b - 0.12, r + 0.12, t + 0.12),
                scale=5)
    total = sum(f.stat().st_size for f in DST.rglob("*") if f.is_file())
    print(f"fixture written to {DST} ({total / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
