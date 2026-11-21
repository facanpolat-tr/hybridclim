"""
Hybrid high-resolution temperature and precipitation surfaces (CHELSA x ERA5 x DEM)
====================================================================================
Climate surfaces at DEM resolution (default 30 m, UTM) for flood and catchment models
(HEC-HMS, HEC-RAS 2D, SWAT, SCS-CN ...).

Sources and their roles:
  CHELSA v2.1 (1 km, 1981-2010) : station-corrected spatial pattern (orography, rain shadow).
  ERA5-Land / ERA5 (Open-Meteo)  : current period and long daily series (1981-2024);
                                   climate change and extreme-precipitation statistics.
  DEM (Copernicus GLO-30 or file): sub-kilometre elevation effect (lapse rate, orographic gradient).

Method (every step is reported in meta.json and diagnostics.png):
  1) CHELSA -> DEM grid: bilinear + elevation-residual correction
        X(30 m) = bilinear(X_1km) + beta * (z - bilinear(zbar_1km))
     beta is one regression slope of CHELSA on mean pixel elevation over the window
     (T: lapse rate; P: orographic gradient, applied only if p < 0.05, z = 1 km Gaussian-smoothed DEM).
  2) Change factors (the ERA5 bias cancels between its own two periods):
        dT_i = T_ERA5(target) - T_ERA5(1981-2010)   ->  T = T_CHELSA,30m + dT(x, y)
        CF_i = P_ERA5(target) / P_ERA5(1981-2010)   ->  P = P_CHELSA,30m * CF(x, y)
     Point values are spread by thin-plate spline; smoothing (or a constant) is chosen by LOOCV.
  3) Design rainfall:
     - annual maximum daily precipitation of the target period at every ERA5 node;
     - calendar-day -> sliding-24 h factor f24 from hourly ERA5 (fallback WMO/Hershfield 1.13);
     - spatial pattern: P24_mean(x, y) = f24 * r(x, y) * P(x, y), r = Pmax / Pannual;
     - index-flood regional frequency analysis: L-moment GEV growth curve, year-block bootstrap
       90 % interval; P24_T(x, y) = P24_mean(x, y) * x_T.
  4) Optional station correction (ratio for precipitation, difference for temperature).
  5) Station validation of four products (validation.py).
  6) Hosking-Wallis D, H, Z diagnostics for ERA5 cells and station series (rfa.py).

Usage:
    hybridclim --data-dir data/example
    hybridclim --data-dir ... --period 1991-2020 --stations stations.csv --stations-ams ams.csv

Station CSV columns: name, lat, lon and at least one of (same target period):
    T_annual (°C), P_annual (mm/yr), P24_mean_annual_max (mm)
"""

import argparse
import json
import logging
import math
import os
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import shapely
from matplotlib.colors import LightSource
from matplotlib.ticker import MaxNLocator
from pyproj import CRS, Transformer
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds as window_from_bounds
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter
from scipy.stats import linregress
from shapely.geometry import shape

from . import rfa
from . import validation as val
from .download import BOUNDARY_SUFFIX

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DEFAULT_DEM = os.environ.get("HYBRIDCLIM_DEM", "copernicus")
COP_URL = ("/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{lat}_00_{lon}_00_DEM/"
           "Copernicus_DSM_COG_10_{lat}_00_{lon}_00_DEM.tif")
CHELSA_URL = ("https://os.zhdk.cloud.switch.ch/chelsav2/GLOBAL/climatologies/1981-2010/bio/"
              "CHELSA_{v}_1981-2010_V.2.1.tif")
BASE = (1981, 2010)                 # CHELSA period = change-factor reference
CELL_HALF_T = 0.05                  # ERA5-Land cell 0.1 deg (temperature)
CELL_HALF_P = 0.125                 # ERA5 cell 0.25 deg (precipitation)
GRID_PAD_DEG = 0.07                 # margin around the boundary (catchments may extend beyond it)
CHELSA_PAD_DEG = 0.4                # CHELSA window also covers the outer ERA5 cells
PRECIP_DEM_SMOOTH_M = 1000.0
F24_DEFAULT = 1.13                  # WMO (Hershfield) calendar day -> 24 h
RETURN_PERIODS = [2, 5, 10, 25, 50, 100, 500]
SMOOTH_CANDIDATES = [0.0, 0.01, 0.1, 1.0, 10.0]
N_BOOT = 1000
HALO = [pe.withStroke(linewidth=2.2, foreground="white")]


# --- Helpers ------------------------------------------------------------------------------

def utm_crs_for(lon, lat):
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1)


def to_grid(src, src_tr, src_crs, shape_, dst_tr, dst_crs, resampling):
    dst = np.full(shape_, np.nan, dtype=np.float64)
    reproject(src.astype(np.float64), dst, src_transform=src_tr, src_crs=src_crs,
              src_nodata=np.nan, dst_transform=dst_tr, dst_crs=dst_crs, dst_nodata=np.nan,
              resampling=resampling)
    return dst


def write_tif(path, arr, tr, crs, desc):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype="float32", crs=crs, transform=tr, nodata=-9999.0,
                       compress="deflate", tiled=True) as dst:
        dst.write(np.where(np.isfinite(arr), arr, -9999.0).astype("float32"), 1)
        dst.set_band_description(1, desc)


class Surface:
    """Surface from point values: thin-plate spline (degree 1) or constant, chosen by LOOCV."""

    def __init__(self, xy, v, label, log_space=False):
        self.label, self.log = label, log_space
        y = np.log(v) if log_space else np.asarray(v, float)
        self.n = len(y)
        cands = [("constant", None)] + ([("tps", s) for s in SMOOTH_CANDIDATES] if self.n >= 5 else [])
        best = None
        for kind, s in cands:
            pred = np.array([self._fit(xy[m], y[m], kind, s)(xy[i:i + 1])[0]
                             for i in range(self.n) for m in [np.arange(self.n) != i]])
            rmse = float(np.sqrt(np.mean((self._back(pred) - self._back(y)) ** 2)))
            if best is None or rmse < best[2] - 1e-12:
                best = (kind, s, rmse, pred)
        self.kind, self.s, self.rmse, loo = best
        self.loo = self._back(loo)
        self.f = self._fit(xy, y, self.kind, self.s)
        log.info(f"  [{label}] surface: {self.kind}{'' if self.s is None else f' (s={self.s})'}, "
                 f"LOOCV RMSE={self.rmse:.4g}, n={self.n}")

    def _back(self, y):
        return np.exp(y) if self.log else y

    @staticmethod
    def _fit(xy, y, kind, s):
        if kind == "constant" or len(y) < 4:
            mu = float(np.mean(y))
            return lambda q: np.full(len(q), mu)
        mu, sd = y.mean(), y.std() or 1.0
        rbf = RBFInterpolator(xy, (y - mu) / sd, kernel="thin_plate_spline", smoothing=s, degree=1)
        return lambda q: rbf(q) * sd + mu

    def __call__(self, q):
        return self._back(self.f(q))


# --- Input data ---------------------------------------------------------------------------

def load_boundary(data_dir):
    files = sorted(data_dir.glob(f"*{BOUNDARY_SUFFIX}"))
    if not files:
        raise SystemExit(f"No boundary file: {data_dir}/*{BOUNDARY_SUFFIX}")
    feat = json.loads(files[0].read_text(encoding="utf-8"))["features"][0]
    return (shape(feat["geometry"]), feat.get("properties", {}).get("name", ""),
            files[0].name.removesuffix(BOUNDARY_SUFFIX))


def _annual(f, var, target, how):
    """Annual series over complete years + base/target masks; None if coverage is too short."""
    d = pd.read_parquet(f)
    y = d.index.year
    cnt = d.groupby(y)[var].count()
    full = cnt[cnt >= 365].index
    d = d[y.isin(full)]
    g = d.groupby(d.index.year)[var]
    ann = getattr(g, how)()
    b = (ann.index >= BASE[0]) & (ann.index <= BASE[1])
    t = (ann.index >= target[0]) & (ann.index <= target[1])
    if b.sum() < 25 or t.sum() < 0.8 * (target[1] - target[0] + 1):
        log.warning(f"{f.name}: insufficient period coverage, skipped")
        return None
    meta = dict(lat=round(float(d["lat"].iloc[0]), 4), lon=round(float(d["lon"].iloc[0]), 4),
                model_z=float(d["elevation_m"].iloc[0]))
    return d, g, ann, b, t, meta


def _files(data_dir, sub):
    files = sorted((data_dir / sub).glob("*.parquet"))
    if not files:
        raise SystemExit(f"No {sub} data. Run first: hybridclim-download --data-dir ...")
    return files


def load_temp_points(data_dir, target):
    """ERA5-Land (0.1 deg) temperature points: T_base, T_target, dT."""
    rows = []
    for f in _files(data_dir, "daily_era5land"):
        r = _annual(f, "temperature_2m_mean", target, "mean")
        if r:
            _, _, ann, b, t, meta = r
            rows.append({**meta, "T_base": ann[b].mean(), "T_target": ann[t].mean()})
    pts = pd.DataFrame(rows).drop_duplicates(["lat", "lon"]).reset_index(drop=True)
    pts["dT"] = pts["T_target"] - pts["T_base"]
    return pts


def load_precip_points(data_dir, target):
    """ERA5 (0.25 deg) precipitation points: CF and annual maximum daily; plus the AMS table."""
    rows, am_series = [], {}
    for f in _files(data_dir, "daily_era5"):
        r = _annual(f, "precipitation_sum", target, "sum")
        if r:
            _, g, ann, b, t, meta = r
            am = g.max()
            key = (meta["lat"], meta["lon"])
            if key in am_series:
                continue
            rows.append({**meta, "P_base": ann[b].mean(), "P_target": ann[t].mean(),
                         "Pmax_day_mean": am[t].mean(), "n_years_target": int(t.sum())})
            am_series[key] = am[t]
    pts = pd.DataFrame(rows).reset_index(drop=True)
    pts["CF"] = pts["P_target"] / pts["P_base"]
    pts["r_max"] = pts["Pmax_day_mean"] / pts["P_target"]
    return pts, pd.DataFrame(am_series)        # rows: years, columns: points


def chelsa_cell_mean(arr, tr, lat, lon, half):
    """Mean of the 1 km CHELSA pixels inside a model cell (+-half deg)."""
    rows, cols = np.indices(arr.shape)
    plon = tr.c + (cols + 0.5) * tr.a
    plat = tr.f + (rows + 0.5) * tr.e
    m = (np.abs(plon - lon) <= half) & (np.abs(plat - lat) <= half)
    return float(np.nanmean(arr[m])) if np.isfinite(arr[m]).any() else np.nan


def f24_from_hourly(data_dir, target):
    """Pooled ratio of sliding-24 h maximum to calendar-day maximum, from hourly data."""
    num = den = 0.0
    nyr = 0
    files = sorted((data_dir / "hourly").glob("*.parquet")) + sorted((data_dir / "hourly_era5").glob("*.parquet"))
    for f in files:
        p = pd.read_parquet(f, columns=["precipitation"])["precipitation"]
        for yr, py in p.groupby(p.index.year):
            if py.count() < 8700 or not (target[0] - 30 <= yr <= target[1]):
                continue
            roll = py.rolling(24, min_periods=24).sum().max()
            day = py.groupby(py.index.date).sum().max()
            if day > 1:
                num, den, nyr = num + roll, den + day, nyr + 1
    if nyr < 5:
        log.warning(f"Not enough hourly data for f24; using the WMO value {F24_DEFAULT}.")
        return F24_DEFAULT, 0
    f = num / den
    log.info(f"f24 (sliding 24 h / calendar day) = {f:.3f}  ({nyr} point-years)")
    return float(np.clip(f, 1.0, 1.25)), nyr


def copernicus_tiles(bounds_ll):
    """URLs of the 1x1 deg Copernicus GLO-30 tiles (AWS Open Data, COG)."""
    l, b, r, t = bounds_ll
    out = []
    for la in range(math.floor(b), math.ceil(t)):
        for lo in range(math.floor(l), math.ceil(r)):
            out.append(COP_URL.format(lat=f"{'N' if la >= 0 else 'S'}{abs(la):02d}",
                                      lon=f"{'E' if lo >= 0 else 'W'}{abs(lo):03d}"))
    return out


def read_dem(path, bounds, crs, res):
    """Read the DEM onto the target grid. path: a file or 'copernicus' (tiles read remotely)."""
    xmin, ymin, xmax, ymax = bounds
    w, h = int(round((xmax - xmin) / res)), int(round((ymax - ymin) / res))
    tr = from_origin(xmin, ymax, res, res)
    out = np.full((h, w), np.nan, dtype=np.float32)
    if str(path).lower() == "copernicus":
        sources = copernicus_tiles(transform_bounds(crs, 4326, *bounds, densify_pts=21))
        log.info(f"Copernicus GLO-30: reading {len(sources)} tile(s) remotely...")
    else:
        sources = [str(path)]
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="4"):
        for p in sources:
            try:
                src = rasterio.open(p)
            except rasterio.errors.RasterioIOError:
                if len(sources) > 1:                    # no tile exists over open sea
                    continue
                raise
            with src:
                sb = transform_bounds(crs, src.crs, *bounds, densify_pts=21)
                sbb = src.bounds
                if sb[2] < sbb.left or sb[0] > sbb.right or sb[3] < sbb.bottom or sb[1] > sbb.top:
                    continue
                px = abs(src.transform.a)
                win = window_from_bounds(sb[0] - 3 * px, sb[1] - 3 * px, sb[2] + 3 * px, sb[3] + 3 * px,
                                         src.transform).round_offsets().round_lengths()
                a = src.read(1, window=win, boundless=True,
                             fill_value=src.nodata or -9999).astype(np.float32)
                if src.nodata is not None:
                    a[a == src.nodata] = np.nan
                a[a < -500] = np.nan
                tile = np.full_like(out, np.nan)
                reproject(a, tile, src_transform=src.window_transform(win), src_crs=src.crs,
                          src_nodata=np.nan, dst_transform=tr, dst_crs=crs, dst_nodata=np.nan,
                          resampling=Resampling.bilinear)
                out = np.where(np.isnan(out), tile, out)
    if np.isnan(out).mean() > 0.5:
        raise SystemExit(f"DEM does not cover the area: {path}")
    return out, tr


def read_chelsa(var, bounds_ll, cache):
    if cache.exists():
        with rasterio.open(cache) as s:
            return s.read(1).astype(float), s.transform
    log.info(f"Reading CHELSA {var} window remotely...")
    l, b, r, t = bounds_ll
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="4",
                      GDAL_HTTP_RETRY_DELAY="2"):
        with rasterio.open("/vsicurl/" + CHELSA_URL.format(v=var)) as s:
            inv = ~s.transform
            c0, r0 = inv * (l, t)
            c1, r1 = inv * (r, b)
            win = Window(math.floor(c0), math.floor(r0), math.ceil(c1) - math.floor(c0),
                         math.ceil(r1) - math.floor(r0))
            raw = s.read(1, window=win)
            tr = s.window_transform(win)
            arr = raw * s.scales[0] + s.offsets[0]
    arr = np.where(raw == np.iinfo(raw.dtype).max, np.nan, arr).astype(float)
    with rasterio.open(cache, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype="float32", crs="EPSG:4326", transform=tr, nodata=np.nan) as d:
        d.write(arr.astype("float32"), 1)
    return arr, tr


def downscale_chelsa(x1k, tr1k, dem_used, dem, tr, crs, label, only_if_significant):
    """Downscale CHELSA to the DEM grid. Returns (surface, info)."""
    ll = "EPSG:4326"
    z1k = to_grid(dem, tr, crs, x1k.shape, tr1k, ll, Resampling.average)
    xb = to_grid(x1k, tr1k, ll, dem.shape, tr, crs, Resampling.bilinear)
    xn = to_grid(x1k, tr1k, ll, dem.shape, tr, crs, Resampling.nearest)
    xb = np.where(np.isnan(xb), xn, xb)
    zb = to_grid(z1k, tr1k, ll, dem.shape, tr, crs, Resampling.bilinear)
    ok = np.isfinite(z1k) & np.isfinite(x1k)
    r = linregress(z1k[ok], x1k[ok])
    apply = (r.pvalue < 0.05) or not only_if_significant
    info = dict(n_1km=int(ok.sum()), slope_per_100m=float(r.slope * 100), r2=float(r.rvalue ** 2),
                p=float(r.pvalue), applied=bool(apply))
    out = xb + (r.slope * np.where(np.isfinite(zb), dem_used - zb, 0.0) if apply else 0.0)
    log.info(f"  [{label}] CHELSA 1 km px={info['n_1km']}, slope={info['slope_per_100m']:+.3f}/100 m, "
             f"R²={info['r2']:.2f} -> 30 m correction {'APPLIED' if apply else 'not applied'}")
    return out, info


# --- Frequency analysis -------------------------------------------------------------------

def growth_and_tests(wide, label, n_sim):
    """Regional growth curve + Hosking-Wallis D/H/Z report."""
    tab, k, k_raw = rfa.regional_growth(wide, RETURN_PERIODS, n_boot=N_BOOT)
    if k != k_raw:
        log.info(f"[{label}] GEV k={k_raw:+.3f} > 0 (bounded above) -> Gumbel (k=0) on the safe side.")
    log.info(f"[{label}] growth x_T: " + ", ".join(f"T{t}={v:.2f}" for t, v in zip(tab.T_years, tab.x_T)))
    st, rep = rfa.hw_report(wide, n_sim=n_sim)
    rep.update(gev_k=k, gev_k_raw=k_raw)
    if rep.get("H1") is not None:
        log.info(f"[{label}] N={rep['N']}, H1={rep['H1']:.2f} ({rep['H1_class']}), H2={rep['H2']:.2f}, "
                 f"rho={rep['rho_mean']:.2f}, Z: " + ", ".join(f"{d}={z:+.2f}" for d, z in rep["Z"].items()))
        if rep["rho_mean"] > 0.3:
            log.warning(f"[{label}] high intersite correlation (rho={rep['rho_mean']:.2f}): H is biased "
                        "towards homogeneity and the effective sample is smaller than N x n.")
        if "GEV" not in rep["Z_accepted"]:
            log.warning(f"[{label}] GEV |Z| > 1.64: GEV may not fit "
                        f"(accepted: {rep['Z_accepted'] or 'none'}).")
    if rep["discordant"]:
        log.warning(f"[{label}] discordant sites (D > {rep['D_critical']}): {rep['discordant']}")
    return tab, k, st, rep


# --- Station correction -------------------------------------------------------------------

def station_correction(st, fields, to_utm, gx, gy, tr):
    """Correct the maps with the station/map ratio (precipitation) or difference (temperature).

    Returns (report, long table of LOOCV predictions).
    """
    sx, sy = to_utm.transform(st["lon"].values, st["lat"].values)
    col = ((np.array(sx) - tr.c) / tr.a).astype(int)
    row = ((np.array(sy) - tr.f) / tr.e).astype(int)
    inn = (row >= 0) & (row < gx.shape[0]) & (col >= 0) & (col < gx.shape[1])
    report, loo_rows = {}, []
    for key, (colname, _, mult) in val.VARS.items():
        if colname not in st or key not in fields:
            continue
        m = inn & st[colname].notna().values
        if m.sum() == 0:
            continue
        mv = fields[key][row[m], col[m]]
        ok = np.isfinite(mv)
        obs, mv = st[colname].values[m][ok], mv[ok]
        corr = obs / mv if mult else obs - mv
        xy = np.c_[np.array(sx)[m][ok], np.array(sy)[m][ok]] / 1000
        surf = Surface(xy, corr, f"stations-{key}", log_space=mult)
        q = np.c_[gx.ravel(), gy.ravel()]
        c = surf(q).reshape(gx.shape)
        before = float(np.sqrt(np.mean((mv - obs) ** 2)))
        loo = mv * surf.loo if mult else mv + surf.loo
        after = float(np.sqrt(np.mean((loo - obs) ** 2)))
        report[key] = dict(n=int(ok.sum()), rmse_before=before, rmse_after_loocv=after,
                           surface=surf.kind, mean_correction=float(np.mean(corr)))
        names = st["name"].astype(str).values[m][ok]
        loo_rows += [dict(name=a, variable=key, product="Hybrid+stations (LOOCV)", observed=float(o),
                          predicted=float(p)) for a, o, p in zip(names, obs, loo)]
        log.info(f"  [stations-{key}] n={ok.sum()}, RMSE before={before:.2f}, after (LOOCV)={after:.2f}")
        fields[key] = fields[key] * c if mult else fields[key] + c
    return report, pd.DataFrame(loo_rows)


# --- Plotting -----------------------------------------------------------------------------

def nice_levels(vmin, vmax, n=10):
    return MaxNLocator(nbins=n, steps=[1, 2, 2.5, 5, 10]).tick_values(vmin, vmax)


def draw(ax, fig, field, hs, extent, geom_utm, pxy, crs, cmap, title, unit, fmt, note,
         pt_label="ERA5 cell centre"):
    xmin, xmax, ymin, ymax = extent
    ax.imshow(hs, extent=extent, cmap="gray", vmin=0, vmax=1, origin="upper", zorder=0)
    lv = nice_levels(np.nanpercentile(field, 1), np.nanpercentile(field, 99), 10)
    cf = ax.contourf(field, levels=lv, extent=extent, origin="upper", cmap=cmap, alpha=0.75,
                     extend="both", zorder=1)
    cs = ax.contour(field, levels=lv[1:-1:2], extent=extent, origin="upper", colors="#1d3557",
                    linewidths=0.45, alpha=0.8, zorder=2)
    for t in ax.clabel(cs, fmt=fmt, fontsize=6.5, inline=True):
        t.set_path_effects(HALO)
    for poly in getattr(geom_utm, "geoms", [geom_utm]):
        x, y = poly.exterior.xy
        ax.plot(x, y, color="white", lw=3.0, zorder=3)
        ax.plot(x, y, color="#202020", lw=1.3, zorder=4)
    ax.scatter(pxy[:, 0], pxy[:, 1], s=22, marker="+", color="#202020", lw=1.1, zorder=6,
               label=pt_label)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.85)

    to_geo = Transformer.from_crs(crs, 4326, always_xy=True)
    to_utm = Transformer.from_crs(4326, crs, always_xy=True)
    lon_a, lat_a = to_geo.transform([xmin, xmax], [ymin, ymax])
    lons = np.arange(math.ceil(min(lon_a) * 10) / 10, max(lon_a), 0.1)
    lats = np.arange(math.ceil(min(lat_a) * 10) / 10, max(lat_a), 0.1)
    yc, xc = (ymin + ymax) / 2, (xmin + xmax) / 2
    ax.set_xticks([to_utm.transform(l, to_geo.transform(xc, ymin)[1])[0] for l in lons])
    ax.set_xticklabels([f"{l:.1f}°E" for l in lons])
    ax.set_yticks([to_utm.transform(to_geo.transform(xmin, yc)[0], l)[1] for l in lats])
    ax.set_yticklabels([f"{l:.1f}°N" for l in lats], rotation=90, va="center")
    ax.tick_params(labelsize=7, direction="in", length=3)

    ax.annotate("", xy=(0.94, 0.95), xytext=(0.94, 0.87), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.7", color="#202020"))
    ax.text(0.94, 0.955, "N", transform=ax.transAxes, ha="center", va="bottom", fontsize=9,
            fontweight="bold", path_effects=HALO)
    target_len = (xmax - xmin) / 4                      # scale bar ~ a quarter of the width
    e = 10 ** math.floor(math.log10(target_len))
    seg = max(b for b in (e, 2 * e, 5 * e) if b <= target_len) / 2
    x0, y0 = xmin + (xmax - xmin) * 0.05, ymin + (ymax - ymin) * 0.04
    for i in range(2):
        ax.add_patch(plt.Rectangle((x0 + i * seg, y0), seg, (ymax - ymin) * 0.008,
                                   facecolor="#202020" if i == 0 else "white", edgecolor="#202020",
                                   lw=0.8, zorder=8))
    for d, s in ((0, "0"), (seg, f"{seg / 1000:g}"), (2 * seg, f"{2 * seg / 1000:g} km")):
        ax.text(x0 + d, y0 + (ymax - ymin) * 0.014, s, ha="center", va="bottom", fontsize=6.5,
                path_effects=HALO, zorder=8)
    cb = fig.colorbar(cf, ax=ax, orientation="horizontal", shrink=0.85, pad=0.06, aspect=35)
    cb.set_label(unit, fontsize=8); cb.ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=10.5, loc="left", pad=6)
    if note:
        ax.text(0.0, -0.035, note, transform=ax.transAxes, fontsize=6.3, color="#555555", va="top")


def diagnostics(tp, pp, surfs, growth, k, out, growth_st=None, k_st=None):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5), dpi=170, layout="constrained")
    for ax, df_, xc, yc, pad, xl, yl, ttl in (
            (axes[0, 0], tp, "T_chelsa_cell", "T_base", 0.3, "CHELSA cell mean 1981–2010, °C",
             "ERA5-Land 1981–2010, °C", "Temperature: source consistency (same period)"),
            (axes[0, 1], pp, "P_chelsa_cell", "P_base", 40, "CHELSA cell mean 1981–2010, mm/yr",
             "ERA5 1981–2010, mm/yr", "Precipitation: source consistency (ERA5 bias cancels in CF)")):
        ax.scatter(df_[xc], df_[yc], color="#2F6EBA", s=36, edgecolor="white")
        lim = [np.nanmin([df_[xc].min(), df_[yc].min()]) - pad,
               np.nanmax([df_[xc].max(), df_[yc].max()]) + pad]
        ax.plot(lim, lim, "--", color="#999")
        ax.set(xlabel=xl, ylabel=yl, title=ttl)
    ax = axes[1, 0]
    names = ["ΔT (°C)", "CF (-)", "Pmax/Pannual (-)"]
    vals = [surfs["dT"].rmse, surfs["CF"].rmse / pp.CF.mean() * 100,
            surfs["r_max"].rmse / pp.r_max.mean() * 100]
    bars = ax.bar(names, vals, color="#2F6EBA")
    for b_, v, u in zip(bars, vals, ["°C", "%", "%"]):
        ax.text(b_.get_x() + b_.get_width() / 2, v, f"{v:.2f} {u}", ha="center", va="bottom", fontsize=8)
    ax.set(ylabel="LOOCV RMSE (ΔT: °C, others: %)",
           title="Correction surfaces: leave-one-out cross-validation")
    ax = axes[1, 1]
    T = growth["T_years"]
    ax.fill_between(T, growth["x_T_lo5"], growth["x_T_hi95"], color="#2F6EBA", alpha=.2,
                    label="90 % confidence interval (bootstrap)")
    ax.plot(T, growth["x_T"], "o-", color="#2F6EBA",
            label="ERA5 regional Gumbel (GEV k>0 → k=0)" if k == 0 else f"ERA5 regional GEV (k={k:+.2f})")
    if growth_st is not None:
        ax.fill_between(T, growth_st["x_T_lo5"], growth_st["x_T_hi95"], color="#C0392B", alpha=.15)
        ax.plot(T, growth_st["x_T"], "s-", color="#C0392B",
                label="Station regional Gumbel" if k_st == 0 else f"Station regional GEV (k={k_st:+.2f})")
    ax.set_xscale("log"); ax.set_xticks(T); ax.set_xticklabels(T)
    ax.set(xlabel="Return period (years)", ylabel="Growth factor x_T",
           title="Extreme-precipitation growth curve")
    ax.legend(fontsize=8, frameon=False)
    for a in axes.flat:
        a.grid(color="#E5E5E5"); a.set_axisbelow(True); a.title.set_fontsize(9.5)
        a.xaxis.label.set_fontsize(8); a.yaxis.label.set_fontsize(8)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


# --- Main ---------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Hybrid CHELSA x ERA5 x DEM climate surfaces")
    ap.add_argument("--data-dir", required=True, help="Data folder written by hybridclim-download")
    ap.add_argument("--dem", default=DEFAULT_DEM,
                    help="DEM GeoTIFF path, or 'copernicus' (GLO-30 read remotely from AWS; default, "
                         "can be changed with the HYBRIDCLIM_DEM environment variable)")
    ap.add_argument("--res", type=float, default=30.0, help="Output resolution (m)")
    ap.add_argument("--period", default="1995-2024", help="Target climate period YYYY-YYYY")
    ap.add_argument("--stations", help="Optional station CSV (see module docstring)")
    ap.add_argument("--stations-ams", help="Station annual-maximum daily precipitation CSV: name,year,P_max")
    ap.add_argument("--stations-p24", choices=["calendar-day", "24h"], default="calendar-day",
                    help="Definition of station P24/AMS values: calendar-day maxima (converted with f24) "
                         "or sliding 24 h maxima")
    ap.add_argument("--growth", choices=["era5", "stations"], default="era5",
                    help="Growth curve for the design maps (stations: needs --stations-ams)")
    ap.add_argument("--n-sim", type=int, default=500, help="Number of Hosking-Wallis H/Z simulations")
    ap.add_argument("--no-correction", action="store_true",
                    help="Use stations for validation only; do not correct the maps")
    ap.add_argument("--out", help="Output folder (default: ./output/<region>)")
    args = ap.parse_args()
    if args.growth == "stations" and not args.stations_ams:
        raise SystemExit("--growth stations requires --stations-ams.")
    if args.stations_ams and not args.stations:
        raise SystemExit("--stations-ams requires --stations (station coordinates).")

    target = tuple(int(v) for v in args.period.split("-"))
    data_dir = Path(args.data_dir).resolve()
    is_cop = args.dem.lower() == "copernicus"
    dem_path = args.dem if is_cop else Path(args.dem)
    dem_name = "Copernicus GLO-30" if is_cop else dem_path.stem
    if not is_cop and not dem_path.exists():
        raise SystemExit(f"DEM not found: {dem_path}  (--dem <file> or --dem copernicus)")
    geom, label, slug = load_boundary(data_dir)
    name = label.split(",")[0] if label else slug
    out = Path(args.out) if args.out else Path.cwd() / "output" / slug
    out.mkdir(parents=True, exist_ok=True)
    per = f"{target[0]}–{target[1]}"

    # 1) Grid and DEM
    c = geom.centroid
    crs = utm_crs_for(c.x, c.y)
    to_utm = Transformer.from_crs(4326, crs, always_xy=True)
    geom_utm = shapely.transform(geom, lambda a: np.c_[to_utm.transform(a[:, 0], a[:, 1])])
    gb = geom.buffer(GRID_PAD_DEG).bounds
    bu = transform_bounds(4326, crs, *gb, densify_pts=21)
    res = args.res
    bu = (math.floor(bu[0] / res) * res, math.floor(bu[1] / res) * res,
          math.ceil(bu[2] / res) * res, math.ceil(bu[3] / res) * res)
    dem, tr = read_dem(dem_path, bu, crs, res)
    dem_f = np.where(np.isnan(dem), np.nanmean(dem), dem)
    dem_s = gaussian_filter(dem_f, PRECIP_DEM_SMOOTH_M / res)
    log.info(f"Grid: {dem.shape[1]}x{dem.shape[0]} px @ {res:g} m, {crs.name}")
    rows, cols = np.indices(dem.shape)
    gx = (tr.c + (cols + 0.5) * tr.a) / 1000.0
    gy = (tr.f + (rows + 0.5) * tr.e) / 1000.0
    q = np.c_[gx.ravel(), gy.ravel()]

    # 2) CHELSA -> 30 m
    bll = transform_bounds(crs, 4326, *bu, densify_pts=21)
    bll = (bll[0] - CHELSA_PAD_DEG, bll[1] - CHELSA_PAD_DEG,
           bll[2] + CHELSA_PAD_DEG, bll[3] + CHELSA_PAD_DEG)
    t1k, ttr = read_chelsa("bio1", bll, out / "chelsa_bio1.tif")
    p1k, ptr = read_chelsa("bio12", bll, out / "chelsa_bio12.tif")
    T_ch, info_T = downscale_chelsa(t1k, ttr, dem_f, dem_f, tr, crs, "T", only_if_significant=False)
    P_ch, info_P = downscale_chelsa(p1k, ptr, dem_s, dem_f, tr, crs, "P", only_if_significant=True)
    P_ch = np.clip(P_ch, 0, None)

    # 3) ERA5 points and change-factor surfaces
    tp = load_temp_points(data_dir, target)
    pp, am_df = load_precip_points(data_dir, target)
    if len(tp) < 3 or len(pp) < 3:
        raise SystemExit("At least 3 temperature and 3 precipitation points are required.")
    for df_, arr, tr1, half, col in ((tp, t1k, ttr, CELL_HALF_T, "T_chelsa_cell"),
                                     (pp, p1k, ptr, CELL_HALF_P, "P_chelsa_cell")):
        df_[col] = [chelsa_cell_mean(arr, tr1, la, lo, half) for la, lo in zip(df_.lat, df_.lon)]
    txy = np.c_[to_utm.transform(tp["lon"].values, tp["lat"].values)]
    pxy = np.c_[to_utm.transform(pp["lon"].values, pp["lat"].values)]
    log.info(f"Temperature: {len(tp)} ERA5-Land points | Precipitation: {len(pp)} ERA5 points | "
             f"base {BASE[0]}–{BASE[1]}, target {per}")
    log.info(f"  mean dT={tp.dT.mean():+.2f} °C, mean CF={pp.CF.mean():.3f}, "
             f"ERA5/CHELSA P (base)={np.nanmean(pp.P_base / pp.P_chelsa_cell):.2f}, "
             f"ERA5L-CHELSA T (base)={np.nanmean(tp.T_base - tp.T_chelsa_cell):+.2f} °C")

    surfs = {"dT": Surface(txy / 1000, tp["dT"].values, "dT"),
             "CF": Surface(pxy / 1000, pp["CF"].values, "CF", log_space=True),
             "r_max": Surface(pxy / 1000, pp["r_max"].values, "Pmax/Pannual", log_space=True)}
    dT = surfs["dT"](q).reshape(dem.shape)
    CF = surfs["CF"](q).reshape(dem.shape)
    rM = surfs["r_max"](q).reshape(dem.shape)
    tp["dT_loocv"] = surfs["dT"].loo
    pp["CF_loocv"], pp["r_max_loocv"] = surfs["CF"].loo, surfs["r_max"].loo

    T_h = T_ch + dT
    P_h = P_ch * CF
    f24, n_f24 = f24_from_hourly(data_dir, target)
    P24 = f24 * rM * P_h
    fields = {"T": T_h, "P": P_h, "P24": P24}

    st_rep, val_rep, st_wide = {}, {}, None
    if args.stations:
        st, st_wide, conv = val.read_stations(args.stations, args.stations_ams, target,
                                              args.stations_p24, f24)
        log.info(f"Station validation: {len(st)} stations"
                 f"{f', {st_wide.shape[1]} AMS series' if st_wide is not None else ''}"
                 f" (P24 conversion x{conv:.3f})")
        vtab = val.sample_products(st, tp, pp, {"T": (t1k, ttr), "P": (p1k, ptr)},
                                   {"T": T_h, "P": P_h, "P24": P24}, tr, to_utm, f24)
        if not args.no_correction:
            log.info("Applying station correction...")
            st_rep, loo = station_correction(st, fields, to_utm, gx, gy, tr)
            T_h, P_h, P24 = fields["T"], fields["P"], fields["P24"]
            vtab = pd.concat([vtab, loo], ignore_index=True)
        if len(vtab):
            mt = val.metrics(vtab)
            vtab.round(4).to_csv(out / "validation_stations.csv", index=False)
            mt.round(4).to_csv(out / "validation_summary.csv", index=False)
            val.plot(vtab, out / "validation_stations.png", plt)
            val_rep = mt.astype({"product": str, "variable": str}).round(4).to_dict("records")
            for r_ in val_rep:
                log.info(f"  [validation {r_['variable']}] {r_['product']:<26} n={r_['n']:<3} "
                         f"ME={r_['ME']:+.3g}  RMSE={r_['RMSE']:.3g}")

    growth_e, k_e, hw_e_tab, hw_e = growth_and_tests(am_df, "ERA5", args.n_sim)
    hw_e_tab.round(4).to_csv(out / "hosking_wallis_era5.csv", index=False)
    growth_s = k_s = hw_s = None
    if st_wide is not None and st_wide.shape[1] >= 2:
        growth_s, k_s, hw_s_tab, hw_s = growth_and_tests(st_wide, "Stations", args.n_sim)
        hw_s_tab.round(4).to_csv(out / "hosking_wallis_stations.csv", index=False)
        growth_s.round(4).to_csv(out / "growth_curve_stations.csv", index=False)
        for T_, a_, b_ in zip(RETURN_PERIODS, growth_e.x_T, growth_s.x_T):
            log.info(f"  T{T_}: x_T ERA5={a_:.3f}  stations={b_:.3f}  ratio={a_ / b_:.3f}")
    elif args.growth == "stations":
        raise SystemExit("A station growth curve needs at least 2 stations with >= 10 years of AMS.")
    growth, k = (growth_s, k_s) if args.growth == "stations" else (growth_e, k_e)
    nodata = np.isnan(dem)
    for a in (T_h, P_h, P24):
        a[nodata] = np.nan

    # 4) Outputs
    write_tif(out / "temperature_annual_mean_C.tif", T_h, tr, crs, f"annual_mean_temperature_C_{per}")
    write_tif(out / "precipitation_annual_total_mm.tif", P_h, tr, crs, f"annual_total_precipitation_mm_{per}")
    write_tif(out / "precip_24h_mean_annual_max_mm.tif", P24, tr, crs, f"mean_annual_max_24h_precip_mm_{per}")
    design = {}
    for T, xT in zip(growth["T_years"], growth["x_T"]):
        design[T] = P24 * xT
        write_tif(out / f"precip_24h_T{T}_mm.tif", design[T], tr, crs, f"P24_T{T}yr_mm")
    write_tif(out / "dem_grid_m.tif", dem, tr, crs, "dem_m")
    tp.round(4).to_csv(out / "points_temperature_era5land.csv", index=False)
    pp.round(4).to_csv(out / "points_precipitation_era5.csv", index=False)
    growth.round(4).to_csv(out / "growth_curve.csv", index=False)

    inside = geometry_mask([geom_utm], out_shape=dem.shape, transform=tr, invert=True)

    def zs(a):
        v = a[inside & np.isfinite(a)]
        return dict(min=float(v.min()), mean=float(v.mean()), max=float(v.max()))
    meta = dict(region=label, target_period=per, base_period=f"{BASE[0]}–{BASE[1]}", crs=crs.to_string(),
                resolution_m=res, dem=str(dem_path), n_points_temperature=len(tp),
                n_points_precipitation=len(pp),
                chelsa_downscaling=dict(T=info_T, P=info_P),
                source_consistency_base=dict(
                    ERA5L_minus_CHELSA_T=float(np.nanmean(tp.T_base - tp.T_chelsa_cell)),
                    ERA5_over_CHELSA_P=float(np.nanmean(pp.P_base / pp.P_chelsa_cell))),
                change_factors=dict(dT_mean=float(tp.dT.mean()), CF_mean=float(pp.CF.mean()),
                                    surfaces={k_: dict(type=s.kind, smoothing=s.s, loocv_rmse=s.rmse)
                                              for k_, s in surfs.items()}),
                f24=dict(value=f24, point_years=n_f24), gev_k=float(k),
                growth_source=args.growth,
                growth=growth.round(4).to_dict("records"),
                growth_era5=growth_e.round(4).to_dict("records"),
                growth_stations=None if growth_s is None else growth_s.round(4).to_dict("records"),
                hosking_wallis=dict(era5=hw_e, stations=hw_s),
                station_correction=st_rep, validation=val_rep,
                station_p24_definition=args.stations_p24 if args.stations else None,
                within_boundary=dict(T=zs(T_h), P=zs(P_h), P24_mean=zs(P24),
                                     **{f"P24_T{T}": zs(v) for T, v in design.items()}),
                created=date.today().isoformat())
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=float),
                                   encoding="utf-8")
    diagnostics(tp, pp, surfs, growth_e, k_e, out / "diagnostics.png", growth_s, k_s)

    ls = LightSource(azdeg=315, altdeg=45)
    hs = ls.hillshade(dem_f, vert_exag=2.0, dx=res, dy=res)
    extent = [bu[0], bu[0] + dem.shape[1] * res, bu[3] - dem.shape[0] * res, bu[3]]
    src = (f"Data: CHELSA v2.1 (1 km, 1981–2010) × Open-Meteo ERA5-Land T ({len(tp)}) / "
           f"ERA5 P ({len(pp)} cells, "
           f"{per}) × {dem_name} {res:g} m{' × stations' if st_rep else ''} · "
           f"{crs.name} · {date.today().isoformat()}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 8.4), dpi=220, layout="constrained")
    draw(axes[0], fig, T_h, hs, extent, geom_utm, txy, crs, "RdYlBu_r",
         "(a) Mean annual temperature", "°C", "%.1f",
         f"CHELSA + lapse {info_T['slope_per_100m']:+.2f} °C/100 m + ΔT ({surfs['dT'].kind}, "
         f"mean {tp.dT.mean():+.2f} °C)", "ERA5-Land cell centre")
    draw(axes[1], fig, P_h, hs, extent, geom_utm, pxy, crs, "YlGnBu",
         "(b) Mean annual precipitation", "mm / yr", "%.0f",
         f"CHELSA × CF ({surfs['CF'].kind}, mean {pp.CF.mean():.3f})"
         f"{', orographic ' + format(info_P['slope_per_100m'], '+.1f') + ' mm/100 m' if info_P['applied'] else ''}")
    fig.suptitle(f"{name} — Hybrid temperature and precipitation ({per})", fontsize=14, x=0.01, ha="left")
    fig.text(0.01, -0.01, src, fontsize=7, color="#555555")
    fig.savefig(out / "map_temperature_precipitation.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    show = [None, 25, 100, 500]
    fig, axes = plt.subplots(2, 2, figsize=(14, 15), dpi=200, layout="constrained")
    for ax, T in zip(axes.flat, show):
        if T is None:
            draw(ax, fig, P24, hs, extent, geom_utm, pxy, crs, "PuBu",
                 "(a) Mean annual maximum 24-hour precipitation", "mm", "%.0f",
                 f"f24={f24:.3f} · ERA5 Pmax/Pannual ratio × hybrid precipitation")
        else:
            g = growth[growth.T_years == T].iloc[0]
            draw(ax, fig, design[T], hs, extent, geom_utm, pxy, crs, "YlOrRd",
                 f"({'bcd'[show.index(T) - 1]}) {T}-year 24-hour precipitation", "mm", "%.0f",
                 f"x_T={g.x_T:.2f} (90 % CI {g.x_T_lo5:.2f}–{g.x_T_hi95:.2f}) · "
                 f"{'regional Gumbel' if k == 0 else 'regional GEV'} "
                 f"({'station' if args.growth == 'stations' else 'ERA5'} growth curve)")
    fig.suptitle(f"{name} — Design precipitation (24 h, {per})", fontsize=14, x=0.01, ha="left")
    fig.text(0.01, -0.005, src + ("" if st_rep.get("P24") else
             " · WARNING: no station correction; ERA5 usually underestimates daily extremes."),
             fontsize=7, color="#555555")
    fig.savefig(out / "map_design_precipitation.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    b = meta["within_boundary"]
    log.info(f"Within boundary: T {b['T']['min']:.1f}–{b['T']['max']:.1f} °C | P {b['P']['min']:.0f}–"
             f"{b['P']['max']:.0f} mm/yr | P24 mean {b['P24_mean']['mean']:.0f} mm | "
             f"P24 T100 {b['P24_T100']['min']:.0f}–{b['P24_T100']['max']:.0f} mm")
    log.info(f"Done -> {out}")


if __name__ == "__main__":
    main()
