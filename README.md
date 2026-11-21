# HybridClim

Station-anchored, DEM-resolution (default 30 m, UTM) **annual temperature, annual precipitation
and 24-hour design-rainfall rasters** for flood models (HEC-HMS, HEC-RAS 2D, SWAT, SCS-CN …),
built from CHELSA × ERA5-Land/ERA5 × DEM, with built-in Hosking–Wallis regional frequency
diagnostics and independent station validation.

## Why hybrid?

| Source | Strength | Weakness | Role |
|---|---|---|---|
| CHELSA v2.1 (1 km) | Station-corrected spatial pattern that resolves orography | Fixed 1981–2010 period, no daily extremes | Spatial pattern |
| ERA5-Land (0.1°) temperature, ERA5 (0.25°) precipitation, via Open-Meteo | 1981–present daily series, hourly data | Absolute bias, underestimates daily extremes | Change since 1981–2010 and extreme statistics |
| DEM (Copernicus GLO-30 or any GeoTIFF) | 30 m elevation | — | Sub-kilometre elevation effect |

ERA5 contributes only its change between two of its own periods (difference for temperature,
ratio for precipitation), so its systematic bias largely cancels while CHELSA's
station-corrected level and pattern are kept.

## Installation

```bash
git clone https://github.com/facanpolat-tr/hybridclim && cd hybridclim
pip install -e .
python -m unittest discover -s tests
```

Python ≥ 3.10. Dependencies: numpy, pandas, pyarrow, scipy, rasterio, pyproj, shapely,
matplotlib, requests. Internet access is needed for Open-Meteo, CHELSA and (by default) the
Copernicus DEM; CHELSA windows are cached in the output folder.

## Quick start

```bash
# 1) ERA5-Land / ERA5 daily series (1981–2024) and 10 years of hourly ERA5 precipitation
hybridclim-download --data-dir data/example --boundary example/example_boundary.geojson

# 2) 30 m surfaces (DEM: Copernicus GLO-30, read remotely)
hybridclim --data-dir data/example
```

Outputs go to `./output/<region>/`. The download stays within Open-Meteo's free-tier limits
(a 12 s pause per request) and can be interrupted and resumed; the example takes about
10 minutes.

### Options of `hybridclim`

| Option | Meaning |
|---|---|
| `--period 1991-2020` | Target climate period (default 1995–2024) |
| `--res 30` | Output resolution (m) |
| `--dem <file>\|copernicus` | DEM GeoTIFF, or Copernicus GLO-30 from AWS Open Data (default; env `HYBRIDCLIM_DEM`) |
| `--stations stations.csv` | Station climatology: `name,lat,lon` + any of `T_annual` (°C), `P_annual` (mm/yr), `P24_mean_annual_max` (mm) |
| `--stations-ams ams.csv` | Station annual-maximum daily precipitation, long format `name,year,P_max` (mm); series with ≥ 10 years are used |
| `--stations-p24 calendar-day\|24h` | Station maxima are calendar-day (default; converted with f24) or sliding 24 h |
| `--no-correction` | Use stations for validation only; leave the maps uncorrected |
| `--growth era5\|stations` | Growth curve used for the design maps |
| `--n-sim 500` | Number of Hosking–Wallis simulations |
| `--out <folder>` | Output folder |

### Data folder

`hybridclim-download` writes `<region>_boundary.geojson`, `daily_era5land/`, `daily_era5/` and
`hourly_era5/` (one Parquet file per grid node). `<region>` is taken from the boundary file
name; the optional GeoJSON property `name` is used in map titles.

## Method

1. **CHELSA → DEM grid:** `X = bilinear(X_1km) + β·(z − bilinear(z̄_1km))`; β is one regression
   slope of CHELSA on mean pixel elevation over the window (temperature: lapse rate;
   precipitation: orographic gradient, applied only if p < 0.05, with a 1 km Gaussian-smoothed DEM).
2. **Change factors:** at each ERA5 cell `ΔT = T̄(target) − T̄(1981–2010)`,
   `CF = P̄(target) / P̄(1981–2010)`, spread by thin-plate spline with smoothing chosen by
   leave-one-out cross-validation (LOOCV); a constant is used when it scores better.
   `T = T_CHELSA + ΔT`, `P = P_CHELSA × CF`.
3. **24-hour design rainfall:** `P24 = f24 · r · P`, with `r = P̄max / P̄annual` from ERA5 and the
   calendar-day → sliding-24 h factor `f24` estimated from hourly ERA5 (fallback WMO 1.13).
   Index-flood regional frequency analysis: annual maxima are pooled, an L-moment GEV growth
   curve is fitted (Gumbel if the shape is bounded above, k > 0), with 90 % intervals from a
   year-block bootstrap. `P24_T = P24 · x_T`, T = 2 … 500 years.
4. **Station correction (optional):** station/map ratio (precipitation) or difference
   (temperature), spread by the same LOOCV-selected surface model.
5. **Independent validation** (`hybridclim/validation.py`): at each station, raw ERA5 (nearest
   cell), raw CHELSA (1 km, 1981–2010), the hybrid surface, and the station-corrected surface
   under LOOCV are compared with the observation (ME, MAE, RMSE, PBIAS, r).
6. **Hosking–Wallis diagnostics** (`hybridclim/rfa.py`): discordancy D_i, heterogeneity H1/H2
   (kappa simulation), goodness-of-fit Z for GEV, GLO, GNO, PE3, GPA, and mean intersite
   correlation ρ̄, for ERA5 cells and station series separately. H < −2 is reported as a sign
   of intersite correlation, not of homogeneity.

## Outputs (`output/<region>/`)

| File | Content |
|---|---|
| `temperature_annual_mean_C.tif` | Mean annual temperature (°C) |
| `precipitation_annual_total_mm.tif` | Annual precipitation (mm) |
| `precip_24h_mean_annual_max_mm.tif` | Mean annual maximum 24 h precipitation (mm) |
| `precip_24h_T{2…500}_mm.tif` | T-year 24 h design precipitation (mm) |
| `dem_grid_m.tif` | DEM on the same grid |
| `meta.json` | Every choice and diagnostic of the run |
| `map_temperature_precipitation.png`, `map_design_precipitation.png` | Maps |
| `diagnostics.png` | Source consistency, LOOCV, growth curves |
| `validation_stations.png` | Station 1:1 plots |
| `points_*.csv`, `growth_curve*.csv`, `hosking_wallis_*.csv`, `validation_*.csv` | Tables |

All GeoTIFFs share one UTM grid (NoData −9999) and extend ~7 km beyond the boundary: clip with
the catchment, not the administrative boundary.

## Limitations

- ERA5 underestimates convective daily extremes; without station correction, design rainfall
  is a lower bound (stated on the maps).
- ERA5 cells are strongly correlated: the effective sample of the pooled analysis is close to
  the record length, so T > 2 × record length (e.g. 500 yr) is extrapolation, and the
  Hosking–Wallis H and Z measures computed on ERA5 cells are not reliable. Use station series
  for distribution choice.
- Sub-daily durations (IDF) need station standard-duration maxima.
- The correction surfaces resolve ~11 km features; finer detail comes from CHELSA and the DEM.

## Tests

```bash
python -m unittest discover -s tests
```

- `test_rfa.py`, `test_validation.py`: unit tests of the analysis modules.
- `test_rfa_reference.py`: regional frequency analysis against Hosking's R packages
  `lmomRFA`/`lmom` on the Cascades region (reference values in `tests/data/lmomrfa_cascades.json`,
  produced by `tests/data/make_lmomrfa_reference.R`).
- `test_end_to_end.py`: the full workflow on a bundled 1.5 MB fixture (`tests/data/fixture`,
  built by `tests/data/make_fixtures.py`) with network access disabled, compared with a stored
  snapshot. After an intended change of results:
  `HYBRIDCLIM_UPDATE_SNAPSHOT=1 python -m unittest tests.test_end_to_end`.

## Contributing and support

Bug reports and feature requests: GitHub issues. Pull requests are welcome; please add a unit
test under `tests/` and make sure `python -m unittest discover -s tests` passes.
Questions: facanpolat@hotmail.com.

## Citation

See [CITATION.cff](CITATION.cff). Data: CHELSA v2.1 (Karger et al., 2017), ERA5 (Hersbach et
al., 2020), ERA5-Land (Muñoz-Sabater et al., 2021), Copernicus DEM GLO-30, Open-Meteo.
