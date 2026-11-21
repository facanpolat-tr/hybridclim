---
title: 'HybridClim: station-anchored, DEM-resolution temperature, precipitation and design-rainfall surfaces from CHELSA and ERA5'
tags:
  - Python
  - hydrology
  - design rainfall
  - regional frequency analysis
  - L-moments
  - reanalysis
  - downscaling
authors:
  - name: Fethi Ahmet Canpolat
    orcid: 0000-0002-6084-7735
    affiliation: 1
affiliations:
  - name: Fırat University, Elazığ, Türkiye
    index: 1
date: 22 November 2026
bibliography: paper.bib
---

# Summary

Flood and catchment models such as HEC-HMS, HEC-RAS 2D and SWAT are increasingly run on
10–30 m terrain grids, yet their climate inputs usually come from sparse rain gauges, 1 km
climatologies or 10–30 km reanalyses. `HybridClim` is a Python package that produces, for any
region, mean annual temperature, annual precipitation and T-year 24-hour design-rainfall
rasters on the grid of a digital elevation model (DEM). It combines the station-corrected
1 km CHELSA v2.1 climatology [@karger2017] for the spatial pattern, ERA5-Land
[@munozsabater2021] and ERA5 [@hersbach2020] daily series retrieved through the Open-Meteo
API [@zippenfenig2023] for change since 1981–2010 and for extremes, and a 30 m DEM
(Copernicus GLO-30 by default) for sub-kilometre elevation effects. Design rainfall is derived
by an index-flood regional frequency analysis with L-moment GEV growth curves
[@hosking1990; @hoskingwallis1997]. Every run writes its choices and diagnostics to a
machine-readable report, so no surface is delivered without an estimate of its error.

# Statement of need

Engineers and researchers who build flood models in sparsely gauged regions must assemble
climate inputs by hand: interpolating a few gauges, rescaling a global climatology, or taking
reanalysis values at face value. Each choice is rarely documented, and the error of the
resulting rasters is rarely quantified. Reanalyses are known to underestimate convective
daily extremes, and climatologies cover a fixed past period, so neither alone is adequate for
design.

`HybridClim` targets hydrologists, flood-hazard practitioners and students who need
reproducible, terrain-resolution climate inputs and an honest statement of their uncertainty.
Two commands download the data and build the rasters. When gauge data are available, the same
run validates four products at each station — raw ERA5, raw CHELSA, the hybrid surface, and the
station-corrected surface under leave-one-out cross-validation — and compares regional growth
curves estimated from ERA5 and from gauges. This makes the software suitable both for
practice (design rainfall with confidence intervals for ungauged catchments) and for research
on how far reanalysis-based design values can substitute for gauges.

# State of the field

*High-resolution climate data and downscaling tools.* Global climatologies (CHELSA, WorldClim)
and reanalyses (ERA5, ERA5-Land) are distributed as data, and CHELSA-W5E5 [@karger2023w5e5]
provides daily 1 km forcing. Several tools derive local climate from these sources:
`chelsa-cmip6` [@karger2023cmip6] applies the delta-change method to CHELSA for user-defined
regions and future periods, ClimateDT [@marchi2024] downscales CHELSA-based monthly series
with a local lapse rate through a web portal, `KrigR` [@kusch2022] downloads ERA5(-Land) and
downscales it by kriging in R, and `climate4R` [@iturbide2019] focuses on statistical
downscaling of climate model output. These tools target climatologies at about 1 km and do
not estimate design rainfall.

*Regional frequency analysis.* L-moment regional frequency analysis is implemented in the R
packages `lmom` and `lmomRFA` [@lmomRFA], in Hosking's original Fortran routines and their
Python port `lmoments3` [@lmoments3], and in `RainFA` [@rainfa], which adds clustering and
Hosking–Wallis tests for gauge series. All of them operate on station series and do not
produce gridded surfaces. @rivoire2022 mapped high return levels of daily ERA5 precipitation
over Europe by regionalised extreme-value models at the native 0.25° grid, and
@touloumidis2026 built a Google Earth Engine workflow for ERA5-Land design rainfall, showing
that gauge-based design values exceed ERA5-Land values by 13–46 %; neither downscales to
terrain resolution or corrects with stations.

To our knowledge no open tool combines (i) change-factor coupling of a station-corrected
climatology with a reanalysis, (ii) DEM-resolution downscaling, (iii) Hosking–Wallis regional
frequency analysis with full diagnostics and (iv) built-in multi-product station validation
in one reproducible workflow. `HybridClim` reuses established methods rather than proposing a
new downscaling algorithm; its contribution is to chain them with explicit, logged choices and
error estimates for flood-model inputs.

# Software design

The package has two command-line entry points and two importable modules:

- `hybridclim-download` (`hybridclim.download`) retrieves 1981–present daily ERA5-Land temperature
  and ERA5 precipitation for grid nodes covering a user-supplied boundary, plus recent hourly
  ERA5 precipitation, and caches them as Parquet files. It respects the free API limits and
  resumes after interruption.
- `hybridclim` (`hybridclim.maps`) builds the rasters in six steps: (1) bilinear interpolation
  of CHELSA with an elevation-residual correction,
  $X_{30\,m} = \mathrm{bilin}(X_{1\,km}) + \beta\,(z - \mathrm{bilin}(\bar z_{1\,km}))$;
  (2) temperature deltas and precipitation change factors between 1981–2010 and the target
  period, spread by thin-plate splines whose smoothing (or a constant) is selected by
  leave-one-out cross-validation; (3) the mean annual maximum 24-hour rainfall
  $P_{24} = f_{24}\, r\, P$, where $f_{24}$ converts calendar-day to sliding 24-hour maxima
  and is estimated from hourly data (fallback 1.13, @wmo2009), followed by a pooled GEV growth
  curve with year-block bootstrap intervals; (4) optional station correction; (5) station
  validation; and (6) diagnostics.
- `hybridclim.rfa` is a dependency-light implementation of regional frequency analysis:
  sample L-moments, GEV growth curves, the four-parameter kappa distribution [@hosking1994],
  discordancy $D_i$, heterogeneity $H_1$ and $H_2$ by kappa simulation, goodness-of-fit $Z$
  for the GEV, GLO, GNO, PE3 and GPA distributions, and mean intersite correlation.
- `hybridclim.validation` reads station climatologies and annual-maximum series and computes ME, MAE,
  RMSE, PBIAS and correlation per product.

The two analysis modules have no GIS dependencies and are covered by unit tests that check
L-moments against population values, the kappa distribution against its GEV, GLO and GPA
special cases, the discordancy identity $\sum_i D_i = N$, and the behaviour of $H_1$ on
simulated homogeneous and heterogeneous regions. The regional frequency routines are also
tested against Hosking's own R implementation [`lmomRFA`, @lmomRFA] on the Cascades region of
@hoskingwallis1997: discordancy and kappa parameters agree to within $10^{-5}$, the GEV growth
curve to 0.1 %, and $H$ and $Z$ within Monte Carlo error. An offline end-to-end test runs the
complete workflow on a bundled 1.5 MB fixture with network access disabled and compares the
results with a stored snapshot; all tests run on every commit through GitHub Actions. Design choices that affect safety are
explicit and logged: a GEV shape bounded above falls back to Gumbel, a region with
$H_1 \ge 2$, a rejected GEV or discordant sites raise warnings, and design maps produced
without station correction carry a printed warning that they are lower bounds.

# Research impact statement

`HybridClim` is being used to prepare catchment-scale climate and design-rainfall inputs
for hydrological modelling studies in Türkiye. A companion research article, currently in
preparation, validates the hybrid temperature, precipitation and T-year design-rainfall
surfaces against the station network of the Turkish State Meteorological Service (MGM)
across contrasting climate regions. Because the workflow needs only open global data
(CHELSA, ERA5/ERA5-Land, Copernicus DEM) and an optional station table, it is aimed at
hydrologists and design engineers who must supply DEM-resolution climate inputs to models
such as HEC-HMS, HEC-RAS 2D and SWAT in regions where gauge data are sparse or restricted.

# AI usage disclosure

Parts of the code, the unit tests, the documentation and this manuscript were drafted or
revised with the assistance of Claude Opus 5.5 (Anthropic). The author designed the method,
reviewed, tested and edited all AI-assisted output, and takes full responsibility for the
content of the software and the paper.

# Acknowledgements

We acknowledge the providers of CHELSA, ERA5/ERA5-Land (Copernicus Climate Change Service),
the Copernicus DEM, and the Open-Meteo API. This work received no specific funding.

# References
