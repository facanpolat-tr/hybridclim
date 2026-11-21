"""
Station validation: comparison of the hybrid surfaces with independent observations.

Four products are compared at every station:
  ERA5 raw                 : nearest ERA5-Land (T) / ERA5 (P) cell, target period
                             (P24 = f24 * the cell's mean annual maximum daily precipitation)
  CHELSA raw (1981-2010)   : nearest 1 km CHELSA pixel (no P24); includes the period difference
  Hybrid                   : the 30 m surface built without stations
  Hybrid+stations (LOOCV)  : leave-one-out prediction of the station-corrected surface, i.e.
                             each station is read from a map corrected without its own data

Metrics: ME (mean error), MAE, RMSE, PBIAS (%, precipitation only), r (Pearson, n >= 3).
"""

import numpy as np
import pandas as pd

PRODUCTS = ["ERA5 raw", "CHELSA raw (1981-2010)", "Hybrid", "Hybrid+stations (LOOCV)"]
# key: (station CSV column, unit, multiplicative)
VARS = {"T": ("T_annual", "°C", False), "P": ("P_annual", "mm/yr", True),
        "P24": ("P24_mean_annual_max", "mm", True)}
TITLES = {"T": "Mean annual temperature", "P": "Annual precipitation",
          "P24": "Mean annual max 24 h precipitation"}


def read_stations(csv, ams_csv=None, target=None, p24_kind="calendar-day", f24=1.13, min_years=10):
    """Station CSV (+ optional annual-maximum series) -> station table and wide AMS table.

    p24_kind: 'calendar-day' -> P24_mean_annual_max and the AMS are calendar-day maxima and are
              converted to sliding 24 h with f24 (same definition as the maps); '24h' -> already
              sliding 24 h.
    AMS CSV columns: name, year, P_max (mm).
    """
    st = pd.read_csv(csv)
    for c in ("name", "lat", "lon"):
        if c not in st:
            raise SystemExit(f"Station CSV has no '{c}' column: {csv}")
    st["name"] = st["name"].astype(str)
    conv = f24 if p24_kind == "calendar-day" else 1.0
    wide = None
    if ams_csv:
        a = pd.read_csv(ams_csv)
        for c in ("name", "year", "P_max"):
            if c not in a:
                raise SystemExit(f"AMS CSV has no '{c}' column: {ams_csv}")
        a["name"] = a["name"].astype(str)
        unknown = sorted(set(a["name"]) - set(st["name"]))
        if unknown:
            raise SystemExit(f"AMS stations missing from the station CSV: {unknown}")
        wide = a.pivot_table(index="year", columns="name", values="P_max", aggfunc="max").sort_index()
        wide = wide.loc[:, wide.notna().sum() >= min_years]
        if target is not None:
            tw = wide.loc[(wide.index >= target[0]) & (wide.index <= target[1])]
            n_ok = tw.notna().sum()
            mean_t = tw.mean()[n_ok >= min_years]
            if "P24_mean_annual_max" not in st:
                st["P24_mean_annual_max"] = np.nan
            fill = st["P24_mean_annual_max"].isna() & st["name"].isin(mean_t.index)
            st.loc[fill, "P24_mean_annual_max"] = st.loc[fill, "name"].map(mean_t)
    if "P24_mean_annual_max" in st:
        st["P24_mean_annual_max"] = st["P24_mean_annual_max"] * conv
    return st, wide, conv


def _nearest(df, lat, lon):
    d = (df["lat"].values - lat) ** 2 + ((df["lon"].values - lon) * np.cos(np.radians(lat))) ** 2
    return df.iloc[int(np.argmin(d))]


def _pixel(arr, tr, x, y):
    c = int((x - tr.c) / tr.a)
    r = int((y - tr.f) / tr.e)
    if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]:
        return float(arr[r, c])
    return np.nan


def sample_products(st, tp, pp, chelsa, hybrid, tr, to_utm, f24):
    """Long table: name, variable, product, observed, predicted (LOOCV rows are added later)."""
    rows = []
    sx, sy = to_utm.transform(st["lon"].values, st["lat"].values)
    for i, s in st.reset_index(drop=True).iterrows():
        et, ep = _nearest(tp, s.lat, s.lon), _nearest(pp, s.lat, s.lon)
        raw = {"T": et.T_target, "P": ep.P_target, "P24": f24 * ep.Pmax_day_mean}
        for key, (col, _, _) in VARS.items():
            if col not in st or not np.isfinite(s.get(col, np.nan)):
                continue
            ch = np.nan
            if key in chelsa:
                arr, ctr = chelsa[key]
                ch = _pixel(arr, ctr, s.lon, s.lat)
            for product, v in (("ERA5 raw", raw[key]), ("CHELSA raw (1981-2010)", ch),
                               ("Hybrid", _pixel(hybrid[key], tr, sx[i], sy[i]))):
                if np.isfinite(v):
                    rows.append(dict(name=s["name"], variable=key, product=product,
                                     observed=float(s[col]), predicted=float(v)))
    return pd.DataFrame(rows)


def metrics(df):
    """Metrics table, one row per variable x product."""
    out = []
    for (var, product), g in df.groupby(["variable", "product"], sort=False):
        e = g.predicted.values - g.observed.values
        o = g.observed.values
        r = float(np.corrcoef(o, g.predicted.values)[0, 1]) if len(g) >= 3 and o.std() > 0 else np.nan
        out.append(dict(variable=var, product=product, n=len(g), ME=e.mean(),
                        MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()),
                        PBIAS_pct=(100 * e.sum() / o.sum()) if VARS[var][2] else np.nan, r=r))
    t = pd.DataFrame(out)
    t["product"] = pd.Categorical(t["product"], PRODUCTS, ordered=True)
    t["variable"] = pd.Categorical(t["variable"], list(VARS), ordered=True)
    return t.sort_values(["variable", "product"]).reset_index(drop=True)


def plot(df, out, plt):
    vars_ = [v for v in VARS if v in set(df.variable)]
    if not vars_:
        return
    colors = dict(zip(PRODUCTS, ["#9A9A9A", "#C9A227", "#2F6EBA", "#C0392B"]))
    fig, axes = plt.subplots(1, len(vars_), figsize=(4.4 * len(vars_), 4.4), dpi=170,
                             layout="constrained", squeeze=False)
    for ax, v in zip(axes[0], vars_):
        g = df[df.variable == v]
        for product in PRODUCTS:
            h = g[g["product"] == product]
            if len(h):
                rmse = np.sqrt(np.mean((h.predicted - h.observed) ** 2))
                ax.scatter(h.observed, h.predicted, s=28, color=colors[product], edgecolor="white",
                           label=f"{product} (RMSE {rmse:.3g})")
        lo = np.nanmin(g[["observed", "predicted"]].values)
        hi = np.nanmax(g[["observed", "predicted"]].values)
        pad = 0.05 * (hi - lo or 1)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="#999", lw=1)
        ax.set(xlabel=f"Station ({VARS[v][1]})", ylabel=f"Product ({VARS[v][1]})", title=TITLES[v])
        ax.grid(color="#E5E5E5"); ax.set_axisbelow(True)
        ax.legend(fontsize=6.5, frameon=False, loc="upper left")
        ax.title.set_fontsize(9.5)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
