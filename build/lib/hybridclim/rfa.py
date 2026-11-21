"""
Regional frequency analysis (Hosking & Wallis, 1997): L-moments, GEV growth curves,
discordancy (D), heterogeneity (H) and goodness-of-fit (Z) measures.

References
  Hosking, J.R.M. (1990). L-moments. J. R. Stat. Soc. B 52, 105-124.
  Hosking, J.R.M. (1994). The four-parameter kappa distribution. IBM J. Res. Dev. 38, 251-258.
  Hosking, J.R.M., Wallis, J.R. (1997). Regional Frequency Analysis. Cambridge Univ. Press.

Input: one annual-maximum series per site (station or grid cell). Pure numpy/scipy, so the
module can be used and tested independently of the mapping workflow.
"""

import math

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import gamma as G, gammaln

EULER = 0.5772156649

# Critical values of the discordancy measure, Hosking & Wallis (1997), Table 3.1 (N < 15)
D_CRIT = {5: 1.333, 6: 1.648, 7: 1.917, 8: 2.140, 9: 2.329, 10: 2.491, 11: 2.632,
          12: 2.757, 13: 2.869, 14: 2.971}

# tau4 = sum_j A_j tau3^j polynomial approximations (Hosking & Wallis 1997, Appendix A);
# GEV and GLO use exact expressions
TAU4_POLY = {
    "GNO": [0.12282, 0, 0.77518, 0, 0.12279, 0, -0.13638, 0, 0.11368],
    "PE3": [0.12240, 0, 0.30115, 0, 0.95812, 0, -0.57488, 0, 0.19383],
    "GPA": [0, 0.20196, 0.95924, -0.20096, 0.04061],
}


# --- L-moments ----------------------------------------------------------------------------

def lmoments(x):
    """Sample L-moments (l1, l2, t, t3, t4); t = L-CV. Requires n >= 4."""
    x = np.sort(np.asarray(x, float))
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 4:
        raise ValueError(f"at least 4 values are needed for L-moments (n={n})")
    i = np.arange(1, n + 1)
    b0 = x.mean()
    b1 = np.sum((i - 1) / (n - 1) * x) / n
    b2 = np.sum((i - 1) * (i - 2) / ((n - 1) * (n - 2)) * x) / n
    b3 = np.sum((i - 1) * (i - 2) * (i - 3) / ((n - 1) * (n - 2) * (n - 3)) * x) / n
    l1, l2 = b0, 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    l4 = 20 * b3 - 30 * b2 + 12 * b1 - b0
    return l1, l2, l2 / l1, l3 / l2, l4 / l2


def site_table(series):
    """n, t, t3, t4 per series. series: {name: array} or DataFrame (one column per site)."""
    if isinstance(series, pd.DataFrame):
        series = {c: series[c].dropna().values for c in series.columns}
    rows = []
    for name, s in series.items():
        s = np.asarray(s, float)
        s = s[np.isfinite(s)]
        if len(s) < 4:
            continue
        _, _, t, t3, t4 = lmoments(s)
        rows.append(dict(name=name, n=len(s), t=t, t3=t3, t4=t4))
    return pd.DataFrame(rows)


def regional_average(st):
    """Record-length weighted regional L-moment ratios (t, t3, t4)."""
    w = st["n"].values / st["n"].sum()
    return tuple(float(np.sum(w * st[c].values)) for c in ("t", "t3", "t4"))


# --- Distributions ------------------------------------------------------------------------

def gev_k_from_t3(t3):
    """Hosking et al. (1985) approximation; |error| < 9e-4 for -0.5 < tau3 < 0.5."""
    c = 2 / (3 + t3) - math.log(2) / math.log(3)
    return 7.8590 * c + 2.9554 * c ** 2


def gev_tau4(k):
    if abs(k) < 1e-8:
        return 0.1504                                    # Gumbel
    a, b, c = 2.0 ** -k, 3.0 ** -k, 4.0 ** -k
    return (5 * (1 - c) - 10 * (1 - b) + 6 * (1 - a)) / (1 - a)


def tau4_of(dist, t3):
    """L-kurtosis of a distribution with L-skewness t3 (used by the Z statistic)."""
    if dist == "GEV":
        return gev_tau4(gev_k_from_t3(t3))
    if dist == "GLO":
        return (1 + 5 * t3 ** 2) / 6
    return float(np.polyval(TAU4_POLY[dist][::-1], t3))


def gev_growth(t2, t3, T, cap_k=False):
    """Mean-one GEV growth curve (Hosking L-moment estimator).

    cap_k: a shape k > 0 (tail bounded above) is unsafe for design, so fall back to Gumbel (k = 0).
    Returns (x_T, k).
    """
    T = np.asarray(T, float)
    k = gev_k_from_t3(t3)
    if cap_k and k > 0:
        k = 0.0
    y = -np.log(1 - 1 / T)
    if abs(k) < 1e-6:                                   # Gumbel
        a = t2 / math.log(2)
        xi = 1 - EULER * a
        return xi - a * np.log(y), k
    a = t2 * k / ((1 - 2 ** -k) * G(1 + k))
    xi = 1 - a * (1 - G(1 + k)) / k
    return xi + a / k * (1 - y ** k), k


# Four-parameter kappa: x(F) = xi + alpha/k * [1 - ((1 - F^h)/h)^k]

def _kappa_g(k, h, r):
    """Hosking (1994) g_r; nan outside the valid region."""
    r = np.asarray(r, float)
    if k <= -1:
        return np.full(r.shape, np.nan)
    if abs(h) < 1e-8:
        return np.exp(gammaln(1 + k)) * r ** -k
    if h > 0:
        lg = np.log(r) + gammaln(1 + k) + gammaln(r / h) - (1 + k) * np.log(h) - gammaln(1 + k + r / h)
        return np.exp(lg)
    a = -k - r / h
    if np.any(a <= 0):
        return np.full(r.shape, np.nan)
    lg = np.log(r) + gammaln(1 + k) + gammaln(a) - (1 + k) * np.log(-h) - gammaln(1 - r / h)
    return np.exp(lg)


def kappa_tau(k, h):
    """(tau3, tau4) of the kappa distribution, with (g1, g2)."""
    g1, g2, g3, g4 = _kappa_g(k, h, [1, 2, 3, 4])
    d = g1 - g2
    return (-g1 + 3 * g2 - 2 * g3) / d, (g1 - 6 * g2 + 10 * g3 - 5 * g4) / d, (g1, g2)


def fit_kappa(t, t3, t4):
    """Mean-one kappa (xi, alpha, k, h) with L-CV t. Falls back to GLO (h = -1) when tau4 lies
    above the GLO line, as recommended by Hosking & Wallis (1997)."""
    def res(p):
        a, b, _ = kappa_tau(*p)
        return [a - t3, b - t4] if np.isfinite(a) and np.isfinite(b) else [1e3, 1e3]
    best = None
    k0 = gev_k_from_t3(t3)
    for h0 in (0.01, 0.3, -0.3, 1.0, -0.8):
        sol = least_squares(res, [k0 if abs(k0) > 1e-3 else 1e-3, h0],
                            bounds=([-0.999, -1.0], [10.0, 10.0]))
        if best is None or sol.cost < best.cost:
            best = sol
    if best.cost < 1e-10:
        k, h = best.x
        kind = "kappa"
    else:
        k, h, kind = -t3, -1.0, "GLO"
    if abs(k) < 1e-6:
        k = 1e-6
    _, _, (g1, g2) = kappa_tau(k, h)
    alpha = t * k / (g1 - g2)
    xi = 1 - alpha * (1 - g1) / k
    return dict(xi=float(xi), alpha=float(alpha), k=float(k), h=float(h), type=kind)


def kappa_quantile(F, p):
    F = np.asarray(F, float)
    k, h = p["k"], p["h"]
    y = -np.log(F) if abs(h) < 1e-8 else (1 - F ** h) / h
    return p["xi"] + p["alpha"] / k * (1 - y ** k)


# --- Hosking-Wallis measures --------------------------------------------------------------

def discordancy(st):
    """D_i (Hosking & Wallis 1997, eq. 3.3). Undefined (nan) for N <= 3."""
    U = st[["t", "t3", "t4"]].values
    N = len(U)
    if N <= 3:
        return np.full(N, np.nan), np.nan
    d = U - U.mean(axis=0)
    A = d.T @ d
    Ai = np.linalg.pinv(A)
    D = N / 3 * np.einsum("ij,jk,ik->i", d, Ai, d)
    crit = D_CRIT.get(N, 3.0) if N >= 5 else np.nan     # no critical value for N = 4
    return D, crit


def _v_stats(st, tR, t3R):
    w = st["n"].values
    V1 = math.sqrt(np.sum(w * (st["t"].values - tR) ** 2) / w.sum())
    V2 = np.sum(w * np.sqrt((st["t"].values - tR) ** 2 + (st["t3"].values - t3R) ** 2)) / w.sum()
    return V1, V2


def heterogeneity(st, n_sim=500, seed=42):
    """H1 (L-CV), H2 (L-CV + L-skewness) and goodness-of-fit Z (GEV, GLO, GNO, PE3, GPA).

    Simulation: independent samples from a kappa distribution fitted to the regional
    L-moments, keeping each site's record length. Intersite correlation biases H low.
    """
    tR, t3R, t4R = regional_average(st)
    kap = fit_kappa(tR, t3R, t4R)
    rng = np.random.default_rng(seed)
    n = st["n"].values.astype(int)
    V = np.empty((n_sim, 2))
    t4_sim = np.empty(n_sim)
    for m in range(n_sim):
        sims = [kappa_quantile(rng.uniform(size=ni), kap) for ni in n]
        s = pd.DataFrame([dict(n=ni, **dict(zip(("t", "t3", "t4"), lmoments(x)[2:])))
                          for ni, x in zip(n, sims)])
        tr_, t3r_, t4r_ = regional_average(s)
        V[m] = _v_stats(s, tr_, t3r_)
        t4_sim[m] = t4r_
    Vobs = _v_stats(st, tR, t3R)
    H = (np.array(Vobs) - V.mean(axis=0)) / V.std(axis=0, ddof=1)
    B4 = float(np.mean(t4_sim - t4R))
    s4 = float(np.sqrt((np.sum((t4_sim - t4R) ** 2) - n_sim * B4 ** 2) / (n_sim - 1)))
    Z = {d: float((tau4_of(d, t3R) - t4R + B4) / s4) for d in ("GEV", "GLO", "GNO", "PE3", "GPA")}
    return dict(t_R=tR, t3_R=t3R, t4_R=t4R, H1=float(H[0]), H2=float(H[1]),
                H1_class=h_class(H[0]), kappa=kap, n_sim=n_sim, Z=Z,
                V_obs=[float(v) for v in Vobs], V_sim_mean=V.mean(axis=0).tolist(),
                V_sim_sd=V.std(axis=0, ddof=1).tolist(),
                tau4_fit={d: float(tau4_of(d, t3R)) for d in Z},
                Z_accepted=[d for d, z in Z.items() if abs(z) <= 1.64])


def h_class(H):
    """Hosking & Wallis (1997, sec. 4.3). H < -2 is not evidence of homogeneity: it points to
    intersite correlation or less dispersion than expected."""
    if H < -2:
        return "negative H: intersite correlation likely (not evidence of homogeneity)"
    return "acceptably homogeneous" if H < 1 else ("possibly heterogeneous" if H < 2
                                                   else "definitely heterogeneous")


def mean_intersite_corr(wide):
    """Mean intersite correlation over common years (rho-bar)."""
    c = wide.corr(min_periods=10).values
    iu = np.triu_indices_from(c, 1)
    v = c[iu]
    return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan


# --- Regional growth curve ----------------------------------------------------------------

def regional_growth(wide, T, n_boot=1000, seed=42, cap_k=True, min_n=5):
    """Index-flood growth curve with a year-block bootstrap 90 % interval.

    wide: rows = years, columns = sites (missing years may be NaN).
    Returns (table, k, k_raw).
    """
    def fit(a, cap):
        st = site_table({c: a[c].dropna().values for c in a.columns
                         if a[c].notna().sum() >= min_n})
        if st.empty:
            return np.full(len(T), np.nan), np.nan
        tR, t3R, _ = regional_average(st)
        return gev_growth(tR, t3R, T, cap_k=cap)
    k_raw = fit(wide, False)[1]
    xT, k = fit(wide, cap_k)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(wide), len(wide))
        boots.append(fit(wide.iloc[idx].reset_index(drop=True), cap_k)[0])
    boots = np.array(boots)
    tab = pd.DataFrame({"T_years": list(T), "x_T": xT,
                        "x_T_lo5": np.nanpercentile(boots, 5, axis=0),
                        "x_T_hi95": np.nanpercentile(boots, 95, axis=0)})
    return tab, float(k), float(k_raw)


def hw_report(wide, n_sim=500, seed=42):
    """Full Hosking-Wallis report for one region: per-site table and summary dict."""
    st = site_table(wide)
    D, crit = discordancy(st)
    st["D"] = D
    st["discordant"] = D > crit if np.isfinite(crit) else False
    rep = heterogeneity(st, n_sim=n_sim, seed=seed) if len(st) >= 2 else {}
    rep.update(N=len(st), D_critical=crit, discordant=st.loc[st["discordant"], "name"].tolist(),
               rho_mean=mean_intersite_corr(wide))
    return st, rep
