"""
Reference test against Hosking's own implementation (R packages lmomRFA and lmom).

tests/data/lmomrfa_cascades.json was produced by tests/data/make_lmomrfa_reference.R for the
'Cascades' region of Hosking & Wallis (1997): 19 sites with record lengths and sample L-moment
ratios. Deterministic quantities (regional averages, D, V_obs, kappa and GEV parameters, growth
curve) must agree closely; simulated quantities (H, Z) agree within Monte Carlo error.
"""

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from hybridclim import rfa

REF = json.loads((Path(__file__).resolve().parent / "data" / "lmomrfa_cascades.json")
                 .read_text(encoding="utf-8"))
SITES = pd.DataFrame(REF["sites"])


class AgainstLmomRFA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.het = rfa.heterogeneity(SITES, n_sim=4000, seed=1)

    def test_regional_average(self):
        got = rfa.regional_average(SITES)
        for g, k in zip(got, ("t", "t3", "t4")):
            self.assertAlmostEqual(g, REF["regional"][k], places=9, msg=k)

    def test_discordancy(self):
        D, crit = rfa.discordancy(SITES)
        np.testing.assert_allclose(D, REF["D"], atol=1e-6)
        self.assertEqual(crit, REF["Dcrit"])

    def test_observed_dispersion(self):
        np.testing.assert_allclose(self.het["V_obs"], REF["vobs"], rtol=1e-8)

    def test_kappa_fit(self):
        r = REF["regional"]
        p = rfa.fit_kappa(r["t"], r["t3"], r["t4"])
        for k in ("xi", "alpha", "k", "h"):
            self.assertAlmostEqual(p[k], REF["kappa"][k], delta=1e-5, msg=k)

    def test_simulated_dispersion(self):
        # Mean and spread of V under the fitted kappa: Monte Carlo agreement within ~2 %
        np.testing.assert_allclose(self.het["V_sim_mean"], REF["vbar"], rtol=0.02)
        np.testing.assert_allclose(self.het["V_sim_sd"], REF["vsd"], rtol=0.05)

    def test_heterogeneity(self):
        self.assertAlmostEqual(self.het["H1"], REF["H"][0], delta=0.15)
        self.assertAlmostEqual(self.het["H2"], REF["H"][1], delta=0.15)

    def test_tau4_of_fitted_distributions(self):
        # GEV and GLO are exact; GNO, PE3 and GPA use the Hosking-Wallis polynomial approximations
        for d in ("GEV", "GLO", "GNO", "PE3", "GPA"):
            self.assertAlmostEqual(self.het["tau4_fit"][d], REF["t4fit"][d.lower()], delta=1e-3, msg=d)

    def test_goodness_of_fit(self):
        for d in ("GEV", "GLO", "GNO", "PE3", "GPA"):
            ref = REF["Z"][d.lower()]
            self.assertAlmostEqual(self.het["Z"][d], ref, delta=0.05 * abs(ref) + 0.15, msg=d)
        accepted = {d for d in ("GEV", "GLO", "GNO", "PE3", "GPA") if abs(REF["Z"][d.lower()]) <= 1.64}
        self.assertEqual(set(self.het["Z_accepted"]), accepted)

    def test_gev_growth_curve(self):
        r = REF["regional"]
        xT, k = rfa.gev_growth(r["t"], r["t3"], REF["gev_growth"]["T"], cap_k=False)
        self.assertAlmostEqual(k, REF["gev"]["k"], delta=1e-3)
        np.testing.assert_allclose(xT, REF["gev_growth"]["x_T"], rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
