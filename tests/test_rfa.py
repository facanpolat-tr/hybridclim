"""Unit tests for regional frequency analysis:  python -m unittest discover -s tests"""

import math
import unittest

import numpy as np
import pandas as pd

from hybridclim import rfa

RP = [2, 5, 10, 25, 50, 100, 500]


class LMoments(unittest.TestCase):
    def test_gumbel_population_ratios(self):
        x = np.random.default_rng(0).gumbel(size=400_000)
        _, l2, _, t3, t4 = rfa.lmoments(x)
        self.assertAlmostEqual(l2, math.log(2), delta=3e-3)     # lambda2 = alpha ln 2
        self.assertAlmostEqual(t3, 0.1699, delta=3e-3)
        self.assertAlmostEqual(t4, 0.1504, delta=3e-3)

    def test_uniform_exact(self):
        # Uniform distribution: tau3 = 0, tau4 = 0; sample L-moments are unbiased
        x = np.linspace(0, 1, 1001)
        _, l2, _, t3, t4 = rfa.lmoments(x)
        self.assertAlmostEqual(l2, 1 / 6, places=3)
        self.assertAlmostEqual(t3, 0, places=6)
        self.assertAlmostEqual(t4, 0, places=3)

    def test_too_short(self):
        with self.assertRaises(ValueError):
            rfa.lmoments([1, 2, 3])


class Distributions(unittest.TestCase):
    def test_kappa_reduces_to_gev_glo_gpa(self):
        for k in (-0.2, 0.05, 0.2):
            t3, t4, _ = rfa.kappa_tau(k, 0.0)
            self.assertAlmostEqual(t4, rfa.gev_tau4(k), places=9)
        k = -0.15
        t3, t4, _ = rfa.kappa_tau(k, -1.0)                     # GLO: tau3 = -k
        self.assertAlmostEqual(t3, -k, places=9)
        self.assertAlmostEqual(t4, (1 + 5 * k * k) / 6, places=9)
        k = 0.1
        t3, t4, _ = rfa.kappa_tau(k, 1.0)                      # GPA
        self.assertAlmostEqual(t3, (1 - k) / (3 + k), places=9)
        self.assertAlmostEqual(t4, (1 - k) * (2 - k) / ((3 + k) * (4 + k)), places=9)

    def test_tau4_polynomials(self):
        self.assertAlmostEqual(rfa.tau4_of("GNO", 0.0), 0.1226, delta=5e-4)   # normal
        self.assertAlmostEqual(rfa.tau4_of("PE3", 0.0), 0.1226, delta=5e-4)
        t3 = (1 - 0.1) / 3.1
        self.assertAlmostEqual(rfa.tau4_of("GPA", t3), 0.1345, delta=2e-3)
        self.assertAlmostEqual(rfa.tau4_of("GEV", 0.1699), 0.1504, delta=1e-3)

    def test_fit_kappa_roundtrip(self):
        p = rfa.fit_kappa(0.2, 0.15, 0.16)
        self.assertEqual(p["type"], "kappa")
        x = rfa.kappa_quantile(np.random.default_rng(1).uniform(size=300_000), p)
        l1, _, t, t3, t4 = rfa.lmoments(x)
        for a, b in ((l1, 1), (t, 0.2), (t3, 0.15), (t4, 0.16)):
            self.assertAlmostEqual(a, b, delta=4e-3)

    def test_fit_kappa_falls_back_to_glo(self):
        p = rfa.fit_kappa(0.2, 0.1, 0.40)                      # tau4 far above the GLO line
        self.assertEqual(p["type"], "GLO")

    def test_gumbel_growth_curve(self):
        t = 0.2
        xT, k = rfa.gev_growth(t, 0.1699, RP, cap_k=True)
        self.assertEqual(k, 0.0)                               # k ~ +1e-4 -> falls back to Gumbel
        a = t / math.log(2)
        exp = 1 - rfa.EULER * a - a * np.log(-np.log(1 - 1 / np.array(RP, float)))
        np.testing.assert_allclose(xT, exp, rtol=1e-9)

    def test_gev_growth_mean_one(self):
        xT, k = rfa.gev_growth(0.25, 0.25, [1 / (1 - F) for F in np.linspace(0.001, 0.999, 20001)])
        self.assertAlmostEqual(float(np.mean(xT)), 1.0, delta=0.01)   # integral of x(F) dF = 1


class HoskingWallis(unittest.TestCase):
    @staticmethod
    def region(cvs, n=40, seed=0):
        rng = np.random.default_rng(seed)
        wide = {}
        for i, cv in enumerate(cvs):
            a = cv / math.log(2)                               # Gumbel with mean 1 and L-CV cv
            wide[f"s{i}"] = 1 - rfa.EULER * a + a * rng.gumbel(size=n)
        return pd.DataFrame(wide)

    def test_discordancy_sums_to_N(self):
        st = rfa.site_table(self.region([0.2] * 12))
        D, crit = rfa.discordancy(st)
        self.assertAlmostEqual(D.sum(), len(st), places=6)
        self.assertEqual(crit, 2.757)

    def test_discordancy_small_region(self):
        D, crit = rfa.discordancy(rfa.site_table(self.region([0.2] * 3)))
        self.assertTrue(np.isnan(D).all() and np.isnan(crit))

    def test_homogeneous_vs_heterogeneous(self):
        hom = rfa.heterogeneity(rfa.site_table(self.region([0.2] * 12, seed=3)), n_sim=200)
        het = rfa.heterogeneity(rfa.site_table(self.region(np.linspace(0.08, 0.40, 12), seed=3)),
                                n_sim=200)
        self.assertLess(hom["H1"], 1.0)
        self.assertGreater(het["H1"], 2.0)
        self.assertIn("GEV", hom["Z_accepted"])                # Gumbel samples should not reject GEV

    def test_regional_growth_matches_unweighted_mean(self):
        """With equal record lengths the weighted regional average equals the plain mean."""
        wide = self.region([0.18, 0.22, 0.2, 0.25, 0.19], n=30, seed=5) * 50
        tab, k, _ = rfa.regional_growth(wide, RP, n_boot=50)
        norm = wide / wide.mean()
        rat = np.array([rfa.lmoments(norm[c])[2:4] for c in norm])
        ref, _ = rfa.gev_growth(rat[:, 0].mean(), rat[:, 1].mean(), RP, cap_k=True)
        np.testing.assert_allclose(tab.x_T.values, ref, rtol=1e-12)
        self.assertTrue((tab.x_T_lo5 <= tab.x_T_hi95).all())

    def test_h_class(self):
        self.assertTrue(rfa.h_class(-2.5).startswith("negative H"))
        self.assertEqual(rfa.h_class(0.5), "acceptably homogeneous")
        self.assertEqual(rfa.h_class(1.5), "possibly heterogeneous")
        self.assertEqual(rfa.h_class(2.5), "definitely heterogeneous")

    def test_hw_report_keys(self):
        st, rep = rfa.hw_report(self.region([0.2] * 8, seed=2), n_sim=100)
        for key in ("H1", "H2", "Z", "D_critical", "rho_mean", "kappa"):
            self.assertIn(key, rep)
        self.assertEqual(len(st), 8)


if __name__ == "__main__":
    unittest.main()
