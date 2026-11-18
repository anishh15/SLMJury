"""Smoke tests for slmjury.analysis.stats on small, hand-checkable inputs."""

import math
import unittest

from slmjury.analysis import stats


class TestMcNemar(unittest.TestCase):

    def test_exact_p(self):
        # 17 discordant pairs one way, 3 the other:
        # p = 2 * (C(20,0) + C(20,1) + C(20,2) + C(20,3)) / 2**20 = 2702 / 2**20
        a = [1] * 17 + [0] * 3 + [1] * 5
        b = [0] * 17 + [1] * 3 + [1] * 5
        r = stats.mcnemar(a, b)
        self.assertEqual((r.b, r.c, r.n_discordant, r.n_pairs), (17, 3, 20, 25))
        self.assertAlmostEqual(r.p_exact, 2702 / 2 ** 20, places=15)

    def test_no_disagreement(self):
        r = stats.mcnemar([1, 0, 1], [1, 0, 1])
        self.assertEqual(r.p_exact, 1.0)
        self.assertTrue(math.isnan(r.odds_ratio))

    def test_unpaired_inputs_raise(self):
        with self.assertRaises(ValueError):
            stats.mcnemar([1, 0], [1])


class TestWilson(unittest.TestCase):

    def test_known_interval(self):
        lo, hi = stats.wilson_interval(8, 10)
        self.assertAlmostEqual(lo, 49.016247, places=5)
        self.assertAlmostEqual(hi, 94.331785, places=5)

    def test_empty(self):
        lo, hi = stats.wilson_interval(0, 0)
        self.assertTrue(math.isnan(lo) and math.isnan(hi))


class TestBenjaminiHochberg(unittest.TestCase):

    def test_qvalues(self):
        out = stats.benjamini_hochberg([0.01, 0.04, 0.03, 0.5])
        expected = [0.04, 0.16 / 3, 0.16 / 3, 0.5]
        for q, e in zip(out["qvalues"], expected):
            self.assertAlmostEqual(q, e, places=12)
        self.assertEqual(out["reject"], [True, False, False, False])
        self.assertEqual(out["n_reject"], 1)

    def test_empty(self):
        self.assertEqual(stats.benjamini_hochberg([])["n"], 0)


class TestFullPrecision(unittest.TestCase):

    def test_rows_keep_full_precision(self):
        a = [1] * 60 + [0] * 2
        b = [0] * 60 + [1] * 2
        row = stats.mcnemar(a, b).as_row()
        self.assertEqual(row["mcnemar_p_exact"], stats.mcnemar(a, b).p_exact)
        self.assertGreater(row["mcnemar_p_exact"], 0.0)
        self.assertLess(row["mcnemar_p_exact"], 1e-12)
        boot = stats.cluster_bootstrap_diff([1, 0, 1, 1, 0, 1], [0, 0, 1, 0, 1, 1], ["x"] * 6, n_boot=200)
        self.assertEqual(boot.as_row()["diff_pp"], boot.point)

    def test_bh_orders_tiny_p_values_that_rounding_would_tie(self):
        q = stats.benjamini_hochberg([3e-9, 1e-7, 0.03])["qvalues"]
        self.assertAlmostEqual(q[0], 9e-9, places=20)
        self.assertAlmostEqual(q[1], 1.5e-7, places=20)
        self.assertLess(q[0], q[1])


if __name__ == "__main__":
    unittest.main()
