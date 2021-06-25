from unittest import TestCase
import unittest

import numpy as np
from denseweight import DenseWeight


class BasicTest(TestCase):
    def test_does_it_work(self):
        eps = 1e-6
        y = np.random.normal(size=1000)
        dw = DenseWeight(eps=eps)
        weights = dw.fit(y)
        w = dw([0.0])
        self.assertTrue(len(weights) == len(y))
        self.assertIsInstance(weights[0], float)
        self.assertGreaterEqual(weights[0], eps)
        self.assertIsInstance(w[0], float)
        self.assertGreaterEqual(w[0], eps)

    def test_alpha_0(self):
        eps = 1e-6
        alpha = 0.0
        y = np.random.normal(size=1000)
        dw = DenseWeight(alpha=alpha, eps=eps)
        dw.fit(y)
        weights = dw([-1.0, -0.5, 0.0, 0.5, 1.0])
        self.assertTrue([w == 1.0 for w in weights])

    def test_forgot_fitting(self):
        dw = DenseWeight()
        y = [1.0]
        self.assertRaises(ValueError, dw.eval, y)
        self.assertRaises(ValueError, dw.eval_single, 1.0)
        self.assertRaises(ValueError, dw.get_density, y)


if __name__ == '__main__':
    unittest.main()
