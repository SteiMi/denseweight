#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for DenseWeight.
"""
import functools
from typing import Optional
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from KDEpy import FFTKDE
from denseweight.utils import bisection


class DenseWeight:
    r"""
    This class implements DenseWeight.
    Parameters
    ----------
    y : array-like
        The target values based on which DenseWeight calculates weights.
    bw : float or str
        Bandwidth or bandwidth selection method. If a float is passed, it
        is the standard deviation of the kernel. If a string it passed, it
        is the bandwidth selection method, see cls._bw_methods.keys() for
        choices.
    Examples
    --------
    >>> data = np.random.randn(2**10)
    >>> # (1) Automatic bw selection using Improved Sheather Jones (ISJ)
    >>> x, y = FFTKDE(bw='ISJ').fit(data).evaluate()
    >>> # (2) Explicit choice of kernel and bw (standard deviation of kernel)
    >>> x, y = FFTKDE(kernel='triweight', bw=0.5).fit(data).evaluate()
    >>> weights = data + 10
    >>> # (3) Using a grid and weights for the data
    >>> y = FFTKDE(kernel='epa', bw=0.5).fit(data, weights).evaluate(x)
    >>> # (4) If you supply your own grid, it must be equidistant
    >>> y = FFTKDE().fit(data)(np.linspace(-10, 10, num=2**12))
    References
    ----------
    - TBA
    """
    def __init__(self, y, alpha: float = 1.0, bandwidth: Optional[float] = None):
        self.alpha = alpha
        print('DenseWeight alpha:', self.alpha)

        if bandwidth:
            bandwidth_used = bandwidth

        else:
            silverman_bandwidth = 1.06 * np.std(y) * np.power(len(y), (-1.0 / 5.0))

            print('Using Silverman Bandwidth', silverman_bandwidth)
            bandwidth_used = silverman_bandwidth

        self.kernel = FFTKDE(bw=bandwidth_used).fit(y, weights=None)

        x, y_dens_grid = self.kernel.evaluate(4096)  # Default precision is 1024
        self.x = x

        # Min-Max Scale to 0-1 since pdf's can actually exceed 1
        # See: https://stats.stackexchange.com/questions/5819/kernel-density-estimate-takes-values-larger-than-1
        self.y_dens_grid = (
            MinMaxScaler().fit_transform(y_dens_grid.reshape(-1, 1)).flatten()
        )

        self.y_dens = np.vectorize(self.get_density)(y)

        self.eps = 1e-6
        w_star = np.maximum(1 - self.alpha * self.y_dens, self.eps)
        self.mean_w_star = np.mean(w_star)
        self.relevances = w_star / self.mean_w_star

    def get_density(self, y):
        idx = bisection(self.x, y)
        try:
            dens = self.y_dens_grid[idx]
        except IndexError:
            if idx <= -1:
                idx = 0
            elif idx >= len(self.x):
                idx = len(self.x) - 1
            dens = self.y_dens_grid[idx]
        return dens

    @functools.lru_cache(maxsize=100000)
    def eval_single(self, y):
        dens = self.get_density(y)
        return np.maximum(1 - self.alpha * dens, self.eps) / self.mean_w_star

    def eval(self, y):
        ys = y.flatten().tolist()
        rels = np.array(list(map(self.eval_single, ys)))[:, None]
        return rels

    def __call__(self, y):
        return self.eval(y)
