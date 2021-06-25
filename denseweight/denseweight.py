#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for DenseWeight.
"""
import functools
from typing import Optional, Union
import numpy as np
from numpy.typing import ArrayLike
from sklearn.preprocessing import MinMaxScaler
from KDEpy import FFTKDE
from denseweight.utils import bisection


class DenseWeight:
    r"""
    This class implements DenseWeight.
    Parameters
    ----------
    alpha : float
        Alpha value for DenseWeight. Adjusts the intensity of density-based
        weighting.
    bandwidth : float or str
        Bandwidth or bandwidth selection method. If a float is passed, it
        is the standard deviation of the kernel. If a string it passed, it
        is the bandwidth selection method, see cls._bw_methods.keys() for
        choices.
    eps : float
        Epsilon for DenseWeight. Sets the minimum weight a data point can
        receive.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        bandwidth: Optional[Union[float, str]] = None,
        eps: float = 1e-6,
    ):
        self.alpha = alpha
        self.bandwidth = bandwidth
        self.eps = eps

    def fit(self, y: ArrayLike, grid_points=4096) -> np.ndarray:

        if self.bandwidth is None:
            silverman_bandwidth = 1.06 * np.std(y) * np.power(len(y), (-1.0 / 5.0))
            self.bandwidth = silverman_bandwidth

        self.kernel = FFTKDE(bw=self.bandwidth).fit(y, weights=None)

        x, y_dens_grid = self.kernel.evaluate(grid_points)  # Default precision is 1024
        self.x = x

        # Min-Max Scale to 0-1 since pdf's can actually exceed 1
        # See: https://stats.stackexchange.com/questions/5819/kernel-density-estimate-takes-values-larger-than-1
        self.y_dens_grid = (
            MinMaxScaler().fit_transform(y_dens_grid.reshape(-1, 1)).flatten()
        )

        self.y_dens = np.vectorize(self.get_density)(y)

        w_star = np.maximum(1 - self.alpha * self.y_dens, self.eps)
        self.mean_w_star = np.mean(w_star)
        self.weights = w_star / self.mean_w_star
        return self.weights

    def get_density(self, y: ArrayLike) -> np.ndarray:
        try:
            idx = bisection(self.x, y)
        except AttributeError:
            raise ValueError("Must call fit first!")
        try:
            dens = self.y_dens_grid[idx]
        except IndexError:
            if idx <= -1:
                idx = 0
            elif idx >= len(self.x):
                idx = len(self.x) - 1
            dens = self.y_dens_grid[idx]
        return dens

    # Cache 10000 values to speed things up
    @functools.lru_cache(maxsize=10000)
    def eval_single(self, y: float) -> float:
        dens = self.get_density(y)
        return np.maximum(1 - self.alpha * dens, self.eps) / self.mean_w_star

    def eval(self, y: ArrayLike) -> np.ndarray:
        np_y = np.array(y)
        y_l = np_y.flatten().tolist()

        rels = np.array(list(map(self.eval_single, y_l)))
        return rels

    def __call__(self, y: ArrayLike) -> np.ndarray:
        return self.eval(y)
