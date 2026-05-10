# -*- coding: utf-8 -*-
"""
Created on Fri May 8 2026
@name:   Option Variance Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from support.finance import Concepts, Alerting
from support.equations import Equations
from scipy.spatial import cKDTree

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["VarianceCalculator", "ExclusionCalculator", "InclusionCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class VarianceCalculator(Equations, Alerting):
    mae = lambda forward, strike, option: np.log(forward / strike) * option.astype(int)
    tiv = lambda implied, tau: tau * np.square(implied)

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        variance = self.execute(options, *args, **kwargs)
        variance = pd.concat([options, variance], axis=1)
        self.alert(variance, title="Calculated", instrument=Concepts.Securities.Instrument.OPTION)
        return variance


class NeighborhoodCalculator(Alerting, ABC):
    def __init__(self, *args, neighbors=25, threshold=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.__neighbors = int(neighbors)
        self.__threshold = int(threshold)

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        mask = options["tau"].notna() & options["mae"].notna() & options["tiv"].notna()
        options = options[mask].dropna(how="all", inplace=False)
        options = self.execute(options, *args, **kwargs)
        self.alert(options, title="Calculated", instrument=Concepts.Securities.Instrument.OPTION)
        return options

    @staticmethod
    def variance(t, k, w, n):
        mad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x))) + 1e-12
        diff = lambda x: x - np.median(x)
        t = np.asarray(t, dtype=float)
        k = np.asarray(k, dtype=float)
        w = np.asarray(w, dtype=float)
        t = diff(t) / mad(t)
        k = diff(k) / mad(t)
        tk = np.column_stack([t, k])
        tree = cKDTree(tk)
        _, ij = tree.query(tk, k=n)
        for index in range(len(w)):
            wij = w[ij[index]]
            yield 1.4826 * mad(wij)

    @abstractmethod
    def execute(self, options, *args, **kwargs): pass
    @abstractmethod
    def calculate(self, *args, **kwargs): pass

    @property
    def neighbors(self): return self.__neighbors
    @property
    def threshold(self): return self.__threshold


class ExclusionCalculator(NeighborhoodCalculator):
    def execute(self, options, *args, neighbors, **kwargs):
        tau = options["tau"].to_numpy(dtype=float)
        mae = options["mae"].to_numpy(dtype=float)
        tiv = options["tiv"].to_numpy(dtype=float)
        variance = self.calculate(tau, mae, tiv, neighbors)
        mask = variance > self.threshold
        return options.loc[~mask]

    def calculate(self, t, k, w, n):
        σ = self.variance(t, k, w, n)
        σ = np.fromiter(σ, dtype=np.float64)
        return σ


class InclusionCalculator(NeighborhoodCalculator):
    def execute(self, options, tau, mae, tiv, *args, surface, neighbors, **kwargs):
        tau = options["tau"].to_numpy(dtype=float)
        mae = options["mae"].to_numpy(dtype=float)
        tiv = options["tiv"].to_numpy(dtype=float)
        zscore = self.calculate(tau, mae, tiv, neighbors, surface)
        zscore = pd.Series(zscore, name="zsr")
        return pd.concat([options, zscore], axis=1)

    def calculate(self, t, k, w, n, f):
        μ = np.vectorize(f)(t, k)
        σ = self.variance(t, k, w, n)
        σ = np.fromiter(σ, dtype=np.float64)
        ε = np.quantile(σ[σ > 0], 0.1) if np.any(σ > 0) else 1e-8
        z = (w - μ) / np.maximum(σ, ε)
        return z



