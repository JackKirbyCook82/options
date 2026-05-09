# -*- coding: utf-8 -*-
"""
Created on Fri May 8 2026
@name:   Option Variance Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd

from support.finance import Concepts, Alerting
from support.equations import Equations
from scipy.spatial import cKDTree

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["VarianceCalculator", "VarianceScreener", "VariationCalculator"]
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


class VarianceScreener(Alerting):
    def __init__(self, *args, neighbors=12, threshold=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.__neighbors = neighbors
        self.__threshold = threshold

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        mask = options["tau"].notna() & options["mae"].notna() & options["tiv"].notna()
        options = options[mask].dropna(how="all", inplace=False)
        if len(options) < max(3, self.neighbors + 1): return options
        tau = options["tau"].to_numpy(dtype=float)
        mae = options["mae"].to_numpy(dtype=float)
        tiv = options["tiv"].to_numpy(dtype=float)
        residuals = self.residuals(tau, mae, tiv)
        residuals = np.fromiter(residuals, dtype=float, count=len(options))
        mask = residuals > self.threshold
        options = options.loc[~mask]
        self.alert(options, title="Screened", instrument=Concepts.Securities.Instrument.OPTION)
        return options

    def residuals(self, tau, mae, tiv):
        tau = np.asarray(tau, dtype=float)
        mae = np.asarray(mae, dtype=float)
        tiv = np.asarray(tiv, dtype=float)
        tau = (tau - np.median(tau)) / self.deviation(tau)
        mae = (mae - np.median(mae)) / self.deviation(mae)
        xy = np.column_stack([tau, mae])
        tree = cKDTree(xy)
        _, ij = tree.query(xy, k=self.neighbors + 1)
        for index in range(len(tiv)):
            nbr = tiv[ij[index, 1:]]
            deviation = self.deviation(nbr)
            yield np.abs(tiv[index] - np.median(nbr)) / deviation

    @staticmethod
    def deviation(axis):
        axis = np.asarray(axis, dtype=float)
        return 1.4826 * np.median(np.abs(axis - np.median(axis))) + 1e-12

    @property
    def neighbors(self): return self.__neighbors
    @property
    def threshold(self): return self.__threshold


class VariationCalculator(Alerting):
    def __init__(self, *args, neighbors=25, **kwargs):
        super().__init__(*args, **kwargs)
        self.__neighbors = neighbors

    def __call__(self, options, surface, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        t = options["tau"].to_numpy(np.float64)
        k = options["mae"].to_numpy(np.float64)
        w = options["tiv"].to_numpy(np.float64)
        variation = self.variation(t, k, w, surface)
        variation = pd.Series(variation, name="zsr")
        variation = pd.concat([options, variation], axis=1)
        self.alert(variation, title="Calculated", instrument=Concepts.Securities.Instrument.OPTION)
        return variation

    def variation(self, t, k, w, f):
        n = self.neighbors
        μ = self.average(t, k, w, f)
        σ = self.deviation(t, k, w, n)
        σ = np.fromiter(σ, dtype=np.float64)
        ε = np.quantile(σ[σ > 0], 0.1) if np.any(σ > 0) else 1e-8
        z = (w - μ) / np.maximum(σ, ε)
        return z

    @staticmethod
    def average(t, k, w, f):
        μ = np.vectorize(f)(t, k)
        return w - μ

    @staticmethod
    def deviation(t, k, w, n):
        t = t / np.std(t)
        k = k / np.std(k)
        tk = np.column_stack([t, k])
        tree = cKDTree(tk)
        _, ij = tree.query(tk, k=n)
        for index in range(len(w)):
            wij = w[ij[index]]
            mad = np.median(np.abs(wij - np.median(wij)))
            yield 1.4826 * mad

    @property
    def neighbors(self): return self.__neighbors


