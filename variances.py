# -*- coding: utf-8 -*-
"""
Created on Fri May 8 2026
@name:   Option Variance Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from abc import ABC
from scipy.spatial import cKDTree

from finance.variables import Enumerations
from finance.logging import Logging
from support.equations import Equations

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["VarianceCalculator", "StandardizingCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class NeighborhoodCalculator(Logging, ABC):
    def __init__(self, *args, neighbors=25, **kwargs):
        super().__init__(*args, **kwargs)
        self.__neighbors = int(neighbors)

    def neighborhood(self, t, k, w):
        t = self.independent(t)
        k = self.independent(k)
        w = np.asarray(w, dtype=float)
        tk = np.column_stack([t, k])
        tree = cKDTree(tk)
        n = min(self.neighbors, len(w))
        _, ij = tree.query(tk, k=n)
        if n == 1: ij = ij[:, None]
        for index in range(len(w)):
            wij = w[ij[index]]
            yield self.dependent(wij)

    @staticmethod
    def independent(x):
        x = np.asarray(x, dtype=float)
        center = np.median(x)
        distance = np.abs(x - center)
        scale = 1.4826 * np.median(distance)
        return (x - center) / max(scale, 1e-12)

    @staticmethod
    def dependent(y):
        x = np.asarray(y, dtype=float)
        center = np.median(x)
        distance = np.abs(x - center)
        scale = 1.4826 * np.median(distance)
        return scale + 1e-12

    @property
    def neighbors(self): return self.__neighbors


class ScreeningError(Exception): pass
class ScreeningCalculator(NeighborhoodCalculator):
    def __init__(self, *args, quantile=0.95, multiple=2.5, **kwargs):
        assert (0.0 < quantile < 1.0) and (multiple > 1.0)
        super().__init__(*args, **kwargs)
        self.__quantile = float(quantile)
        self.__multiple = float(multiple)

    def screen(self, variance):
        tau = variance["tau"].to_numpy(dtype=float)
        mae = variance["mae"].to_numpy(dtype=float)
        tiv = variance["tiv"].to_numpy(dtype=float)
        ntiv = self.neighborhood(tau, mae, tiv)
        ntiv = np.fromiter(ntiv, dtype=np.float64)
        valid = np.isfinite(ntiv) & (ntiv > 0)
        if not valid.any(): raise ScreeningError()
        ntiv = ntiv[valid]
        quantile = np.quantile(ntiv, self.quantile)
        median = np.median(ntiv)
        mask = ntiv > max(quantile, self.multiple * median)
        return variance.loc[~mask]

    @property
    def quantile(self): return self.__quantile
    @property
    def multiple(self): return self.__multiple


class CleaningCalculator(ABC):
    @staticmethod
    def clean(options):
        mask = options["tau"].notna() & options["mae"].notna() & options["tiv"].notna()
        options = options[mask].dropna(how="all", inplace=False)
        return options


class VarianceCalculator(ScreeningCalculator, CleaningCalculator, Equations):
    mae = lambda forward, strike, option: np.log(forward / strike.astype(float)) * option.astype(int)
    tiv = lambda implied, tau: tau * np.square(implied)

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        variance = self.execute(options, *args, **kwargs)
        options = pd.concat([options, variance], axis=1)
        options = self.clean(options)
        options = self.screen(options)
        self.results(options, title="Calculated", instrument=Enumerations.Instrument.OPTION)
        return options


class StandardizingCalculator(NeighborhoodCalculator):
    def __call__(self, options, surface, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        tau = options["tau"].to_numpy(dtype=float)
        mae = options["mae"].to_numpy(dtype=float)
        tiv = options["tiv"].to_numpy(dtype=float)
        standard = self.standard(tau, mae, tiv, surface)
        standard = pd.Series(standard, name="zscore", index=options.index)
        options = pd.concat([options, standard], axis=1)
        self.results(options, title="Calculated", instrument=Enumerations.Instrument.OPTION)
        return options

    def standard(self, t, k, w, f):
        μ = np.vectorize(f.z)(t, k)
        σ = self.neighborhood(t, k, w)
        σ = np.fromiter(σ, dtype=np.float64)
        ε = np.quantile(σ[σ > 0], 0.1) if np.any(σ > 0) else 1e-8
        z = (w - μ) / np.maximum(σ, ε)
        return z



