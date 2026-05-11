# -*- coding: utf-8 -*-
"""
Created on Fri May 8 2026
@name:   Option Variance Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from abc import ABC

from support.finance import Concepts, Alerting
from support.equations import Equations
from scipy.spatial import cKDTree

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["VarianceCalculator", "StandardCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class NeighborhoodCalculator(ABC):
    def __init__(self, *args, neighbors=25, **kwargs):
        super().__init__(*args, **kwargs)
        self.__neighbors = int(neighbors)

    def neighborhood(self, t, k, w):
        mad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x))) + 1e-12
        diff = lambda x: x - np.median(x)
        t = np.asarray(t, dtype=float)
        k = np.asarray(k, dtype=float)
        w = np.asarray(w, dtype=float)
        t = diff(t) / mad(t)
        k = diff(k) / mad(t)
        tk = np.column_stack([t, k])
        tree = cKDTree(tk)
        _, ij = tree.query(tk, k=self.neighbors)
        for index in range(len(w)):
            wij = w[ij[index]]
            yield 1.4826 * mad(wij)

    @property
    def neighbors(self): return self.__neighbors


class ScreeningCalculator(NeighborhoodCalculator):
    def __init__(self, *args, threshold=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.__threshold = int(threshold)

    def screen(self, options):
        tau = options["tau"].to_numpy(dtype=float)
        mae = options["mae"].to_numpy(dtype=float)
        tiv = options["tiv"].to_numpy(dtype=float)
        ntiv = self.neighborhood(tau, mae, tiv)
        ntiv = np.fromiter(ntiv, dtype=np.float64)
        mask = ntiv > self.threshold
        return options.loc[~mask]

    @property
    def threshold(self): return self.__threshold


class CleaningCalculator(ABC):
    @staticmethod
    def clean(options):
        mask = options["tau"].notna() & options["mae"].notna() & options["tiv"].notna()
        options = options[mask].dropna(how="all", inplace=False)
        return options


class VarianceCalculator(ScreeningCalculator, CleaningCalculator, Equations, Alerting):
    mae = lambda forward, strike, option: np.log(forward / strike) * option.astype(int)
    tiv = lambda implied, tau: tau * np.square(implied)

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        variance = self.execute(options, *args, **kwargs)
        variance = pd.concat([options, variance], axis=1)
        variance = self.clean(variance)
        variance = self.screen(variance)
        self.alert(variance, title="Calculated", instrument=Concepts.Securities.Instrument.OPTION)
        return variance


class StandardCalculator(NeighborhoodCalculator, CleaningCalculator, Alerting):
    def __call__(self, options, surface, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        options = self.clean(options)
        tau = options["tau"].to_numpy(dtype=float)
        mae = options["mae"].to_numpy(dtype=float)
        tiv = options["tiv"].to_numpy(dtype=float)
        standard = self.standard(tau, mae, tiv, surface)
        standard = pd.Series(standard, name="ziv")
        standard = pd.concat([options, standard], axis=1)
        standard = self.clean(standard)
        self.alert(standard, title="Calculated", instrument=Concepts.Securities.Instrument.OPTION)
        return standard

    def standard(self, t, k, w, f):
        μ = np.vectorize(f)(t, k)
        σ = self.neighborhood(t, k, w)
        σ = np.fromiter(σ, dtype=np.float64)
        ε = np.quantile(σ[σ > 0], 0.1) if np.any(σ > 0) else 1e-8
        z = (w - μ) / np.maximum(σ, ε)
        return z



