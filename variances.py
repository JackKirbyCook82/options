# -*- coding: utf-8 -*-
"""
Created on Fri May 8 2026
@name:   Option Variance Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from dataclasses import dataclass
from datetime import date as Date

from finance.enumerations import Instrument
from finance.logging import Logging
from support.equations import Equations

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["VarianceCalculator", "VarianceScreener", "VarianceStandardizer"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True)
class Neighborhood:
    neighbors: int = 25

    def __call__(self, t, k, w):
        t = self.independent(t)
        k = self.independent(k)
        w = np.asarray(w, dtype=float)
        n = min(self.neighbors, len(w))
        tk = np.column_stack([t, k])
        tree = cKDTree(tk)
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
        y = np.asarray(y, dtype=float)
        center = np.median(y)
        distance = np.abs(y - center)
        scale = 1.4826 * np.median(distance)
        return scale + 1e-12


class VarianceCalculator(Logging, Equations):
    tau = lambda expire: (pd.to_datetime(expire) - pd.Timestamp(Date.today())).dt.days / 365
    mae = lambda forward, strike, option: np.log(forward / strike.astype(float)) * option.astype(int)
    tiv = lambda implied, tau: tau * np.square(implied)

    def __call__(self, options, /, **kwargs):
        assert isinstance(options, pd.DataFrame)
        variance = self.execute(options, **kwargs)
        options = pd.concat([options, variance], axis=1)
        self.results(options, title="Calculated", instrument=Instrument.OPTION)
        return options


class VarianceError(Exception): pass
class VarianceScreener(Logging):
    def __init__(self, *args, neighbors=25, quantile=0.95, multiple=2.5, **kwargs):
        assert (0.0 < quantile < 1.0) and (multiple > 1.0)
        super().__init__(*args, **kwargs)
        self.__neighborhood = Neighborhood(neighbors)
        self.__quantile = float(quantile)
        self.__multiple = float(multiple)

    def __call__(self, options, /, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if bool(options.empty): return options
        mask = options["tau"].notna() & options["mae"].notna() & options["tiv"].notna()
        options = options[mask].dropna(how="all", inplace=False)
        previous = len(options.index)
        options = self.screener(options)
        post = len(options.index)
        sizes = dict(previous=previous, post=post)
        self.results(options, title="Screener", instrument=Instrument.OPTION, **sizes)
        return options

    def screener(self, options):
        tau = options["tau"].to_numpy(dtype=float)
        mae = options["mae"].to_numpy(dtype=float)
        tiv = options["tiv"].to_numpy(dtype=float)
        ntiv = self.neighborhood(tau, mae, tiv)
        ntiv = np.fromiter(ntiv, dtype=np.float64)
        valid = np.isfinite(ntiv) & (ntiv > 0)
        if not valid.any(): raise VarianceError()
        ntiv = ntiv[valid]
        quantile = np.quantile(ntiv, self.quantile)
        median = np.median(ntiv)
        mask = ntiv > max(quantile, self.multiple * median)
        return options.loc[~mask]

    @property
    def neighborhood(self): return self.__neighborhood
    @property
    def quantile(self): return self.__quantile
    @property
    def multiple(self): return self.__multiple


class VarianceStandardizer(Logging):
    def __init__(self, *args, neighbors, **kwargs):
        super().__init__(*args, **kwargs)
        self.__neighborhood = Neighborhood(neighbors)

    def __call__(self, options, surface, /, **kwargs):
        assert isinstance(options, pd.DataFrame)
        tau = options["tau"].to_numpy(dtype=float)
        mae = options["mae"].to_numpy(dtype=float)
        tiv = options["tiv"].to_numpy(dtype=float)
        standard = self.standardize(tau, mae, tiv, surface)
        standard = pd.Series(standard, name="zscore", index=options.index)
        options = pd.concat([options, standard], axis=1)
        self.results(options, title="Calculated", instrument=Instrument.OPTION)
        return options

    def standardize(self, t, k, w, f):
        μ = np.vectorize(f)(t, k)
        σ = self.neighborhood(t, k, w)
        σ = np.fromiter(σ, dtype=np.float64)
        ε = np.quantile(σ[σ > 0], 0.1) if np.any(σ > 0) else 1e-8
        z = (w - μ) / np.maximum(σ, ε)
        return z

    @property
    def neighborhood(self): return self.__neighborhood


