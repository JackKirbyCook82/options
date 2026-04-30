# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 2026
@name:   Surface Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.spatial import cKDTree

from support.finance import Concepts, Alerting
from support.equations import Equations
from support.concepts import NumRange

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SurfaceScreener", "SurfaceCalculator", "LocalCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True)
class Axes: tau: float; mae: float


class SurfaceCalculator(Equations, Alerting):
    mae = lambda forward, strike, option: np.log(forward / strike) * option.astype(int)
    tiv = lambda implied, tau: tau * np.square(implied)

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        mask = options["tau"].notna() & options["mae"].notna() & options["tiv"].notna()
        options = options[mask].dropna(how="all", inplace=False)
        surface = self.execute(options, *args, **kwargs)
        options = pd.concat([options, surface], axis=1)
        self.alert(options, title="Calculated", instrument=Concepts.Securities.Instrument.OPTION)
        return options


class SurfaceScreener(Alerting):
    def __init__(self, *args, neighbors=12, threshold=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.__neighbors = neighbors
        self.__threshold = threshold

    def __call__(self, options, *args, **kwargs):
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


class LocalCalculator(Alerting):
    def __init__(self, *args, count=None, quantity=15, coverage=Axes(tau=5, mae=10), radius=Axes(tau=0.15, mae=0.05), **kwargs):
        assert isinstance(radius, (tuple, Axes))
        super().__init__(*args, **kwargs)
        coverage = Axes(**dict(zip(["tau", "mae"], coverage))) if isinstance(coverage, tuple) else coverage
        radius = Axes(**dict(zip(["tau", "mae"], radius))) if isinstance(radius, tuple) else radius
        self.__quantity = int(quantity)
        self.__coverage = coverage
        self.__radius = radius
        self.__count = count

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        mask = options["tau"].notna() & options["mae"].notna() & options["tiv"].notna()
        options = options[mask].dropna(how="all", inplace=False)
        tau, mae = self.tau(options), self.mae(options)
        pairs, count = [Axes(tau=i, mae=j) for j in mae for i in tau], 0
        for local in self.generator(options, pairs):
            if not self.adequate(local): continue
            self.alert(local, title="Calculated", instrument=Concepts.Securities.Instrument.OPTION)
            yield local; count += 1
            if count is not None and count >= self.count: break

    def generator(self, options, pairs):
        for axes in pairs:
            tau = NumRange.create([axes.tau - self.radius.tau, axes.tau + self.radius.tau])
            mae = NumRange.create([axes.mae - self.radius.mae, axes.mae + self.radius.mae])
            tau = options["tau"].between(tau.minimum, tau.maximum)
            mae = options["mae"].between(mae.minimum, mae.maximum)
            yield options.loc[tau & mae]

    def mae(self, options):
        mae = options["mae"].to_numpy(dtype=float)
        limits = NumRange.create([np.nanmin(mae) + self.radius.mae, np.nanmax(mae) - self.radius.mae])
        step = self.radius.mae / 2
        mae = np.arange(limits.minimum, limits.maximum + step, step, dtype=float)
        order = np.argsort(np.abs(mae))
        return mae[order]

    def tau(self, options):
        tau = np.sort(options["tau"].dropna().unique().astype(float))
        limits = NumRange.create([tau.min() + self.radius.tau, tau.max() - self.radius.tau])
        tau = tau[(tau >= limits.minimum) & (tau <= limits.maximum)]
        tau = np.array(self.alternate(tau))
        return tau

    def adequate(self, local):
        tau = len(local["tau"].nunique()) >= self.coverage.tau
        mae = len(local["mae"].nunique()) >= self.coverage.mae
        quantity = len(local) >= self.quantity
        coverage = tau & mae
        return quantity & coverage

    @staticmethod
    def alternate(array):
        center = array[len(array) // 2]
        left = iter(array[:center][::-1])
        right = iter(array[center:])
        while True:
            try: yield next(left)
            except StopIteration: yield from right; return
            try: yield next(right)
            except StopIteration: yield from left; return

    @property
    def quantity(self): return self.__quantity
    @property
    def coverage(self): return self.__coverage
    @property
    def radius(self): return self.__radius
    @property
    def count(self): return self.__count



