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
from support.surface import Surface

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DatasetScreener", "GeneralCalculator", "LocalCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True)
class Axes: tau: float; mae: float

@dataclass(frozen=False)
class Dataset:
    scatter: pd.DataFrame = None
    surface: Surface = None
    center: Axes = None
    radius: Axes = None

    def __bool__(self): return self.scatter is not None and not self.scatter.empty
    def __len__(self): return len(self.scatter)

    def __post_init__(self):
        tau = self.scatter["tau"].notna()
        mae = self.scatter["mae"].notna()
        tiv = self.scatter["tiv"].notna()
        mask = tau & mae & tiv
        scatter = self.scatter[mask].dropna(how="all", inplace=False)
        columns = dict(zip(list("tau|mae|tiv".split("|")), list("xyz")))
        scatter = scatter.rename(columns=columns)
        self.scatter = scatter

    @property
    def inner(self):
        tau = NumRange.create([self.scatter["tau"].min(), self.scatter["tau"].max()])
        mae = NumRange.create([self.scatter["mae"].min(), self.scatter["mae"].max()])
        return Axes(tau=tau, mae=mae)

    @property
    def outer(self):
        tau = NumRange.create([self.center.tau - self.radius.tau, self.center.tau + self.radius.tau])
        mae = NumRange.create([self.center.mae - self.radius.mae, self.center.mae + self.radius.mae])
        return Axes(tau=tau, mae=mae)


class DatasetCalculator(Alerting): pass
class GeneralCalculator(DatasetCalculator, Equations):
    mae = lambda forward, strike, option: np.log(forward / strike) * option.astype(int)
    tiv = lambda implied, tau: tau * np.square(implied)

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        scatter = self.execute(options, *args, **kwargs)
        scatter = pd.concat([options, scatter], axis=1)
        self.alert(scatter, title="Calculated", instrument=Concepts.Securities.Instrument.OPTION)
        dataset = Dataset(scatter=scatter)
        return dataset


class LocalCalculator(DatasetCalculator):
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
        assert isinstance(options, pd.DataFrame) or isinstance(options, Dataset)
        if isinstance(options, Dataset): options = options.scatter
        mask = options["tau"].notna() & options["mae"].notna() & options["tiv"].notna()
        options = options[mask].dropna(how="all", inplace=False)
        tau, mae = self.tau(options), self.mae(options)
        pairs, count = [Axes(tau=i, mae=j) for j in mae for i in tau], 0
        for center in pairs:
            tau = NumRange.create([center.tau - self.radius.tau, center.tau + self.radius.tau])
            mae = NumRange.create([center.mae - self.radius.mae, center.mae + self.radius.mae])
            tau = options["tau"].between(tau.minimum, tau.maximum)
            mae = options["mae"].between(mae.minimum, mae.maximum)
            scatter = options.loc[tau & mae]
            if not self.adequate(scatter): continue
            self.alert(scatter, title="Calculated", instrument=Concepts.Securities.Instrument.OPTION)
            dataset = Dataset(scatter=scatter, center=center, radius=self.radius)
            yield dataset; count += 1
            if self.count is not None and count >= self.count: break

    def tau(self, options):
        tau = np.sort(options["tau"].dropna().unique().astype(float))
        limits = NumRange.create([tau.min() + self.radius.tau, tau.max() - self.radius.tau])
        tau = tau[(tau >= limits.minimum) & (tau <= limits.maximum)]
        tau = np.fromiter(self.alternate(tau), dtype=float)
        return tau

    def mae(self, options):
        mae = options["mae"].to_numpy(dtype=float)
        limits = NumRange.create([np.nanmin(mae) + self.radius.mae, np.nanmax(mae) - self.radius.mae])
        step = self.radius.mae / 2
        mae = np.arange(limits.minimum, limits.maximum + step, step, dtype=float)
        order = np.argsort(np.abs(mae))
        return mae[order]

    def adequate(self, local):
        tau = local["tau"].nunique() >= self.coverage.tau
        mae = local["mae"].nunique() >= self.coverage.mae
        quantity = len(local) >= self.quantity
        coverage = tau & mae
        return quantity & coverage

    @staticmethod
    def alternate(array):
        center = len(array) // 2
        yield array[center]
        left = iter(array[:center][::-1])
        right = iter(array[center+1:])
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


class DatasetScreener(Alerting):
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



