# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 2026
@name:   Option Dataset Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.spatial import cKDTree

from support.concepts import NumRange, DateRange
from support.finance import Concepts, Alerting
from support.equations import Equations
from support.surface import Surface

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DatasetScreener", "GeneralCalculator", "LocalCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True)
class Variables: tau: float; mae: float

@dataclass(frozen=True)
class Axes: x: float; y: float; z: float = None

@dataclass(frozen=False)
class Dataset:
    scatter: pd.DataFrame = None
    surface: Surface = None
    center: Variables = None
    radius: Variables = None

    def __bool__(self): return self.scatter is not None and not self.scatter.empty
    def __len__(self): return len(self.scatter)

    def __str__(self):
        tickers = "|".join(list(self.scatter["ticker"].unique()))
        expires = DateRange.create(list(self.scatter["expire"].unique()))
        expires = f"{expires.minimum.strftime('%Y%m%d')}->{expires.maximum.strftime('%Y%m%d')}"
        inner, outer = self.inner, self.outer
        variables = [self.string(axis, inner, outer) for axis in list("xy")]
        return "\n".join([f"{tickers}|{expires}[{len(self):.0f}]"] + variables)

    def __post_init__(self):
        mask = (self.scatter["tau"].notna() & self.scatter["mae"].notna() & self.scatter["tiv"].notna())
        self.scatter = self.scatter.loc[mask].copy()

    @staticmethod
    def string(axis, inner, outer):
        inner, outer = getattr(inner, axis), getattr(outer, axis)
        inner = f"({inner.minimum:.03f}, {inner.maximum:.03f})"
        outer = f"({outer.minimum:.03f}, {outer.maximum:.03f})"
        string = f"{str(axis).upper()}[{inner} ∈ {outer}]"
        return string

    @property
    def inner(self):
        x = NumRange.create([self.scatter["tau"].min(), self.scatter["tau"].max()])
        y = NumRange.create([self.scatter["mae"].min(), self.scatter["mae"].max()])
        z = NumRange.create([self.scatter["tiv"].min(), self.scatter["tiv"].max()])
        return Axes(x=x, y=y, z=z)

    @property
    def outer(self):
        x = NumRange.create([self.center.tau - self.radius.tau, self.center.tau + self.radius.tau])
        y = NumRange.create([self.center.mae - self.radius.mae, self.center.mae + self.radius.mae])
        return Axes(x=x, y=y)

    @property
    def xyz(self): return self.scatter.rename(columns=dict(zip("tau,mae,tiv".split(","), list("xyz"))), inplace=False)
    @property
    def x(self): return self.scatter["tau"].rename("x")
    @property
    def y(self): return self.scatter["mae"].rename("y")
    @property
    def z(self): return self.scatter["tiv"].rename("z")


class DatasetCalculator(Alerting):
    @staticmethod
    def average(axis, decimals): return round((axis.minimum + axis.maximum) / 2, decimals)
    @staticmethod
    def distance(axis, decimals): return round((axis.maximum - axis.minimum) / 2, decimals)


class GeneralCalculator(DatasetCalculator, Equations):
    mae = lambda forward, strike, option: np.log(forward / strike) * option.astype(int)
    tiv = lambda implied, tau: tau * np.square(implied)

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        scatter = self.execute(options, *args, **kwargs)
        scatter = pd.concat([options, scatter], axis=1)
        tau = NumRange.create([scatter["tau"].min(), scatter["tau"].max()])
        mae = NumRange.create([scatter["mae"].min(), scatter["mae"].max()])
        center = Variables(tau=self.average(tau, 3), mae=self.average(mae, 3))
        radius = Variables(tau=self.distance(tau, 3), mae=self.distance(mae, 3))
        dataset = Dataset(scatter=scatter, center=center, radius=radius)
        self.alert(scatter, title="Calculated", instrument=Concepts.Securities.Instrument.OPTION)
        return dataset


class LocalCalculator(DatasetCalculator):
    def __init__(self, *args, quantity=15, coverage=Variables(tau=5, mae=10), radius=Variables(tau=0.15, mae=0.05), **kwargs):
        assert isinstance(radius, (tuple, Variables))
        super().__init__(*args, **kwargs)
        coverage = Variables(**dict(zip(["tau", "mae"], coverage))) if isinstance(coverage, tuple) else coverage
        radius = Variables(**dict(zip(["tau", "mae"], radius))) if isinstance(radius, tuple) else radius
        self.__quantity = int(quantity)
        self.__coverage = coverage
        self.__radius = radius

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        mask = options["tau"].notna() & options["mae"].notna() & options["tiv"].notna()
        options = options[mask].dropna(how="all", inplace=False)
        tau, mae = self.tau(options), self.mae(options)
        pairs = [Variables(tau=i, mae=j) for j in mae for i in tau]
        for pair in pairs:
            tau = NumRange.create([pair.tau - self.radius.tau, pair.tau + self.radius.tau])
            mae = NumRange.create([pair.mae - self.radius.mae, pair.mae + self.radius.mae])
            tau = options["tau"].between(tau.minimum, tau.maximum)
            mae = options["mae"].between(mae.minimum, mae.maximum)
            scatter = options.loc[tau & mae]
            if not self.adequate(scatter): continue
            tiv = NumRange.create([scatter["tiv"].min(), scatter["tiv"].max()])
            center = Variables(tau=pair.tau, mae=pair.mae)
            radius = Variables(tau=self.radius.tau, mae=self.radius.mae)
            dataset = Dataset(scatter=scatter, center=center, radius=radius)
            self.alert(scatter, title="Calculated", instrument=Concepts.Securities.Instrument.OPTION)
            yield dataset

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


class DatasetScreener(Alerting):
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



