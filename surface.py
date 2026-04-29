# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 2026
@name:   Surface Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

from support.finance import Concepts, Alerting
from support.equations import Equations
from support.concepts import NumRange

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SurfaceCalculator", "LocalCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass
class Axes: tau: float; mae: float


class SurfaceCalculator(Equations, Alerting):
    mae = lambda forward, strike, option: np.log(forward / strike) * option.astype(int)
    tiv = lambda implied, tau: tau * np.square(implied)

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        surface = self.execute(options, *args, **kwargs)
        surface = pd.concat([options, surface], axis=1)
        mask = surface["tau"].notna() & surface["mae"].notna() & surface["tiv"].notna()
        surface = surface[mask].dropna(how="all", inplace=False)
        self.alert(surface, title="Calculated", instrument=Concepts.Securities.Instrument.OPTION)
        return surface


class LocalCalculator(Alerting):
    def __init__(self, *args, radius=Axes(tau=0.15, mae=0.05), **kwargs):
        assert isinstance(radius, (tuple, Axes))
        super().__init__(*args, **kwargs)
        radius = Axes(**dict(zip(["tau", "mae"], radius))) if isinstance(radius, tuple) else radius
        self.__radius = radius

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        mask = options["tau"].notna() & options["mae"].notna() & options["tiv"].notna()
        options = options[mask].dropna(how="all", inplace=False)
        mae, tau = self.mae(options), self.tau(options)
        pairs = [Axes(tau=i, mae=j) for j in mae for i in tau]
        for local in self.generator(options, pairs):
            self.alert(local, title="Localized", instrument=Concepts.Securities.Instrument.OPTION)
            yield local

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
    def radius(self): return self.__radius



