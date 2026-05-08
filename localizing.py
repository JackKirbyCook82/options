# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 2026
@name:   Option Dataset Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

from support.concepts import NumRange
from support.finance import Concepts, Alerting

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["LocalizingCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True)
class Variables: tau: float; mae: float


class LocalizingCalculator(Alerting):
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
        taus, maes = self.taus(options), self.maes(options)
        pairs = [Variables(tau=tau, mae=mae) for mae in maes for tau in taus]
        for pair in pairs:
            tau = NumRange.create([pair.tau - self.radius.tau, pair.tau + self.radius.tau])
            mae = NumRange.create([pair.mae - self.radius.mae, pair.mae + self.radius.mae])
            tau = options["tau"].between(tau.minimum, tau.maximum)
            mae = options["mae"].between(mae.minimum, mae.maximum)
            local = options.loc[tau & mae]
            if not self.adequate(local): continue
            local.attrs["center"] = Variables(tau=pair.tau, mae=pair.mae)
            local.attrs["radius"] = Variables(tau=self.radius.tau, mae=self.radius.mae)
            self.alert(local, title="Calculated", instrument=Concepts.Securities.Instrument.OPTION)
            yield local

    def taus(self, options):
        tau = np.sort(options["tau"].dropna().unique().astype(float))
        limits = NumRange.create([tau.min() + self.radius.tau, tau.max() - self.radius.tau])
        tau = tau[(tau >= limits.minimum) & (tau <= limits.maximum)]
        tau = np.fromiter(self.alternate(tau), dtype=float)
        return tau

    def maes(self, options):
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
    def average(axis, decimals): return round((axis.minimum + axis.maximum) / 2, decimals)
    @staticmethod
    def distance(axis, decimals): return round((axis.maximum - axis.minimum) / 2, decimals)

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

