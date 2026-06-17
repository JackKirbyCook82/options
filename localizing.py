# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 2026
@name:   Option Localizing Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from typing import Any
from dataclasses import dataclass

from finance.variables import Alerting, Enumerations
from support.custom import NumRange

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["LocalizingCalculator", "LocalizingVariables"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True)
class Window:
    inner: int = 1
    outer: int = 3
    step: int = 1

    def __call__(self, centers):
        pass

@dataclass(frozen=True)
class Radius:
    inner: float = 0.05
    outer: float = 0.12
    step: float = 0.01

    def __call__(self, centers):
        pass

@dataclass(frozen=True)
class Tau:
    window: Window
    coverage: int = 3
    limit: float = 45 / 365

@dataclass(frozen=True)
class Mae:
    radius: Radius
    coverage: int = 10

@dataclass(frozen=True)
class LocalizingVariables:
    tau: Any; mae: Any

    @classmethod
    def create(cls, /, radius, window, coverage, limit):
        assert isinstance(radius, tuple) and len(radius) == 3
        assert isinstance(window, tuple) and len(window) == 3
        assert isinstance(coverage, tuple) and len(coverage) == 2
        radius = Radius(*list(map(float, radius)))
        window = Window(*list(map(float, window)))
        tau = Tau(window=window, coverage=coverage[0], limit=float(limit))
        mae = Mae(radius=radius, coverage=coverage[1])
        return cls(tau=tau, mae=mae)


class LocalizingCalculator(Alerting):
    def __init__(self, *args, variables, samples=35, overlap=0.80, **kwargs):
        assert isinstance(variables, LocalizingVariables)
        super().__init__(*args, **kwargs)
        self.__variables = variables
        self.__overlap = float(overlap)
        self.__samples = int(samples)

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        options = options[options["tau"].notna() & options["mae"].notna() & options["tiv"].notna()].copy()
        for local in self.calculator(options):
            self.alert(local, title="Calculated", instrument=Enumerations.Instrument.OPTION)
            yield local

    def calculator(self, options):
        centers, history = self.centers(options), list()


#    def calculator(self, options):
#        taus, maes, history = self.taus(options), self.maes(options), list()
#        for index, tau in enumerate(taus):
#            low = max(0, index - self.variables.tau.window)
#            high = min(len(taus), index + self.variables.tau.window + 1)
#            for mae in maes:
#                center = LocalizingVariables(tau=tau, mae=mae)
#                radius = LocalizingVariables(tau=len(taus[low:high]), mae=self.variables.mae.radius.inner)
#                while radius.mae <= self.variables.mae.radius.outer:
#                    population = LocalizingVariables(tau=taus[low:high], mae=NumRange.create([mae - radius.mae, mae + radius.mae]))
#                    mask = LocalizingVariables(options["tau"].isin(population.tau), options["mae"].between(population.mae.minimum, population.mae.maximum))
#                    local = options[mask.tau & mask.mae]
#                    if self.adequate(local) and not self.similar(local, history):
#                        local.attrs["population"] = population
#                        local.attrs["center"] = center
#                        local.attrs["radius"] = radius
#                        history.append(set(local.index))
#                        yield local
#                        break
#                    radius = LocalizingVariables(tau=radius.tau, mae=radius.mae + self.variables.mae.radius.step)

    def centers(self, options):
        taus = np.sort(options["tau"].unique().astype(float))
        mae = options["mae"].to_numpy(dtype=float)
        low, high = np.nanmin(mae), np.nanmax(mae)
        step = self.variables.mae.radius.inner / 2
        maes = np.arange(low, high + step, step, dtype=float)
        order = np.argsort(np.abs(maes))
        return LocalizingVariables(taus, maes[order])

    def adequate(self, local):
        tau = local["tau"].nunique() >= self.variables.tau.coverage
        mae = local["mae"].nunique() >= self.variables.mae.coverage
        return (len(local) >= self.samples) and tau and mae

    def similar(self, local, history):
        current = set(local.index)
        for prior in history:
            union = len(current | prior)
            if union == 0: continue
            overlap = len(current & prior) / union
            if overlap >= self.overlap: return True
        return False

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
    def variables(self): return self.__variables
    @property
    def samples(self): return self.__samples
    @property
    def overlap(self): return self.__overlap



