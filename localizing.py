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
__all__ = ["LocalizingCalculator", "Radius", "Variables", "Axes"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True)
class Radius:
    inner: float = 0.05
    outer: float = 0.12
    step: float = 0.01

@dataclass(frozen=True)
class Tau:
    window: int = 2
    coverage: int = 5

@dataclass(frozen=True)
class Mae:
    radius: Radius
    coverage: int = 10

@dataclass(frozen=True)
class Variables: tau: Any; mae: Any
class Axes: Tau = Tau; Mae = Mae


class LocalizingCalculator(Alerting):
    def __init__(self, *args, variables, samples=35, overlap=0.80, **kwargs):
        assert isinstance(variables, Variables)
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
        taus, maes, history = self.taus(options), self.maes(options), list()
        for index, tau in enumerate(taus):
            low = max(0, index - self.variables.tau.window)
            high = min(len(taus), index + self.variables.tau.window + 1)
            for mae in maes:
                center = Variables(tau=tau, mae=mae)
                radius = Variables(tau=len(taus[low:high]), mae=self.variables.mae.radius.inner)
                while radius.mae <= self.variables.mae.radius.outer:
                    population = Variables(tau=taus[low:high], mae=NumRange.create([mae - radius.mae, mae + radius.mae]))
                    mask = Variables(options["tau"].isin(population.tau), options["mae"].between(population.mae.minimum, population.mae.maximum))
                    local = options[mask.tau & mask.mae]
                    if self.adequate(local) and not self.similar(local, history):
                        local.attrs["population"] = population
                        local.attrs["center"] = center
                        local.attrs["radius"] = radius
                        history.append(set(local.index))
                        yield local
                        break
                    radius = Variables(tau=radius.tau, mae=radius.mae + self.variables.mae.radius.step)

    @staticmethod
    def taus(options): return np.sort(options["tau"].unique().astype(float))
    def maes(self, options):
        mae = options["mae"].to_numpy(dtype=float)
        low, high = np.nanmin(mae), np.nanmax(mae)
        step = self.variables.mae.radius.inner / 2
        centers = np.arange(low, high + step, step, dtype=float)
        order = np.argsort(np.abs(centers))
        return centers[order]

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



