# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 2026
@name:   Option Localizing Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from dataclasses import dataclass
from types import SimpleNamespace

from finance.enumerations import Instrument
from finance.logging import Logging
from support.custom import NumRange

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["LocalizingCalculator", "LocalizingVariables"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True)
class Windows:
    inner: int = 1; outer: int = 3; step: int = 1

    def __iter__(self):
        yield from range(self.inner, self.outer + 1, self.step)

@dataclass(frozen=True)
class Tau: population: NDArray[np.floating]; center: float; span: int

@dataclass(frozen=True)
class Taus:
    windows: Windows; coverage: int = 3; limit: float = 45 / 365

    def __call__(self, centers):
        for index, center in enumerate(centers):
            for window in self.windows:
                low = max(0, index - window)
                high = min(len(centers), index + window + 1)
                population = centers[low:high]
                if len(population) == 0: continue
                size = float(np.max(population) - np.min(population))
                if size > self.limit: continue
                yield Tau(population=population, center=center, span=window)


@dataclass(frozen=True)
class Radii:
    inner: float = 0.05; outer: float = 0.12; step: float = 0.01

    def __iter__(self):
        radius = float(self.inner)
        while radius <= self.outer:
            yield radius
            radius += self.step

@dataclass(frozen=True)
class Mae: population: NumRange; center: float; span: float

@dataclass(frozen=True)
class Maes:
    radii: Radii; coverage: int = 10

    def __call__(self, center):
        for radius in self.radii:
            population = NumRange.create([center - radius, center + radius])
            yield Mae(population=population, center=center, span=radius)


@dataclass(frozen=True)
class LocalizingVariables:
    taus: Taus; maes: Maes

    @classmethod
    def create(cls, /, radius, window, coverage, limit):
        assert isinstance(radius, tuple) and len(radius) == 3
        assert isinstance(window, tuple) and len(window) == 3
        assert isinstance(coverage, tuple) and len(coverage) == 2
        radii = Radii(*list(map(float, radius)))
        windows = Windows(*list(map(int, window)))
        assert radii.step > 0 and windows.step >= 1
        assert radii.outer >= radii.inner and windows.outer >= windows.inner
        taus = Taus(windows=windows, coverage=coverage[0], limit=float(limit))
        maes = Maes(radii=radii, coverage=coverage[1])
        return cls(taus=taus, maes=maes)


class LocalizingCalculator(Logging):
    def __init__(self, *args, localizing, samples=35, overlap=0.80, **kwargs):
        assert isinstance(localizing, LocalizingVariables)
        super().__init__(*args, **kwargs)
        self.__localizing = localizing
        self.__overlap = float(overlap)
        self.__samples = int(samples)

    def __call__(self, options, /, **kwargs):
        assert isinstance(options, pd.DataFrame)
        options = options[options["tau"].notna() & options["mae"].notna() & options["tiv"].notna()].copy()
        for local in self.calculator(options):
            self.results(local, title="Calculated", instrument=Instrument.OPTION)
            yield local

    def calculator(self, options):
        centers, history = self.centers(options), list()
        for tau in self.localizing.taus(centers.taus):
            for center in centers.maes:
                for mae in self.localizing.maes(center):
                    localized = self.localized(options, tau, mae)
                    if self.adequate(localized) and not self.similar(localized, history):
                        index = set(localized.index)
                        history.append(index)
                        localized.attrs["tau"] = tau
                        localized.attrs["mae"] = mae
                        yield localized
                        break

    def centers(self, options):
        taus = np.sort(options["tau"].unique().astype(float))
        mae = options["mae"].to_numpy(dtype=float)
        low, high = np.nanmin(mae), np.nanmax(mae)
        step = self.localizing.maes.radii.inner / 2
        maes = np.arange(low, high + step, step, dtype=float)
        order = np.argsort(np.abs(maes))
        return SimpleNamespace(taus=taus, maes=maes[order])

    def adequate(self, localized):
        tau = localized["tau"].nunique() >= self.localizing.taus.coverage
        mae = localized["mae"].nunique() >= self.localizing.maes.coverage
        return (len(localized) >= self.samples) and tau and mae

    def similar(self, localized, history):
        current = set(localized.index)
        for prior in history:
            union = len(current | prior)
            if union == 0: continue
            overlap = len(current & prior) / union
            if overlap >= self.overlap: return True
        return False

    @staticmethod
    def localized(options, tau, mae):
        tau = options["tau"].isin(tau.population)
        mae = options["mae"].between(mae.population.minimum, mae.population.maximum)
        return options[tau & mae]

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
    def localizing(self): return self.__localizing
    @property
    def samples(self): return self.__samples
    @property
    def overlap(self): return self.__overlap



