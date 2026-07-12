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
__all__ = ["LocalizingCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True)
class Windows:
    inner: int = 1; outer: int = 3; step: int = 1

    def __iter__(self):
        yield from range(self.inner, self.outer + 1, self.step)

@dataclass(frozen=True)
class Radii:
    inner: float = 0.05; outer: float = 0.12; step: float = 0.01

    def __iter__(self):
        radius = float(self.inner)
        while radius <= self.outer:
            yield radius
            radius += self.step


@dataclass(frozen=True)
class Tau: population: NDArray[np.floating]; center: float; span: int

@dataclass(frozen=True)
class Mae: population: NumRange; center: float; span: float

@dataclass(frozen=True)
class Local: tau: Tau; mae: Mae


@dataclass(frozen=True)
class Centers: tau: NDArray[np.floating]; mae: NDArray[np.floating]

@dataclass(frozen=True)
class Center: tau: float; mae: float


@dataclass(frozen=True)
class Taus:
    windows: Windows; coverage: int = 3; limit: float = 45 / 365

@dataclass(frozen=True)
class Maes: radii: Radii; coverage: int = 10

@dataclass(frozen=True)
class Localizing:
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


class LocalizingGenerator(object):
    def __init__(self, options, localizing):
        centers = self.create(options, localizing)
        self.__localizing = localizing
        self.__centers = centers

    def __iter__(self):
        for index, tauCenter in enumerate(self.centers.tau):
            for tau in self.taus(index, tauCenter):
                for maeCenter in self.centers.mae:
                    for mae in self.maes(maeCenter):
                        yield Local(tau=tau, mae=mae)

    def taus(self, index, center):
        for window in self.localizing.taus.windows:
            low = max(0, index - window)
            high = min(len(self.centers.tau), index + window + 1)
            population = self.centers.tau[low:high]
            if len(population) == 0: continue
            size = float(np.max(population) - np.min(population))
            if size > self.localizing.taus.limit: continue
            yield Tau(population=population, center=center, span=window)

    def maes(self, center):
        for radius in self.localizing.maes.radii:
            population = NumRange.create([center - radius, center + radius])
            yield Mae(population=population, center=center, span=radius)

    @staticmethod
    def create(options, localizing):
        taus = np.sort(options["tau"].unique().astype(float))
        mae = options["mae"].to_numpy(dtype=float)
        low, high = np.nanmin(mae), np.nanmax(mae)
        step = localizing.maes.radii.inner / 2
        maes = np.arange(low, high + step, step, dtype=float)
        order = np.argsort(np.abs(maes))
        return Centers(tau=taus, mae=maes[order])

    @property
    def localizing(self): return self.__localizing
    @property
    def centers(self): return self.__centers


class ProximityCalculator(Logging):
    def __call__(self, options, spread, **kwargs):
        assert isinstance(options, pd.DataFrame) and not options.empty
        assert isinstance(spread, pd.DataFrame) and not spread.empty
        options = self.cleaner(options)
        proximity = self.calculate(options, spread, **kwargs)
        self.results(Proximity, title="Calculated", instrument=Instrument=OPTION)
        return proximity

    def calculate(self, options, spread, **kwargs):
        pass

    @staticmethod
    def center(spread):
        tau = float(spread["tau"].mean())
        mae = float(spread["mae"].mean())
        return Center(tau=tau, mae=mae)


class LocalizingCalculator(Logging):
    def __init__(self, *args, localizing, samples=35, overlap=0.80, **kwargs):
        assert isinstance(localizing, Localizing)
        super().__init__(*args, **kwargs)
        self.__localizing = localizing
        self.__overlap = float(overlap)
        self.__samples = int(samples)

    def __call__(self, options, /, **kwargs):
        assert isinstance(options, pd.DataFrame) and not options.empty
        options = self.cleaner(options)
        for local in self.calculator(options, **kwargs):
            self.results(local, title="Calculated", instrument=Instrument.OPTION)
            yield local

    def calculator(self, options, **kwargs):
        generator = LocalizingGenerator(options, self.localizing)
        history = list()
        for local in generator:
            localized = self.localizer(options, local)
            if self.adequate(localized) and not self.similar(localized, history):
                index = set(localized.index)
                history.append(index)
                localized.attrs["tau"] = localized.tau
                localized.attrs["mae"] = localized.mae
                yield localized
                break

    def taus(self, centers):
        for index, center in enumerate(centers):
            for window in self.localizing.taus.windows:
                low = max(0, index - window)
                high = min(len(centers), index + window + 1)
                population = centers[low:high]
                if len(population) == 0: continue
                size = float(np.max(population) - np.min(population))
                if size > self.localizing.taus.limit: continue
                yield Tau(population=population, center=center, span=window)

    def maes(self, center):
        for radius in self.localizing.maes.radii:
            population = NumRange.create([center - radius, center + radius])
            yield Mae(population=population, center=center, span=radius)

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
    def cleaner(options):
        mask = options["tau"].notna() & options["mae"].notna() & options["tiv"].notna()
        options = options[mask].copy()
        return options

    @staticmethod
    def localizer(options, local):
        tau = options["tau"].isin(local.tau.population)
        mae = options["mae"].between(local.mae.population.minimum, local.mae.population.maximum)
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







