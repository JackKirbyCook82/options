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
from abc import ABC, abstractmethod

from finance.enumerations import Instrument
from finance.logging import Logging
from support.custom import NumRange

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["PartitionCalculator", "ProximityCalculator", "Localizing"]
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


class LocalizingError(Exception): pass
class PartitionedLocalizingError(Exception): pass
class ProximityLocalizingError(LocalizingError): pass


class LocalizingCalculator(Logging, ABC):
    def __init__(self, *args, localizing, samples=35, overlap=0.80, **kwargs):
        assert isinstance(localizing, Localizing)
        super().__init__(*args, **kwargs)
        self.__localizing = localizing
        self.__overlap = float(overlap)
        self.__samples = int(samples)

    def centers(self, options):
        taus = np.sort(options["tau"].unique().astype(float))
        mae = options["mae"].to_numpy(dtype=float)
        low, high = np.nanmin(mae), np.nanmax(mae)
        step = self.localizing.maes.radii.inner / 2
        maes = np.arange(low, high + step, step, dtype=float)
        order = np.argsort(np.abs(maes))
        return SimpleNamespace(tau=taus, mae=maes[order])

    def taus(self, center, centers, /, index):
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

    @abstractmethod
    def calculator(self, *args, **kwargs): pass
    @abstractmethod
    def generator(self, *args, **kwargs): pass

    @staticmethod
    def contained(localized, spread, key="osi"):
        assert key == "osi"
        if key in localized.columns and key in spread.columns:
            available = set(localized[key].dropna())
            required = set(spread[key].dropna())
            return required.issubset(available)
        return True

    @staticmethod
    def localize(options, local):
        assert isinstance(local, Local)
        tau = options["tau"].isin(local.tau.population)
        mae = options["mae"].between(local.mae.population.minimum, local.mae.population.maximum)
        localized = options[tau & mae].copy()
        localized.attrs["tau"] = local.tau
        localized.attrs["mae"] = local.mae
        return localized

    @staticmethod
    def cleaner(dataframe):
        mask = dataframe["tau"].notna() & dataframe["mae"].notna()
        if "tiv" in dataframe.columns: mask &= dataframe["tiv"].notna()
        return dataframe[mask].copy()

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


class ProximityCalculator(LocalizingCalculator):
    def __call__(self, options, spreads, **kwargs):
        assert isinstance(options, pd.DataFrame) and not options.empty
        assert isinstance(spreads, pd.DataFrame) and not spreads.empty
        options = self.cleaner(options)
        spreads = self.cleaner(spreads)
        proximity = self.calculator(options, spreads, **kwargs)
        self.results(proximity, title="Calculated", instrument=Instrument.OPTION)
        return proximity

    def calculator(self, options, spreads, **kwargs):
        for local in self.generator(options, spreads, **kwargs):
            localized = self.localize(options, local)
            if not self.adequate(localized): continue
            if not self.contained(localized, spreads): continue
            return localized
        raise ProximityLocalizingError()

    def generator(self, options, spreads, **kwargs):
        centers = self.centers(options)
        tauCenter = float(spreads["tau"].mean())
        maeCenter = float(spreads["mae"].mean())
        distances = np.abs(centers.tau.astype(float) - float(tauCenter))
        index = int(np.argmin(distances))
        for tau in self.taus(tauCenter, centers.tau, index=index):
            for mae in self.maes(maeCenter):
                yield Local(tau=tau, mae=mae)


class PartitionCalculator(LocalizingCalculator):
    def __call__(self, options, /, **kwargs):
        assert isinstance(options, pd.DataFrame) and not options.empty
        options = self.cleaner(options)
        for local in self.calculator(options, **kwargs):
            self.results(local, title="Calculated", instrument=Instrument.OPTION)
            yield local

    def calculator(self, options, **kwargs):
        history = list()
        for local in self.generator(options, **kwargs):
            localized = self.localize(options, local)
            if not self.adequate(localized): continue
            if self.similar(localized, history): continue
            index = set(localized.index)
            history.append(index)
            yield localized

    def generator(self, options, **kwargs):
        centers = self.centers(options)
        for index, tauCenter in enumerate(centers.tau):
            for tau in self.taus(tauCenter, centers.tau, index=index):
                for maeCenter in centers.mae:
                    for mae in self.maes(maeCenter):
                        yield Local(tau=tau, mae=mae)



