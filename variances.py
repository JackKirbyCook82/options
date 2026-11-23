# -*- coding: utf-8 -*-
"""
Created on Fri May 8 2026
@name:   Option Variance Objects
@author: Jack Kirby Cook
@file:   options/variances.py

"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from dataclasses import dataclass
import pandas_market_calendars as calenders

from finance.variables import Collection, Variables
from finance.console import LoggingDecorators
from support.equations import DataframeEquation
from support.mixins import LoggingMixin
from support.pipeline import Pipelines

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["VarianceCalculator", "VarianceScreener", "VarianceStandardizer"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


nyse = calenders.get_calendar("NYSE")


class VarianceError(Exception): pass
class SizeError(VarianceError): pass
class NumericError(VarianceError): pass


@dataclass(frozen=True)
class Statistics: center: float; deviation: float; count: int


class StatisticsCalculator(LoggingMixin):
    def __init__(self, *args, neighbors, **kwargs):
        super().__init__(*args, **kwargs)
        self.__neighbors = int(neighbors)

    def statistics(self, t, k, w):
        t = np.asarray(t, dtype=float)
        k = np.asarray(k, dtype=float)
        w = np.asarray(w, dtype=float)
        if len(w) < 2: raise SizeError()
        if len(t) != len(w) or len(k) != len(w): raise SizeError()
        if not np.isfinite(np.column_stack([t, k, w])).all(): raise NumericError()
        t = self.independent(t)
        k = self.independent(k)
        tk = np.column_stack([t, k])
        tree = cKDTree(tk)
        n = min(self.neighbors, len(w) - 1)
        _, indexes = tree.query(tk, k=n + 1)
        for index, neighbors in enumerate(indexes):
            neighbors = neighbors[neighbors != index][:n]
            values = w[neighbors]
            center = float(np.median(values))
            deviation = float(1.4826 * np.median(np.abs(values - center)))
            statistics = Statistics(center=center, deviation=deviation, count=len(values))
            yield statistics

    @staticmethod
    def independent(x):
        x = np.asarray(x, dtype=float)
        center = np.median(x)
        distance = np.abs(x - center)
        scale = 1.4826 * np.median(distance)
        if scale <= 1e-12:
            unique = np.unique(x)
            if len(unique) > 1: scale = np.median(np.diff(unique))
            else: scale = 1.0
        return (x - center) / scale

    @staticmethod
    def dependent(y):
        y = np.asarray(y, dtype=float)
        center = np.median(y)
        distance = np.abs(y - center)
        scale = 1.4826 * np.median(distance)
        return scale + 1e-12

    @property
    def neighbors(self): return self.__neighbors


class VarianceCalculator(DataframeEquation, LoggingMixin):
    tau = lambda expire, today: pd.to_datetime(expire).apply(lambda ending: len(nyse.valid_days(start_date=pd.Timestamp(today) + pd.Timedelta(days=1), end_date=ending))) / 252
    mae = lambda forward, strike: np.log(forward / strike.astype(float))
    tiv = lambda implied, tau: tau * np.square(implied)

    @Pipelines.Function(locator=0)
    @LoggingDecorators.Function(Collection[Variables.Ticker], Collection[Variables.Expire], title="Calculated")
    def __call__(self, options, /, **kwargs):
        if options is None: return None
        assert isinstance(options, pd.DataFrame)
        if options.empty: return None
        variance = self.execute(options, **kwargs)
        options = pd.concat([options, variance], axis=1)
        return options


class VarianceScreener(StatisticsCalculator):
    def __init__(self, *args, threshold=5.0, floor=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.__threshold = float(threshold)
        self.__floor = float(floor)

    @Pipelines.Function(locator=0)
    @LoggingDecorators.Function(Collection[Variables.Ticker], Collection[Variables.Expire], title="Screened")
    def __call__(self, options, /, **kwargs):
        if options is None: return None
        assert isinstance(options, pd.DataFrame)
        if options.empty: return None
        screened = self.screener(options, **kwargs)
        usable = int(np.isfinite(screened["tiv"].to_numpy(dtype=float)).sum())
        self.console(options, usable)
        if not usable: return None
        return screened

    def screener(self, options, /, **kwargs):
        tkw = options[["tau", "mae", "tiv"]].to_numpy(dtype=float)
        valid = np.isfinite(tkw).all(axis=1)
        index = np.flatnonzero(valid)
        if len(index) == 0: return None

        t = options.iloc[index].to_numpy(dtype=float)
        k = options.iloc[index].to_numpy(dtype=float)
        w = options.iloc[index].to_numpy(dtype=float)
        scores = self.scores(t, k, w)
        scores = np.fromiter(scores, dtype=np.float64, count=len(w))

        screened = options.copy()
        rejected = (~np.isfinite(scores) | (np.abs(scores) > self.threshold))
        screened["outlier"] = np.nan
        column = screened.columns.get_loc("outlier")
        screened.iloc[index, column] = scores

        column = screened.columns.get_loc("tiv")
        screened.iloc[~valid, column] = np.nan
        screened.iloc[index[rejected], column] = np.nan
        return screened

    def scores(self, t, k, w):
        statistics = self.statistics(t, k, w)
        for value, statistics in zip(w, statistics):
            scale = max(statistics.deviation, self.floor * abs(statistics.center), 1e-8)
            yield (value - statistics.center) / scale

    def console(self, options, usable):
        tickers = Collection[Variables.Ticker](options)
        expires = Collection[Variables.Expire](options)
        sizes = f"{len(options):.0f}->{int(usable):.0f}"
        percents = f"{(int(usable) / len(options)) * 100:.0f}%"
        super().console(tickers, expires, sizes, percents, title="Screened")

    @property
    def threshold(self): return self.__threshold
    @property
    def floor(self): return self.__floor


class VarianceStandardizer(StatisticsCalculator):
    @Pipelines.Function(locator=0)
    @LoggingDecorators.Function(Collection[Variables.Ticker], Collection[Variables.Expire], title="Standardized")
    def __call__(self, options, surface, /, **kwargs):
        if options is None or surface is None: return None
        assert isinstance(options, pd.DataFrame)
        if options.empty: return None
        standard = self.standardize(options, surface, **kwargs)
        standard = pd.Series(standard, name="zscore", index=options.index)
        options = pd.concat([options, standard], axis=1)
        return options

    def standardize(self, options, surface, /, **kwargs):
        t = options["tau"].to_numpy(dtype=float)
        k = options["mae"].to_numpy(dtype=float)
        w = options["tiv"].to_numpy(dtype=float)
        μ = np.vectorize(surface)(t, k)
        σ = self.deviations(t, k, w)
        σ = np.fromiter(σ, dtype=np.float64)
        ε = np.quantile(σ[σ > 0], 0.1) if np.any(σ > 0) else 1e-8
        z = (w - μ) / np.maximum(σ, ε)
        return z

    def deviations(self, t, k, w):
        for statistics in self.statistics(t, k, w):
            yield statistics.deviation + 1e-12


