# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 2026
@name:   Option Objects
@author: Jack Kirby Cook
@file:   options/__init__.py

"""

import numpy as np
import pandas as pd
from datetime import date as Date
from dataclasses import dataclass

from finance.enumerations import Instrument
from finance.logging import Logging
from support.equations import Equations
from support.custom import NumberRange

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["OptionCalculator", "SanityFilter", "ViabilityFilter", "ViabilityMetrics"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True, slots=True)
class Measure:
    moneyness: float | NumberRange
    tightness: float | NumberRange
    activity: float | NumberRange

class Metric(Measure):
    def __post_init__(self):
        assert self.moneyness is not None and self.moneyness > 0
        assert self.tightness is not None and self.tightness > 0
        assert self.activity is not None and self.activity > 0


class OptionCalculator(Logging, Equations, variables=["moneyness", "tightness", "activity", "market", "gap", "dte"]):
    dte = lambda expire: (pd.to_datetime(expire) - pd.Timestamp(Date.today())).dt.days
    moneyness = lambda underlying, strike, option: np.log(underlying / strike.astype(float)) * option.astype(int)
    activity = lambda supply, demand: np.minimum(supply, demand) / (np.maximum(supply, demand) + 10)
    tightness = lambda gap, market: gap / market
    mean = lambda bid, ask, supply, demand: ((bid * demand) + (ask * supply)) / (demand + supply)
    market = lambda bid, ask: (bid + ask) / 2
    gap = lambda bid, ask: ask - bid

    def __call__(self, options, **kwargs):
        assert isinstance(options, pd.DataFrame)
        scope = self.scope(options, instrument=Instrument.OPTION)
        calculated = self.execute(options, **kwargs)
        options = pd.concat([options, calculated], axis=1)
        self.results(scope=scope, size=len(options), title="Calculated")
        return options


class SanityFilter(Logging, Equations, parameters={"size": 1}):
    sanity = lambda supplied, demanded, bided, asked, realistic: np.logical_and.reduce([supplied, demanded, bided, asked, realistic])
    supplied = lambda supply, *, size: supply.notna() & (supply >= size)
    demanded = lambda demand, *, size: demand.notna() & (demand >= size)
    bided = lambda bid: bid.notna() & np.isfinite(bid) & (bid >= 0)
    asked = lambda ask: ask.notna() & np.isfinite(ask) & (ask >= 0)
    realistic = lambda bid, ask: ask > bid

    def __call__(self, options, **kwargs):
        assert isinstance(options, pd.DataFrame)
        scope = self.scope(options, instrument=Instrument.OPTION)
        sanity = self.execute(options, **kwargs).squeeze()
        filtered = options.where(sanity["sanity"]).dropna(how="all", inplace=False)
        size = (len(options.index), len(filtered.index))
        self.results(scope=scope, size=size, title="Filtered")
        return filtered


class ViabilityMetrics(Metric): pass
class ViabilityFilter(Logging):
    def __init__(self, *args, metric, **kwargs):
        parameters = dict(money=metric.moneyness, tight=metric.tightness, active=metric.activity)
        super().__init__(*args, **parameters, **kwargs)
        self.__metric = metric

    def __call__(self, options, **kwargs):
        assert isinstance(options, pd.DataFrame)
        scope = self.scope(options, instrument=Instrument.OPTION)
        viability = self.execute(options, **kwargs)
        filtered = options.where(viability["viability"]).dropna(how="all", inplace=False)
        size = (len(options.index), len(filtered.index))
        strings = self.breakdown(options)
        self.results(scope=scope, size=size, strings=strings, title="Filtered")
        return filtered

    def execute(self, options, **kwargs):
        moneyness = options["moneyness"].abs() <= self.metric.moneyness
        tightness = options["tightness"] <= self.metric.tightness
        activity = options["activity"] >= self.metric.activity
        viability = np.logical_and.reduce([moneyness, tightness, activity])
        return viability

    def breakdown(self, options):
        boundary = self.boundary(options)
        survival = self.survival(options)
        moneyness = f"|Moneyness| <= {self.metric.moneyness:.2f} [{boundary.moneyness.minimum:+.2f} -> {boundary.moneyness.maximum:+.2f}, {survival.moneyness:.0f}%]"
        tightness = f"Tightness <= {self.metric.tightness:.2f} [{boundary.tightness.minimum:+.2f} -> {boundary.tightness.maximum:+.2f}, {survival.tightness:.0f}%]"
        activity = f"Activity >= {self.metric.activity:.2f} [{boundary.activity.minimum:+.2f} -> {boundary.activity.maximum:+.2f}, {survival.activity:.0f}%]"
        return [moneyness, tightness, activity]

    def survival(self, options):
        moneyness = (options["moneyness"].abs() <= self.metric.moneyness).sum() / len(options.index) * 100
        tightness = (options["tightness"] <= self.metric.tightness).sum() / len(options.index) * 100
        activity = (options["activity"] <= self.metric.activity).sum() / len(options.index) * 100
        return Measure(moneyness, tightness, activity)

    @staticmethod
    def boundary(options):
        options = options[["moneyness", "tightness", "activity"]]
        moneyness = NumberRange(options["moneyness"].to_list())
        tightness = NumberRange(options["tightness"].to_list())
        activity = NumberRange(options["activity"].to_list())
        return Measure(moneyness, tightness, activity)

    @property
    def metric(self): return self.__metric



