# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 2026
@name:   Option Objects
@author: Jack Kirby Cook
@file:   options/__init__.py

"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import date as Date
import pandas_market_calendars as calenders

from finance.enumerations import Instrument
from finance.reporting import Results, Analysis
from support.equations import Equations
from support.custom import NumberRange
from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["OptionCalculator", "SanityFilter", "ViabilityFilter", "ViabilityMetrics"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


nyse = calenders.get_calendar("NYSE")


@dataclass(frozen=True, slots=True)
class Measure:
    moneyness: float | NumberRange
    tightness: float | NumberRange
    activity: float | NumberRange

class Metrics(Measure):
    def __post_init__(self):
        assert self.moneyness is not None and self.moneyness > 0
        assert self.tightness is not None and self.tightness > 0
        assert self.activity is not None and self.activity > 0


class OptionCalculator(Results, Logging, Equations, variables=["moneyness", "tightness", "activity", "market", "gap", "trading", "calender"]):
    trading = lambda expire: pd.to_datetime(expire).apply(lambda ending: len(nyse.valid_days(start_date=pd.Timestamp(Date.today()) + pd.Timedelta(days=1), end_date=ending)))
    business = lambda expire: np.busday_count(np.datetime64(Date.today()) + np.timedelta64(1, "D"), pd.to_datetime(expire).values.astype("datetime64[D]") + np.timedelta64(1, "D"))
    calender = lambda expire: (pd.to_datetime(expire) - pd.Timestamp(Date.today())).dt.days
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
        results = self.results(scope=scope, size=len(options))
        self.console("Calculated", results)
        return options


class SanityFilter(Results, Logging, Equations, parameters={"size": 1}):
    sanity = lambda supplied, demanded, bided, asked, realistic: np.logical_and.reduce([supplied, demanded, bided, asked, realistic])
    supplied = lambda supply, *, size: supply.notna() & (supply >= size)
    demanded = lambda demand, *, size: demand.notna() & (demand >= size)
    bided = lambda bid: bid.notna() & np.isfinite(bid) & (bid >= 0)
    asked = lambda ask: ask.notna() & np.isfinite(ask) & (ask >= 0)
    realistic = lambda bid, ask: ask > bid

    def __call__(self, options, **kwargs):
        assert isinstance(options, pd.DataFrame)
        scope = self.scope(options, instrument=Instrument.OPTION)
        sanity = self.execute(options, **kwargs)["sanity"]
        filtered = options.where(sanity).dropna(how="all", inplace=False)
        size = (len(options.index), len(filtered.index))
        results = self.results(scope=scope, size=size)
        self.console("Filtered", results)
        return filtered


class ViabilityMetrics(Metrics): pass
class ViabilityFilter(Analysis.Viability, Results, Logging):
    def __init__(self, *args, metrics, **kwargs):
        super().__init__(*args, **kwargs)
        self.__metrics = metrics

    def __call__(self, options, **kwargs):
        assert isinstance(options, pd.DataFrame)
        scope = self.scope(options, instrument=Instrument.OPTION)
        viability = self.execute(options, **kwargs)
        filtered = options.where(viability).dropna(how="all", inplace=False)
        size = (len(options.index), len(filtered.index))
        results = self.results(scope=scope, size=size)
        analysis = self.analysis(options)
        self.console("Filtered", results, *analysis)
        return filtered

    def execute(self, options, **kwargs):
        moneyness = options["moneyness"].abs() <= self.metrics.moneyness
        tightness = options["tightness"] <= self.metrics.tightness
        activity = options["activity"] >= self.metrics.activity
        viability = moneyness & tightness & activity
        return viability

    @property
    def metrics(self): return self.__metrics



