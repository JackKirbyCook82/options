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
__all__ = ["OptionCalculator", "SanityFilter", "ViabilityFilter", "ViabilityMetric"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True, slots=True)
class Measure:
    moneyness: float | NumberRange
    tightness: float | NumberRange
    activity: float | NumberRange

class Metric(Measure):
    def __post_init__(self):
        assert self.moneyness > 0
        assert self.tightness > 0
        assert self.activity > 0

    def __call__(self, measure):
        assert isinstance(measure, Measure)
        if abs(measure.moneyness) > self.moneyness: return False
        if measure.tightness > self.tightness: return False
        if measure.activity < self.activity: return False
        return True


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
        self.scope(options, instrument=Instrument.OPTION)
        filtered = options.where(sanity["sanity"]).dropna(how="all", inplace=False)
        size = (len(options.index), len(filtered.index))
        self.results(scope=scope, size=size, title="Filtered")
        return filtered


class ViabilityMetric(Metric): pass
class ViabilityFilter(Logging, Equations, parameters={"tight": None, "money": None, "active": None}):
    viability = lambda moneyed, tightened, activated: np.logical_and.reduce([moneyed, tightened, activated])
    tightened = lambda tightness, *, tight: tightness <= float(tight) if tight is not None else pd.Series(True, index=tightness.index)
    moneyed = lambda moneyness, *, money: abs(moneyness) <= float(money) if money is not None else pd.Series(True, index=moneyness.index)
    activated = lambda activity, *, active: activity >= float(active) if active is not None else pd.Series(True, index=activity.index)

    def __init__(self, *args, metric, **kwargs):
        parameters = dict(money=metric.moneyness, tight=metric.tightness, active=metric.activity)
        super().__init__(*args, **parameters, **kwargs)
        self.__metric = metric

    def __call__(self, options, **kwargs):
        assert isinstance(options, pd.DataFrame)
        scope = self.scope(options, instrument=Instrument.OPTION)
        viability = self.execute(options, **kwargs)
        viable = options.where(viability["viability"]).dropna(how="all", inplace=False)
        size = (len(options.index), len(viable.index))
        strings = self.breakdown(options, viability)
        self.results(scope=scope, size=size, strings=strings, title="Filtered")
        return viable

    def breakdown(self, options, viability):
        boundary = self.boundary(options)
        survival = self.survival(viability)
        moneyness = f"Moneyness<={self.metric.moneyness:.1f}[{survival.moneyness:.0f}%, f{boundary.moneyness.minimum:.1f}->f{boundary.moneyness.maximum:.1f}]"
        tightness = f"Tightness<={self.metric.tightness:.1f}[{survival.tightness:.0f}%, f{boundary.tightness.minimum:.1f}->f{boundary.tightness.maximum:.1f}]"
        activity = f"Activity>={self.metric.activity:.1f}[{survival.activity:.0f}%, f{boundary.activity.minimum:.1f}->f{boundary.activity.maximum:.1f}]"
        return [moneyness, tightness, activity]

    @staticmethod
    def survival(viability):
        tightness = (viability['tightened']).sum() / len(viability.index) * 100
        moneyness = (viability['moneyed']).sum() / len(viability.index) * 100
        activity = (viability['activated']).sum() / len(viability.index) * 100
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



