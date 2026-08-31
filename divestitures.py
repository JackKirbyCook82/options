# -*- coding: utf-8 -*-
"""
Created on Mon Jul 6 2026
@name:   Option Divestiture Objects
@author: Jack Kirby Cook
@file:   options/divestitures.py

"""

import math
import pandas as pd
from typing import Optional
from functools import cached_property
from dataclasses import dataclass, astuple

from options.prospects import Prospect
from finance.enumerations import Intent, Instrument
from finance.logging import Logging
from pandas._config.config import Mode

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DivestitureCalculator", "DivestitureMetric"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True, slots=True)
class Quantative:
    forecasted: float
    capturable: float
    remaining: float

@dataclass(frozen=True, slots=True)
class Measure:
    multiple: float | Quantative
    ratio: float | Quantative
    mode: Optional[str] = None

class Metric(Measure):
    def __post_init__(self):
        assert self.mode in ("ALL", "ANY", None)
        assert 0 < self.multiple < 1
        assert 0 < self.ratio < 1

    def __call__(self, measure):
        assert isinstance(measure, Measure)
        percentage = lambda quantative: quantative.capturable / quantative.forecasted
        multiple = percentage(measure.multiple) <= self.multiple
        ratio = percentage(measure.ratio) <= self.ratio
        if self.mode == Mode.ALL: return multiple and ratio
        elif self.mode == Mode.ANY: return multiple or ratio
        else: raise ValueError(self.mode)


@dataclass(frozen=True, slots=True)
class Priority:
    targets: Measure; weights: Measure

    def __call__(self, divestiture):
        assert isinstance(divestiture, Divestiture)
        percentage = lambda quantative: quantative.capturable / quantative.forecasted
        multiple = percentage(divestiture.multiple)
        ratio = percentage(divestiture.ratio)
        values = Measure(multiple=multiple, ratio=ratio)
        weights, total = astuple(self.weights), sum(astuple(self.weights))
        weights = (weight / total for weight in weights)
        generator = zip(astuple(values), astuple(self.targets), weights)
        function = lambda value, target, weight: weight * math.log(max(value / (value + target), 1e-12))
        return math.exp(sum([function(*arguments) for arguments in generator]))


class Divestiture(Prospect):
    @property
    def slippage(self): return max(self.liquidate, self.costing.slippage.exit * self.gap)
    @property
    def commissions(self): return self.costing.commissions * self.quantities.sum()
    @property
    def intent(self): return Intent.CLOSE

    @property
    def measure(self): return Measure(self.multiple, self.ratio)
    @property
    def priority(self):
        targets = Measure(multiple=1.00, ratio=1.00)
        weights = Measure(multiple=0.45, ratio=0.55)
        priority = Priority(targets=targets, weights=weights)
        return priority(self)

    @cached_property
    def entry(self): return (self.securities["entry"] * self.positions.map(int) * self.quantities).sum()
    @cached_property
    def fees(self): return self.costing.commissions * self.quantities.sum()

    @cached_property
    def multiple(self):
        forecasted = self.edge.forecasted / self.cost
        capturable = self.edge.capturable / self.cost
        remaining = self.edge.remaining / self.cost
        return Quantative(forecasted=forecasted, capturable=capturable, remaining=remaining)

    @cached_property
    def ratio(self):
        forecasted = self.pnl.forecasted / self.var
        capturable = self.pnl.capturable / self.var
        remaining = self.pnl.remaining / self.var
        return Quantative(forecasted=forecasted, capturable=capturable, remaining=remaining)

    @cached_property
    def edge(self):
        forecasted = self.forecast - self.entry
        capturable = self.market - self.entry
        remaining = forecasted - capturable
        return Quantative(forecasted=forecasted, capturable=capturable, remaining=remaining)

    @cached_property
    def pnl(self):
        forecasted = self.edge.forecasted - self.cost - self.fees
        capturable = self.edge.capturable - self.cost - self.fees
        remaining = forecasted - capturable
        return Quantative(forecasted=forecasted, capturable=capturable, remaining=remaining)


class DivestitureMetric(Metric): pass
class DivestitureCalculator(Logging):
    def __init__(self, *args, metric, **kwargs):
        self.__metric = metric

    def __call__(self, holdings, /, **kwargs):
        assert isinstance(holdings, pd.DataFrame)
        scope = self.scope(holdings, instrument=Instrument.OPTION)
        prospects = [Divestiture(spread, securities) for (order, spread), securities in holdings.groupby(["order", "spread"])]
        divestitures = [prospect for prospect in prospects if self.metric(prospect.measure)]
        divestitures.sort(key=lambda prospect: prospect.priority, reverse=True)
        size = (len(prospects), len(divestitures))
        strings = self.breakdown(prospects) if bool(prospects) else []
        self.results(scope=scope, size=size, strings=strings, title="Calculated")
        return divestitures

    @property
    def metric(self): return self.__metric





