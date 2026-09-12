# -*- coding: utf-8 -*-
"""
Created on Mon Jul 6 2026
@name:   Option Divestiture Objects
@author: Jack Kirby Cook
@file:   options/divestitures.py

"""

import math
from types import SimpleNamespace
from functools import cached_property
from dataclasses import dataclass, astuple

from options.targets import Target
from options.prospects import Prospect
from finance.enumerations import Instrument, Intent, Action
from finance.logging import Logging
from support.custom import NumberRange

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DivestitureCalculator", "DivestitureMetrics", "DivestitureTargets", "DivestitureWeights", "DivestiturePriority"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True, slots=True)
class Quantative:
    forecasted: float; capturable: float

#    def __float__(self):
#        return self.capturable / self.forecasted


@dataclass(frozen=True, slots=True)
class Measure: multiple: float; ratio: float

@dataclass(frozen=True, slots=True)
class Metrics(Measure):
    eager: bool

#    def __post_init__(self):
#        assert 0 < self.multiple < 1
#        assert 0 < self.ratio < 1

#    def __call__(self, prospect):
#        assert isinstance(prospect, Divestiture)
#        multiple = float(prospect.multiple) >= self.multiple
#        ratio = float(prospect.ratio) >= self.ratio
#        if bool(self.eager): return multiple or ratio
#        else: return multiple and ratio


@dataclass(frozen=True, slots=True)
class Priority:
    targets: Measure; weights: Measure

#    def __call__(self, prospect):
#        assert isinstance(prospect, Divestiture)
#        values = Measure(multiple=float(prospect.multiple), ratio=float(prospect.ratio))
#        weights, total = astuple(self.weights), sum(astuple(self.weights))
#        weights = (weight / total for weight in weights)
#        generator = zip(astuple(values), astuple(self.targets), weights)
#        function = lambda value, target, weight: weight * math.log(max(value / (value + target), 1e-12))
#        return math.exp(sum([function(*arguments) for arguments in generator]))


class Divestiture(Target, columns="entry"):
    @cached_property
    def entry(self): return (self.securities["entry"] * self.positions.map(int) * self.quantities).sum()

    @cached_property
    def slippage(self): return max(self.liquidate, self.costing.slippage.exit * self.gap)
    @cached_property
    def commissions(self): return self.costing.commissions * self.quantities.sum()
    @cached_property
    def fees(self): return self.costing.commissions * self.quantities.sum()
    @cached_property
    def intent(self): return Intent.CLOSE

    @cached_property
    def multiple(self):
        forecasted = self.edge.forecasted / max(self.cost, 1e-10)
        capturable = self.edge.capturable / self.cost
        return Quantative(forecasted=forecasted, capturable=capturable)

    @cached_property
    def ratio(self):
        forecasted = self.pnl.forecasted / max(self.var, 1e-10)
        capturable = self.pnl.capturable / max(self.var, 1e-10)
        return Quantative(forecasted=forecasted, capturable=capturable)

    @cached_property
    def edge(self):
        forecasted = self.forecast - self.entry
        capturable = self.market - self.entry
        return Quantative(forecasted=forecasted, capturable=capturable)

    @cached_property
    def pnl(self):
        forecasted = self.edge.forecasted - self.cost - self.fees
        capturable = self.edge.capturable - self.cost - self.fees
        return Quantative(forecasted=forecasted, capturable=capturable)


class DivestitureMetrics(Metrics): pass
class DivestitureTargets(Measure): pass
class DivestitureWeights(Measure): pass
class DivestiturePriority(Priority): pass
class DivestitureCalculator(Logging):
    def __init__(self, *args, metrics, priority, costing, **kwargs):
        super().__init__(*args, **kwargs)
        self.__priority = priority
        self.__metrics = metrics
        self.__costing = costing

    def __call__(self, prospects, **kwargs):
        assert isinstance(prospects, list) and all([isinstance(prospect, Prospect) for prospect in prospects])
        scope = self.scope(prospects, instrument=Instrument.SPREAD)
        targets = [Divestiture.create(prospect, costing=self.costing) for prospect in prospects]
        divestitures = [target for target in targets if self.metrics(target)]
        divestitures.sort(key=self.priority, reverse=True)
        size = (len(targets), len(divestitures))
        strings = self.breakdown(targets) if bool(targets) else []
        self.results(scope=scope, size=size, strings=strings, title="Calculated")
        return divestitures

#    def breakdown(self, targets):
#        boundary = self.boundary(targets)
#        survival = self.survival(targets)
#        multiple = f"Multiple >= {self.metrics.multiple:.2f} [{boundary.multiples.minimum:+.2f} -> {boundary.multiples.maximum:+.2f}, {survival.multiples:.0f}%]"
#        ratio = f"Ratio >= {self.metrics.ratio:.2f} [{boundary.ratios.minimum:+.2f} -> {boundary.ratios.maximum:+.2f}, {survival.ratios:.0f}%]"
#        return [multiple, ratio]

#    def survival(self, targets):
#        multiples = [float(target.multiple) >= self.metrics.multiple for target in targets]
#        ratios = [float(target.ratio) >= self.metrics.ratio for target in targets]
#        multiples = sum(multiples) / len(multiples) * 100
#        ratios = sum(ratios) / len(ratios) * 100
#        return SimpleNamespace(multiples=multiples, ratios=ratios)

#    @staticmethod
#    def boundary(targets):
#        multiples = [float(target.multiple) for target in targets]
#        ratios = [float(target.ratio) for target in targets]
#        multiples = NumberRange([min(multiples), max(multiples)])
#        ratios = NumberRange([min(ratios), max(ratios)])
#        return SimpleNamespace(multiples=multiples, ratios=ratios)

    @property
    def priority(self): return self.__priority
    @property
    def metrics(self): return self.__metrics
    @property
    def costing(self): return self.__costing





