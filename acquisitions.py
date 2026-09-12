# -*- coding: utf-8 -*-
"""
Created on Mon Jul 6 2026
@name:   Option Acquisition Objects
@author: Jack Kirby Cook
@file:   options/acquisitions.py

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
__all__ = ["AcquisitionCalculator", "AcquisitionMetrics", "AcquisitionTargets", "AcquisitionWeights", "AcquisitionPriority"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True, slots=True)
class Measure: zspread: float; multiple: float; ratio: float

@dataclass(frozen=True, slots=True)
class Metrics(Measure):
    def __post_init__(self):
        assert self.zspread > 0
        assert self.multiple > 0
        assert self.ratio > 0

    def __call__(self, prospect):
        assert isinstance(prospect, Acquisition)
        if abs(prospect.zspread) <= self.zspread: return False
        if prospect.multiple <= self.multiple: return False
        if prospect.ratio <= self.ratio: return False
        return True


@dataclass(frozen=True, slots=True)
class Priority:
    targets: Measure; weights: Measure

    def __call__(self, prospect):
        assert isinstance(prospect, Acquisition)
        values = Measure(zspread=abs(prospect.zspread), multiple=prospect.multiple, ratio=prospect.ratio)
        weights, total = astuple(self.weights), sum(astuple(self.weights))
        weights = (weight / total for weight in weights)
        generator = zip(astuple(values), astuple(self.targets), weights)
        function = lambda value, target, weight: weight * math.log(max(value / (value + target), 1e-12))
        return math.exp(sum([function(*arguments) for arguments in generator]))


class Acquisition(Target):
    @cached_property
    def slippage(self): return (self.costing.slippage.entry + self.costing.slippage.exit) * self.gap
    @cached_property
    def commissions(self): return self.costing.commissions * self.quantities.sum() * 2
    @cached_property
    def intent(self): return Intent.OPEN

    @cached_property
    def multiple(self): return self.edge / self.cost
    @cached_property
    def ratio(self): return self.pnl / self.var

    @cached_property
    def edge(self): return self.forecast - self.market
    @cached_property
    def pnl(self): return self.edge - self.cost


class AcquisitionMetrics(Metrics): pass
class AcquisitionTargets(Metrics): pass
class AcquisitionWeights(Metrics): pass
class AcquisitionPriority(Priority): pass
class AcquisitionCalculator(Logging):
    def __init__(self, *args, metrics, priority, costing, **kwargs):
        super().__init__(*args, **kwargs)
        self.__priority = priority
        self.__metrics = metrics
        self.__costing = costing

    def __call__(self, prospects, /, **kwargs):
        assert isinstance(prospects, list) and all([isinstance(prospect, Prospect) for prospect in prospects])
        scope = self.scope(prospects, instrument=Instrument.SPREAD)
        targets = [Acquisition.create(prospect, costing=self.costing) for prospect in prospects]
        acquisitions = [target for target in targets if self.metrics(target)]
        acquisitions.sort(key=self.priority, reverse=True)
        size = (len(targets), len(acquisitions))
        strings = self.breakdown(targets) if bool(targets) else []
        self.results(scope=scope, size=size, strings=strings, title="Calculated")
        return acquisitions

    def breakdown(self, targets):
        boundary = self.boundary(targets)
        survival = self.survival(targets)
        zspread = f"|ZSpread| >= {self.metrics.zspread:.2f} [{boundary.zspreads.minimum:+.2f} -> {boundary.zspreads.maximum:+.2f}, {survival.zspreads:.0f}%]"
        multiple = f"Multiple >= {self.metrics.multiple:.2f} [{boundary.multiples.minimum:+.2f} -> {boundary.multiples.maximum:+.2f}, {survival.multiples:.0f}%]"
        ratio = f"Ratio >= {self.metrics.ratio:.2f} [{boundary.ratios.minimum:+.2f} -> {boundary.ratios.maximum:+.2f}, {survival.ratios:.0f}%]"
        return [zspread, multiple, ratio]

    def survival(self, targets):
        zspreads = [target.zspread >= self.metrics.zspread for target in targets]
        multiples = [target.multiple >= self.metrics.multiple for target in targets]
        ratios = [target.ratio >= self.metrics.ratio for target in targets]
        zspreads = sum(zspreads) / len(zspreads) * 100
        multiples = sum(multiples) / len(multiples) * 100
        ratios = sum(ratios) / len(ratios) * 100
        return SimpleNamespace(zspreads=zspreads, multiples=multiples, ratios=ratios)

    @staticmethod
    def boundary(targets):
        zspreads = [target.zspread for target in targets]
        multiples = [target.multiple for target in targets]
        ratios = [target.ratio for target in targets]
        zspreads = NumberRange([min(zspreads), max(zspreads)])
        multiples = NumberRange([min(multiples), max(multiples)])
        ratios = NumberRange([min(ratios), max(ratios)])
        return SimpleNamespace(zspreads=zspreads, multiples=multiples, ratios=ratios)

    @property
    def priority(self): return self.__priority
    @property
    def metrics(self): return self.__metrics
    @property
    def costing(self): return self.__costing



