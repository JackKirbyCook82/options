# -*- coding: utf-8 -*-
"""
Created on Mon Jul 6 2026
@name:   Option Acquisition Objects
@author: Jack Kirby Cook
@file:   options/acquisitions.py

"""

import math
import numpy as np
from functools import cached_property
from dataclasses import dataclass, astuple

from options.targets import Target, Calculator
from finance.enumerations import Intent

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["AcquisitionCalculator", "AcquisitionMetrics", "AcquisitionTargets", "AcquisitionWeights", "AcquisitionPriority"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True, slots=True)
class Measure: zspread: float; multiple: float; ratio: float

@dataclass(frozen=True, slots=True)
class Priority: targets: Measure; weights: Measure

@dataclass(frozen=True, slots=True)
class Metrics(Measure):
    def __post_init__(self):
        assert not np.isnan(self.zspread) and self.zspread > 0
        assert not np.isnan(self.multiple) and self.multiple > 0
        assert not np.isnan(self.ratio) and self.ratio > 0


class AcquisitionTargets(Metrics): pass
class AcquisitionWeights(Metrics): pass
class AcquisitionMetrics(Metrics):
    def __call__(self, prospect):
        assert isinstance(prospect, Acquisition)
        if not np.isfinite(prospect.zspread): return False
        if not np.isfinite(prospect.multiple): return False
        if not np.isfinite(prospect.ratio): return False
        if prospect.zspread <= self.zspread: return False
        if prospect.multiple <= self.multiple: return False
        if prospect.ratio <= self.ratio: return False
        return True


class AcquisitionPriority(Priority):
    def __call__(self, prospect):
        assert isinstance(prospect, Acquisition)
        values = Measure(zspread=prospect.zspread, multiple=prospect.multiple, ratio=prospect.ratio)
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


class AcquisitionCalculator(Calculator, target=Acquisition):
    pass



