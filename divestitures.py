# -*- coding: utf-8 -*-
"""
Created on Mon Jul 6 2026
@name:   Option Divestiture Objects
@author: Jack Kirby Cook
@file:   options/divestitures.py

"""

from dataclasses import dataclass
from functools import cached_property

from options.targets import Target, Calculator
from finance.enumerations import Intent, Action

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DivestitureCalculator", "DivestitureMetrics", "DivestiturePriority"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True, slots=True)
class Measure: pass

@dataclass(frozen=True, slots=True)
class Priority: pass

@dataclass(frozen=True, slots=True)
class Metrics(Measure): pass


class DivestitureMetrics(Metrics):
    def __call__(self, prospect):
        assert isinstance(prospect, Divestiture)


class DivestiturePriority(Priority):
    def __call__(self, prospect):
        assert isinstance(prospect, Divestiture)


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
    def liquidate(self):
        positions = self.positions.map(int).astype(int)
        quantities = self.quantities.astype(float)
        actions = positions * int(self.intent)
        mask = actions.eq(int(Action.BUY))
        prices = self.securities["ask"].where(mask, self.securities["bid"])
        return abs((prices * positions * quantities).sum() - self.market)


class DivestitureCalculator(Calculator, target=Divestiture):
    pass



