# -*- coding: utf-8 -*-
"""
Created on Mon Jul 6 2026
@name:   Option Divestiture Objects
@author: Jack Kirby Cook
@file:   options/divestitures.py

"""

import math
from enum import Enum
from functools import cached_property
from dataclasses import dataclass, field, astuple

from options.prospects import Prospect
from finance.enumerations import Intent
from finance.logging import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DivestitureCalculator", "Weights", "Targets", "Metrics", "Priority", "Mode"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class ModeError(Exception): pass
class Mode: ANY, ALL = range(2)


@dataclass(frozen=True, slots=True)
class Quantative: forecasted: float; capturable: float; remaining: float


@dataclass(frozen=True, slots=True)
class Measures: multiple: float; ratio: float

class Weights(Measures): pass
class Targets(Measures): pass
class Metrics(Measures):
    mode: Enum

    def __post_init__(self):
        assert 0 < self.multiple < 1
        assert 0 < self.ratio < 1

    def __call__(self, divestiture):
        assert isinstance(divestiture, Divestiture)
        percentage = lambda quantative: quantative.capturable / quantative.forecasted
        multiple = percentage(divestiture.multiple) <= self.multiple
        ratio = percentage(divestiture.ratio) <= self.ratio
        if self.mode == Mode.ALL: return multiple and ratio
        elif self.mode == Mode.ANY: return multiple or ratio
        else: raise ModeError(self.mode)


@dataclass(frozen=True, slots=True)
class Priority:
    targets: Measures = field(default_factory=lambda: Measures(multiple=1.00, ratio=1.00))
    weights: Measures = field(default_factory=lambda: Measures(multiple=0.45, ratio=0.55))

    def __call__(self, divestiture):
        assert isinstance(divestiture, Divestiture)
        percentage = lambda quantative: quantative.capturable / quantative.forecasted
        multiple = percentage(divestiture.multiple)
        ratio = percentage(divestiture.ratio)
        values = Measures(multiple=multiple, ratio=ratio)
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


class DivestitureCalculator(Logging):
    pass


#    @cached_property
#    def pnl(self):
#        forecasted = self.forecast - self.entry - self.cost - self.fees
#        realizable = self.market - self.entry - self.cost - self.fees
#        opportunity = forecasted - realizable
#        return PnL(forecasted=forecasted, realizable=realizable, opportunity=opportunity)

#    @cached_property
#    def edge(self):
#        original = self.forecast - self.entry
#        captured = self.market - self.entry
#        remaining = self.forecast - self.market
#        return Edge(original=original, captured=captured, remaining=remaining)


# class DivestitureCreator(ABC, metaclass=RegistryMeta):
#     def __init__(self, *args, costing, **kwargs):
#         self.__costing = costing
#
#     def __call__(self, holdings, /, **kwargs):
#         assert isinstance(holdings, pd.DataFrame)
#         for order, holding in holdings.groupby("order"):
#             valid = self.validator(holding)
#             if not valid: continue
#             prospect = self.creator(holding)
#             yield prospect
#
#     @staticmethod
#     @abstractmethod
#     def validator(holding): pass
#     @staticmethod
#     @abstractmethod
#     def creator(holding): pass
#
#     @property
#     def costing(self): return self.__costing


# class FlyDivestitureCreator(DivestitureCreator, register=Spread.FLY):
#     def validator(self, holding):
#         if len(holding) != 3: return False
#         if holding["ticker"].nunique(dropna=False) != 1: return False
#         if holding["expire"].nunique(dropna=False) != 1: return False
#         if holding["option"].nunique(dropna=False) != 1: return False
#         if holding["strike"].nunique(dropna=False) != 3: return False
#         holding = holding.sort_values("strike")
#         positions = holding["position"].map(int).to_numpy()
#         quantities = holding["quantity"].astype(int).to_numpy()
#         if (positions == int(Position.EMPTY)).any(): return False
#         if (quantities <= 0).any(): return False
#         if not (positions[0] == positions[2] and positions[1] == positions[0] * -1): return False
#         if not (quantities[0] == quantities[2] and quantities[1] == quantities[0] * +2): return False
#         return True
#
#     def creator(self, holding):
#         securities = holding.sort_values("strike").reset_index(drop=True).copy()
#         securities["spread"] = Spread.FLY
#         parameters = dict(costing=self.costing)
#         prospect = Divestiture(Spread.FLY, securities, **parameters)
#         return prospect


# class CalendarDivestitureCreator(DivestitureCreator, register=Spread.CALENDAR):
#     def validator(self, holding):
#         if len(holding) != 2: return False
#         if holding["ticker"].nunique(dropna=False) != 1: return False
#         if holding["expire"].nunique(dropna=False) != 2: return False
#         if holding["option"].nunique(dropna=False) != 1: return False
#         if holding["strike"].nunique(dropna=False) != 1: return False
#         holding = holding.sort_values("expire")
#         positions = holding["position"].map(int).to_numpy()
#         quantities = holding["quantity"].astype(float).to_numpy()
#         if positions[0] != positions[1] * -1: return False
#         if quantities[0] != quantities[1] * +1: return False
#         return True
#
#     def creator(self, holding):
#         securities = (holding.sort_values("expire").reset_index(drop=True).copy())
#         securities["spread"] = Spread.CALENDAR
#         parameters = dict(costing=self.costing)
#         prospect = Divestiture(Spread.CALENDAR, securities, **parameters)
#         return prospect
#



