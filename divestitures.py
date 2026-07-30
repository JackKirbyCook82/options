# -*- coding: utf-8 -*-
"""
Created on Mon Jul 6 2026
@name:   Option Divestiture Objects
@author: Jack Kirby Cook

"""

import pandas as pd
from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import cached_property

from options.prospects import Prospect
from finance.enumerations import Spread, Position, Intent
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DivestitureCreators"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True, slots=True)
class Metrics: pass

@dataclass(frozen=True, slots=True)
class Priority: pass

@dataclass(frozen=True, slots=True)
class PnL: forecasted: float; realizable: float; opportunity: float


class Divestiture(Prospect):
    @property
    def slippage(self): return max(self.premium, self.costing.slippage.exit * self.gap)
    @property
    def commissions(self): return self.costing.commissions * self.quantities.sum()
    @property
    def intent(self): return Intent.CLOSE

    @cached_property
    def entry(self): return (self.securities["entry"] * self.positions.map(int) * self.quantities).sum()
    @property
    def fees(self): return self.costing.commissions * self.quantities.sum()

    @property
    def pnl(self):
        forecasted = self.forecast - self.entry - self.cost - self.fees
        realizable = self.market - self.entry - self.cost - self.fees
        opportunity = forecasted - realizable
        return PnL(forecasted=forecasted, realizable=realizable, opportunity=opportunity)


class DivestitureCreators(object):
    def __new__(cls, *args, spreads, **kwargs):
        spreads = [spread for spread in spreads if spread != Spread.EMPTY]
        instances = {spread: DivestitureCreator[spread](*args, **kwargs) for spread in spreads}
        return instances


class DivestitureCreator(ABC, metaclass=RegistryMeta):
    def __init__(self, *args, scenario, costing, **kwargs):
        self.__scenario = scenario
        self.__costing = costing

    def __call__(self, holdings, /, **kwargs):
        assert isinstance(holdings, pd.DataFrame)
        for order, holding in holdings.groupby("order"):
            valid = self.validator(holding)
            if not valid: continue
            prospect = self.creator(holding)
            yield prospect

    @staticmethod
    @abstractmethod
    def validator(holding): pass
    @staticmethod
    @abstractmethod
    def creator(holding): pass

    @property
    def scenario(self): return self.__scenario
    @property
    def costing(self): return self.__costing


class FlyDivestitureCreator(DivestitureCreator, register=Spread.FLY):
    def validator(self, holding):
        if len(holding) != 3: return False
        if holding["ticker"].nunique(dropna=False) != 1: return False
        if holding["expire"].nunique(dropna=False) != 1: return False
        if holding["option"].nunique(dropna=False) != 1: return False
        if holding["strike"].nunique(dropna=False) != 3: return False
        holding = holding.sort_values("strike")
        positions = holding["position"].map(int).to_numpy()
        quantities = holding["quantity"].astype(int).to_numpy()
        if (positions == int(Position.EMPTY)).any(): return False
        if (quantities <= 0).any(): return False
        if not (positions[0] == positions[2] and positions[1] == positions[0] * -1): return False
        if not (quantities[0] == quantities[2] and quantities[1] == quantities[0] * +2): return False
        return True

    def creator(self, holding):
        securities = holding.sort_values("strike").reset_index(drop=True).copy()
        securities["spread"] = Spread.FLY
        parameters = dict(scenario=self.scenario, costing=self.costing)
        prospect = Divestiture(Spread.FLY, securities, **parameters)
        return prospect


class CalendarDivestitureCreator(DivestitureCreator, register=Spread.CALENDAR):
    def validator(self, holding):
        if len(holding) != 2: return False
        if holding["ticker"].nunique(dropna=False) != 1: return False
        if holding["expire"].nunique(dropna=False) != 2: return False
        if holding["option"].nunique(dropna=False) != 1: return False
        if holding["strike"].nunique(dropna=False) != 1: return False
        holding = holding.sort_values("expire")
        positions = holding["position"].map(int).to_numpy()
        quantities = holding["quantity"].astype(float).to_numpy()
        if positions[0] != positions[1] * -1: return False
        if quantities[0] != quantities[1] * +1: return False
        return True

    def creator(self, holding):
        securities = (holding.sort_values("expire").reset_index(drop=True).copy())
        securities["spread"] = Spread.CALENDAR
        parameters = dict(scenario=self.scenario, costing=self.costing)
        prospect = Divestiture(Spread.CALENDAR, securities, **parameters)
        return prospect



