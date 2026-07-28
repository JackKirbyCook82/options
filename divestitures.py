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
from finance.enumerations import Spread, Position, Intent, Action
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


class Divestiture(Prospect):
    @property
    def commissions(self): return self.costing.commissions * self.quantities.sum()
    @property
    def slippage(self): return max(self.liquidate - self.market, self.slippage.exit * self.gap)
    @property
    def intent(self): return Intent.CLOSE

    @property
    def forecasted(self): return (self.past.revenue + self.future.revenue) / (self.past.expense + self.future.expense + self.cost + self.fees) - 1
    @property
    def realized(self): return (self.past.revenue + self.spot.revenue) / (self.past.expense + self.spot.expense + self.cost + self.fees) - 1
    @property
    def marginal(self): return (self.spot.revenue + self.future.revenue) / (self.spot.expense + self.future.expense + self.cost + self.fees) - 1
    @property
    def available(self): return self.forecasted - self.realized

    @cached_property
    def entry(self): return (self.securities["entry"] * self.positions.map(int) * self.quantities).sum()
    @cached_property
    def fees(self): return (self.costing.commissions * self.quantities).sum()

    @cached_property
    def liquidate(self):
        positions = self.positions.map(int)
        selling = (self.securities["bid"] * ((positions + 1) / 2) * self.quantities).sum()
        buying = (self.securities["ask"] * ((positions - 1) / 2) * self.quantities).sum()
        return selling - buying

    @cached_property
    def future(self): return self.cashflow(self.forecast, action=Action.SELL)
    @cached_property
    def spot(self): return self.cashflow(self.market, action=Action.SELL)
    @cached_property
    def past(self): return self.cashflow(self.entry, action=Action.BUY)


class DivestitureCreators(object):
    def __new__(cls, *args, spreads, **kwargs):
        spreads = [spread for spread in spreads if spread != Spread.EMPTY]
        instances = {spread: DivestitureCreator[spread](*args, **kwargs) for spread in spreads}
        return instances


class DivestitureCreator(ABC, metaclass=RegistryMeta):
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


class FlyDivestitureCreator(DivestitureCreator, register=Spread.FLY):
    def validator(self, holding):
        if len(holding) != 3: return False
        if holding["ticker"].nunique(dropna=False) != 1: return False
        if holding["expire"].nunique(dropna=False) != 1: return False
        if holding["option"].nunique(dropna=False) != 1: return False
        if holding["strike"].nunique(dropna=False) != 3: return False
        holding = holding.sort_values("strike")
        positions = holding["position"].map(int).to_numpy()
        quantities = holding["quantity"].astype(float).to_numpy()
        if positions * quantities != 0: return False
        if Position.EMPTY in positions: return False
        if positions[0] != positions[2]: return False
        if positions[0] == positions[1]: return False
        if positions[1] == positions[2]: return False
        return True

    def creator(self, holding):
        securities = holding.sort_values("strike").reset_index(drop=True).copy()
        securities["spread"] = Spread.FLY
        prospect = Divestiture(Spread.FLY, securities)
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
        if positions * quantities != 0: return False
        if Position.EMPTY in positions: return False
        if positions[0] != positions[1]: return False
        return True

    def creator(self, holding):
        securities = (holding.sort_values("expire").reset_index(drop=True).copy())
        securities["spread"] = Spread.CALENDAR
        prospect = Divestiture(Spread.CALENDAR, securities)
        return prospect



