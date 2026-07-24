# -*- coding: utf-8 -*-
"""
Created on Mon Jul 6 2026
@name:   Option Divestiture Objects
@author: Jack Kirby Cook

"""

import pandas as pd
from abc import ABC, abstractmethod

from options.prospects import Prospect
from finance.enumerations import Spread, Position
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DivestitureCreators"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class Divestiture(Prospect):
    @property
    def spent(self): return self.securities["spent"].sum()
    @property
    def liquidate(self):
        positions = self.positions.map(int)
        selling = (self.securities["bid"] * ((positions + 1) / 2) * self.quantities).sum()
        buying = (self.securities["ask"] * ((positions - 1) / 2) * self.quantities).sum()
        return selling - buying

    @property
    def gain(self): return max(self.market - self.spent, 0)
    @property
    def loss(self): return max(self.spent - self.market, 0)
    @property
    def profit(self): return self.liquidate - self.spent


class DivestitureCreators(object):
    def __new__(cls, *args, spreads, **kwargs):
        spreads = [spread for spread in spreads if spread != Spread.EMPTY]
        instances = [DivestitureCreator[spread](*args, **kwargs) for spread in spreads]
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



