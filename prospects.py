# -*- coding: utf-8 -*-
"""
Created on Sat May 16 2026
@name:   Option Spread Objects
@author: Jack Kirby Cook
@file:   options/prospects.py

"""

import math
import pandas as pd
from itertools import product
from dataclasses import dataclass
from types import SimpleNamespace
from abc import ABC, abstractmethod
from functools import cached_property

from finance.osi import OSI
from finance.enumerations import Spread, Action
from support.custom import DateRange

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Prospect", "ProspectCosting", "ProspectSlippage"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True, slots=True)
class Slippage: entry: float = 0.25; exit: float = 0.35

@dataclass(frozen=True, slots=True)
class Costing: slippage: Slippage; commissions: float = 0.65 / 100

@dataclass(frozen=True, slots=True)
class Scenario: zscore: float; cdays: int; tdays: int; vpts: int

@dataclass(frozen=True, slots=True)
class Greeks: delta: float; gamma: float; theta: float; vega: float

@dataclass(frozen=True, slots=True)
class Risk:
    greeks: Greeks; underlying: float; volatility: float

    def __call__(self, scenario):
        shock = self.shock(scenario.zscore, scenario.tdays)
        delta = self.delta(shock)
        gamma = self.gamma(shock)
        theta = self.theta(scenario.cdays)
        vega = self.vega(scenario.vpts)
        return delta + gamma + theta + vega

    def shock(self, zscore, tdays): return zscore * self.underlying * self.volatility * math.sqrt(tdays / 252)
    def delta(self, shock): return self.greeks.delta * (shock ** 1) / 1
    def gamma(self, shock): return self.greeks.gamma * (shock ** 2) / 2
    def theta(self, cdays): return self.greeks.theta * (cdays / 365)
    def vega(self, vpts): return self.greeks.vega * (vpts / 100)


class ProspectSlippage(Slippage): pass
class ProspectCosting(Costing): pass
class Prospect(ABC):
    def __init__(self, spread, securities, /, costing):
        assert isinstance(securities, pd.DataFrame)
        assert isinstance(costing, Costing)
        assert len(securities["ticker"].unique()) == 1
        assert len(securities["underlying"].unique()) == 1
        assert len(securities["volatility"].unique()) == 1
        assert spread in list(Spread)
        self.__ticker = securities["ticker"].unique()[0]
        self.__expires = DateRange(securities["expire"].to_list())
        self.__underlying = securities["underlying"].unique()[0]
        self.__volatility = securities["volatility"].unique()[0]
        self.__securities = securities
        self.__costing = costing
        self.__spread = spread

    def __iter__(self):
        for osi, position, quantity in zip(self.osi, self.positions, self.quantities):
            yield SimpleNamespace(osi=osi, position=position, quantity=quantity)

    @cached_property
    def forecast(self): return (self.securities["forecast"] * self.positions.map(int) * self.quantities).sum()
    @cached_property
    def market(self): return (self.securities["market"] * self.positions.map(int) * self.quantities).sum()
    @cached_property
    def zscore(self): return (self.securities["zscore"] * self.positions.map(int) * self.quantities).sum()
    @cached_property
    def delta(self): return (self.securities["delta"] * self.positions.map(int) * self.quantities).sum()
    @cached_property
    def gamma(self): return (self.securities["gamma"] * self.positions.map(int) * self.quantities).sum()
    @cached_property
    def theta(self): return (self.securities["theta"] * self.positions.map(int) * self.quantities).sum()
    @cached_property
    def vega(self): return (self.securities["vega"] * self.positions.map(int) * self.quantities).sum()

    @cached_property
    def liquidate(self):
        positions = self.positions.map(int).astype(int)
        quantities = self.quantities.astype(float)
        actions = positions * int(self.intent)
        mask = actions.eq(int(Action.BUY))
        prices = self.securities["ask"].where(mask, self.securities["bid"])
        return abs((prices * positions * quantities).sum() - self.market)

    @cached_property
    def zspread(self):
        if self.spread is Spread.CALENDAR: return self.zscore / (self.quantities.sum() / 2)
        elif self.spread is Spread.FLY: return self.zscore / (self.quantities.sum() / 2)
        else: raise ValueError(self.spread)

    @cached_property
    def var(self):
        generator = product(range(-1, 2), range(-1, 2))
        scenarios = (Scenario(zscore=zscore, vpts=vpts, tdays=1, cdays=1) for zscore, vpts in generator)
        worse = min([self.risk(scenario) for scenario in scenarios]) - self.cost
        return max(self.cost, - worse)

    @property
    def risk(self): return Risk(greeks=self.greeks, underlying=self.underlying, volatility=self.volatility)
    @property
    def greeks(self): return Greeks(delta=self.delta, gamma=self.gamma, theta=self.theta, vega=self.vega)
    @property
    def cost(self): return self.commissions + self.slippage
    @property
    def price(self): return self.market

    @property
    def signature(self): return tuple((str(record.osi), int(record.position), int(record.quantity)) for record in self)
    @property
    def osi(self): return self.securities[["ticker", "expire", "option", "strike"]].apply(OSI, axis=1)

    @property
    def gap(self): return (self.securities["gap"] * self.quantities).sum()
    @property
    def tightness(self): return self.securities["tightness"].max()
    @property
    def moneyness(self): return self.securities["moneyness"].max()
    @property
    def activity(self): return self.securities["activity"].min()

    @property
    def positions(self): return self.securities["position"]
    @property
    def quantities(self): return self.securities["quantity"]

    @property
    def securities(self): return self.__securities
    @property
    def underlying(self): return self.__underlying
    @property
    def volatility(self): return self.__volatility
    @property
    def costing(self): return self.__costing
    @property
    def expires(self): return self.__expires
    @property
    def ticker(self): return self.__ticker
    @property
    def spread(self): return self.__spread

    @property
    @abstractmethod
    def measure(self): pass
    @property
    @abstractmethod
    def priority(self): pass

    @property
    @abstractmethod
    def commissions(self): pass
    @property
    @abstractmethod
    def slippage(self): pass
    @property
    @abstractmethod
    def intent(self): pass




