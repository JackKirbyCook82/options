# -*- coding: utf-8 -*-
"""
Created on Sat May 16 2026
@name:   Option Target Objects
@author: Jack Kirby Cook
@file:   options/targets.py

"""

import math
from itertools import product
from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import cached_property
from types import SimpleNamespace

from finance.enumerations import Action
from options.prospects import Prospect

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Target", "Costing", "Slippage"]
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


class Target(Prospect, ABC, columns=["bid", "ask"]):
    def __init__(self, *args, costing, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(costing, Costing)
        self.__costing = costing

    @cached_property
    def liquidate(self):
        positions = self.positions.map(int).astype(int)
        quantities = self.quantities.astype(float)
        actions = positions * int(self.intent)
        mask = actions.eq(int(Action.BUY))
        prices = self.securities["ask"].where(mask, self.securities["bid"])
        return abs((prices * positions * quantities).sum() - self.market)

    @cached_property
    def var(self):
        generator = product(range(-1, 2), range(-1, 2))
        scenarios = (Scenario(zscore=zscore, vpts=vpts, tdays=1, cdays=1) for zscore, vpts in generator)
        worse = min([self.risk(scenario) for scenario in scenarios]) - self.cost
        return max(self.cost, - worse)

    @cached_property
    def purpose(self): return [SimpleNamespace(action=action, intent=self.intent) for action in self.actions]
    @cached_property
    def actions(self): return self.positions.apply(lambda position: Action(int(self.intent) * int(position)))

    @cached_property
    def risk(self): return Risk(greeks=self.greeks, underlying=self.underlying, volatility=self.volatility)
    @cached_property
    def greeks(self): return Greeks(delta=self.delta, gamma=self.gamma, theta=self.theta, vega=self.vega)

    @cached_property
    def cost(self): return float(self.commissions) + float(self.slippage)
    @cached_property
    def price(self): return float(self.market) * int(self.intent)

    @classmethod
    def create(cls, prospect, costing): return cls(prospect.spread, prospect.securities, costing=costing)

    @property
    @abstractmethod
    def commissions(self): pass
    @property
    @abstractmethod
    def slippage(self): pass
    @property
    @abstractmethod
    def intent(self): pass

    @property
    def costing(self): return self.__costing




