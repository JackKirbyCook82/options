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

from finance.enumerations import Instrument, Action
from finance.reporting import Results, Analysis
from options.prospects import Prospect
from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Target", "Calculator", "Costing", "Slippage", "Measure", "Metrics", "Priority"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True, slots=True)
class Measure: zspread: float; multiple: float; ratio: float

@dataclass(frozen=True, slots=True)
class Priority: targets: Measure; weights: Measure

@dataclass(frozen=True, slots=True)
class Metrics(Measure):
    def __post_init__(self):
        assert self.zspread > 0
        assert self.multiple > 0
        assert self.ratio > 0


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

    @cached_property
    def multiple(self): return self.edge / self.cost
    @cached_property
    def ratio(self): return self.pnl / self.var

    @cached_property
    def edge(self): return self.forecast - self.market
    @cached_property
    def pnl(self): return self.edge - self.cost

    @classmethod
    def create(cls, prospect, costing):
        return cls(prospect.spread, prospect.securities, costing=costing)

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


class Calculator(Analysis.Targets, Results, Logging, ABC):
    def __init_subclass__(cls, /, target, **kwargs):
        super().__init_subclass__()
        cls.__target__ = target

    def __init__(self, *args,  metrics, priority, costing, **kwargs):
        super().__init__(*args, **kwargs)
        self.__priority = priority
        self.__metrics = metrics
        self.__costing = costing

    def __call__(self, prospects, **kwargs):
        assert isinstance(prospects, list) and all([isinstance(prospect, Prospect) for prospect in prospects])
        scope = self.scope(prospects, instrument=Instrument.SPREAD)
        targets = [self.target.create(prospect, costing=self.costing) for prospect in prospects]
        divestitures = [target for target in targets if self.metrics(target)]
        divestitures.sort(key=self.priority, reverse=True)
        size = (len(targets), len(divestitures))
        results = self.results(scope, size)
        analysis = self.analysis(targets) if bool(targets) else []
        self.console("Calculated", results, *analysis)
        return divestitures

    @property
    def target(self): return type(self).__target__
    @property
    def priority(self): return self.__priority
    @property
    def metrics(self): return self.__metrics
    @property
    def costing(self): return self.__costing



