# -*- coding: utf-8 -*-
"""
Created on Sat May 16 2026
@name:   Option Target Objects
@author: Jack Kirby Cook
@file:   options/targets.py

"""

import math
import numpy as np
import pandas_market_calendars as calenders
from typing import Optional
from dataclasses import dataclass
from types import SimpleNamespace
from abc import ABC, abstractmethod
from functools import cached_property
from datetime import date as Date
from datetime import timedelta as Timedelta

from finance.enumerations import Spread, Instrument, Action
from finance.reporting import Results, Analysis
from options.prospects import Prospect
from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Target", "Calculator", "Costing", "Slippage", "Scenario"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


nyse = calenders.get_calendar("NYSE")


@dataclass(frozen=True, slots=True)
class Slippage: entry: float = 0.25; exit: float = 0.35

@dataclass(frozen=True, slots=True)
class Costing: slippage: Slippage; commissions: float = 0.65 / 100

@dataclass(frozen=True, slots=True)
class Greeks: delta: float; gamma: float; theta: float; vega: float

@dataclass(frozen=True, slots=True)
class Scenario: zscore: float; days: int; vpts: int; prob: Optional[float] = None

@dataclass(frozen=True, slots=True)
class Risk:
    greeks: Greeks; underlying: float; volatility: float

    def __call__(self, scenario):
        shock = self.shock(scenario.zscore, scenario.days)
        delta = self.delta(shock)
        gamma = self.gamma(shock)
        theta = self.theta(scenario.days)
        vega = self.vega(scenario.vpts)
        return delta + gamma + theta + vega

    def shock(self, zscore, days): return zscore * self.underlying * self.volatility * math.sqrt(days / 252)
    def delta(self, shock): return self.greeks.delta * (shock ** 1) / 1
    def gamma(self, shock): return self.greeks.gamma * (shock ** 2) / 2
    def theta(self, days): return self.greeks.theta * (days / 252)
    def vega(self, vpts): return self.greeks.vega * (vpts / 100)


class Target(Prospect, ABC, columns="forecast market zscore bid ask gap tightness moneyness activity delta gamma theta vega"):
    def __init__(self, *args, scenarios, costing, halflife, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(scenarios, list)
        assert all([isinstance(scenario, Scenario) for scenario in scenarios])
        assert isinstance(costing, Costing)
        self.__scenarios = scenarios
        self.__halflife = halflife
        self.__costing = costing

    def __iter__(self):
        for osi, purpose, position, quantity in zip(self.osi, self.purpose, self.positions, self.quantities):
            yield SimpleNamespace(osi=osi, purpose=purpose, position=position, quantity=quantity)

    @property
    def signature(self):
        function = lambda record: (str(record.osi), int(record.purpose.action), int(record.purpose.intent), int(record.position), int(record.quantity))
        return tuple(function(record) for record in self)

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

    @property
    def gap(self): return (self.securities["gap"] * self.quantities).sum()
    @property
    def tightness(self): return self.securities["tightness"].max()
    @property
    def moneyness(self): return self.securities["moneyness"].max()
    @property
    def activity(self): return self.securities["activity"].min()

    @cached_property
    def purpose(self): return [SimpleNamespace(action=action, intent=self.intent) for action in self.actions]
    @cached_property
    def actions(self): return self.positions.apply(lambda position: Action(int(self.intent) * int(position)))

    @cached_property
    def greeks(self): return Greeks(delta=self.delta, gamma=self.gamma, theta=self.theta, vega=self.vega)
    @cached_property
    def risk(self): return Risk(greeks=self.greeks, underlying=self.underlying, volatility=self.volatility)

    @cached_property
    def zspread(self):
        if self.spread is Spread.CALENDAR: zspread = self.zscore / (self.quantities.sum() / 2)
        elif self.spread is Spread.FLY: zspread = self.zscore / (self.quantities.sum() / 2)
        else: raise ValueError(self.spread)
        return abs(zspread) * self.factor

    @cached_property
    def factor(self): return 1 - np.power(2, - self.dte / self.halflife)
    @cached_property
    def dte(self): return len(nyse.valid_days(start_date=Date.today() + Timedelta(days=1), end_date=self.expires.minimum))
    @cached_property
    def loss(self): return max(0, - min([self.risk(scenario) for scenario in self.scenarios]))

    @cached_property
    def cost(self): return float(self.commissions) + float(self.slippage)
    @cached_property
    def price(self): return float(self.market) * int(self.intent)

    @cached_property
    def multiple(self): return self.edge / max(self.cost, 1e-4)
    @cached_property
    def ratio(self): return self.pnl / max(self.loss, 1e-2)

    @cached_property
    def edge(self): return (self.forecast - self.market) * self.factor
    @cached_property
    def pnl(self): return self.edge - self.cost

    @classmethod
    def create(cls, prospect, *args, scenarios, costing, halflife, **kwargs):
        arguments = (prospect.spread, prospect.securities)
        parameters = dict(scenarios=scenarios, costing=costing, halflife=halflife)
        return cls(*arguments, **parameters)

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
    def scenarios(self): return self.__scenarios
    @property
    def halflife(self): return self.__halflife
    @property
    def costing(self): return self.__costing


class Calculator(Analysis.Targets, Results, Logging, ABC):
    def __init_subclass__(cls, /, target, **kwargs):
        super().__init_subclass__()
        cls.__target__ = target

    def __init__(self, *args,  metrics, priority, costing, scenarios, halflife, **kwargs):
        super().__init__(*args, **kwargs)
        self.__scenarios = scenarios
        self.__costing = costing
        self.__halflife = halflife
        self.__priority = priority
        self.__metrics = metrics

    def __call__(self, prospects, **kwargs):
        assert isinstance(prospects, list) and all([isinstance(prospect, Prospect) for prospect in prospects])
        scope = self.scope(prospects, instrument=Instrument.SPREAD)
        parameters = dict(costing=self.costing, scenarios=self.scenarios, halflife=self.halflife)
        targets = [self.target.create(prospect, **parameters) for prospect in prospects]
        if not bool(targets): return targets
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
    def scenarios(self): return self.__scenarios
    @property
    def costing(self): return self.__costing
    @property
    def halflife(self): return self.__halflife
    @property
    def priority(self): return self.__priority
    @property
    def metrics(self): return self.__metrics



