# -*- coding: utf-8 -*-
"""
Created on Sat May 16 2026
@name:   Option Target Objects
@author: Jack Kirby Cook
@file:   options/targets.py

"""

from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import cached_property
from types import SimpleNamespace

from finance.enumerations import Spread, Instrument, Action
from finance.reporting import Results, Analysis
from options.prospects import Prospect
from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Target", "Calculator", "Costing", "Slippage", "Measure", "Metrics", "Priority", "Scenario"]
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
class Scenario: zscore: float; cdays: int; tdays: int; vpts: int; prob: float


class Target(Prospect, ABC, columns=["bid", "ask"]):
    def __init__(self, *args, scenarios, costing, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(scenarios, list)
        assert all([isinstance(scenario, Scenario) for scenario in scenarios])
        assert isinstance(costing, Costing)
        self.__scenarios = scenarios
        self.__costing = costing

    @cached_property
    def purpose(self): return [SimpleNamespace(action=action, intent=self.intent) for action in self.actions]
    @cached_property
    def actions(self): return self.positions.apply(lambda position: Action(int(self.intent) * int(position)))

    @cached_property
    def cost(self): return float(self.commissions) + float(self.slippage)
    @cached_property
    def price(self): return float(self.market) * int(self.intent)

    @cached_property
    def zspread(self):
        if self.spread is Spread.CALENDAR: return self.zscore / (self.quantities.sum() / 2)
        elif self.spread is Spread.FLY: return self.zscore / (self.quantities.sum() / 2)
        else: raise ValueError(self.spread)

    @cached_property
    def multiple(self): return self.edge / self.cost
    @cached_property
    def ratio(self): return self.pnl / self.var

    @cached_property
    def edge(self): return self.forecast - self.market
    @cached_property
    def pnl(self): return self.edge - self.cost

    @cached_property
    def expected(self): return
    @cached_property
    def var(self): return

#    @cached_property
#    def var(self): return max(0, - min([self.risk(scenario) for scenario in self.scenarios]))

    @classmethod
    def create(cls, prospect, scenarios, costing):
        arguments = (prospect.spread, prospect.securities)
        parameters = dict(scenarios=scenarios, costing=costing)
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
    def costing(self): return self.__costing


class Calculator(Analysis.Targets, Results, Logging, ABC):
    def __init_subclass__(cls, /, target, **kwargs):
        super().__init_subclass__()
        cls.__target__ = target

    def __init__(self, *args,  metrics, priority, costing, scenarios, **kwargs):
        super().__init__(*args, **kwargs)
        self.__scenarios = scenarios
        self.__costing = costing
        self.__priority = priority
        self.__metrics = metrics

    def __call__(self, prospects, **kwargs):
        assert isinstance(prospects, list) and all([isinstance(prospect, Prospect) for prospect in prospects])
        scope = self.scope(prospects, instrument=Instrument.SPREAD)
        parameters = dict(costing=self.costing, scenarios=self.scenarios)
        targets = [self.target.create(prospect, **parameters) for prospect in prospects]
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
    def priority(self): return self.__priority
    @property
    def metrics(self): return self.__metrics



