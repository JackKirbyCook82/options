# -*- coding: utf-8 -*-
"""
Created on Sat May 16 2026
@name:   Option Spread Objects
@author: Jack Kirby Cook

"""

import math
import pandas as pd
from dataclasses import dataclass
from types import SimpleNamespace
from abc import ABC, abstractmethod
from functools import cached_property

from finance.osi import OSI
from finance.logging import Logging
from finance.enumerations import Spread, Instrument, Position, Action
from support.custom import DateRange

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ProspectCalculator", "Prospect", "Costing", "Slippage"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True, slots=True)
class Slippage: entry: float = 0.25; exit: float = 0.35

@dataclass(frozen=True, slots=True)
class Costing: slippage: Slippage; commissions: float = 0.65 / 100

@dataclass(frozen=True, slots=True)
class Greeks: delta: float; gamma: float; theta: float; vega: float

@dataclass(frozen=True, slots=True)
class Scenario: cdays: int; tdays: int; vpts: float; sigmas: float

@dataclass(frozen=True, slots=True)
class Risk:
    greeks: Greeks; scenario: Scenario; underlying: float; volatility: float; edge: float

    @cached_property
    def shock(self): return self.scenario.sigmas * self.underlying * self.volatility * math.sqrt(self.scenario.tdays / 252)
    @cached_property
    def edge(self): return min(abs(self.edge), 1e-12)

    @property
    def delta(self): return self.greeks.delta * self.shock / self.edge
    @property
    def gamma(self): return 0.5 * self.greeks.gamma * (self.shock ** 2) / self.edge
    @property
    def theta(self): return self.greeks.theta * (self.scenario.cdays / 365) / self.edge
    @property
    def vega(self): return self.greeks.vega * (self.scenario.vpts / 100) / self.edge


class Prospect(ABC):
    def __init__(self, spread, securities, /, scenario, costing):
        assert isinstance(securities, pd.DataFrame)
        assert isinstance(scenario, Scenario)
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
        self.__scenario = scenario
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
    def premium(self):
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
    def position(self): return Position((self.edge > 0) - (self.edge < 0))
    @cached_property
    def cost(self): return self.commissions + self.slippage
    @cached_property
    def edge(self): return self.forecast - self.market

    @property
    def risk(self): return Risk(greeks=self.greeks, scenario=self.scenario, underlying=self.underlying, volatility=self.volatility, edge=self.edge)
    @property
    def greeks(self): return Greeks(delta=self.delta, gamma=self.gamma, theta=self.theta, vega=self.vega)

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
    def scenario(self): return self.__scenario
    @property
    def costing(self): return self.__costing
    @property
    def spread(self): return self.__spread
    @property
    def expires(self): return self.__expires
    @property
    def ticker(self): return self.__ticker

    @property
    @abstractmethod
    def commissions(self): pass
    @property
    @abstractmethod
    def slippage(self): pass
    @property
    @abstractmethod
    def intent(self): pass


class ProspectCalculator(Logging):
    def __init__(self, *args, creators, metrics, priority, **kwargs):
        super().__init__(*args, **kwargs)
        self.__creators = creators
        self.__priority = priority
        self.__metrics = metrics

    def __call__(self, holdings, /, **kwargs):
        assert isinstance(holdings, pd.DataFrame)
        prospects = self.calculate(holdings, **kwargs)
        self.results(prospects, title="Calculator", instrument=Instrument.SPREAD)
        return prospects

    def calculate(self, holdings, /, **kwargs):
        assert isinstance(holdings, pd.DataFrame)
        prospects = self.calculator(holdings, **kwargs)
        prospects = list(prospects)
        priorities = [self.priority(prospect) for prospect in prospects].__getitem__
        prospects = (prospects[index] for index in sorted(range(len(prospects)), key=priorities, reverse=True))
        return prospects

    def calculator(self, holdings, /, **kwargs):
        assert isinstance(holdings, pd.DataFrame)
        for spread, creator in self.creators.items():
            for prospect in creator(holdings, **kwargs):
                if not self.metrics(prospect): continue
                yield prospect

    @property
    def creators(self): return self.__creators
    @property
    def priority(self): return self.__priority
    @property
    def metrics(self): return self.__metrics



