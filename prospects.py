# -*- coding: utf-8 -*-
"""
Created on Sat May 16 2026
@name:   Option Spread Objects
@author: Jack Kirby Cook

"""

import pandas as pd
from typing import Optional
from dataclasses import dataclass
from types import SimpleNamespace
from abc import ABC, abstractmethod
from functools import cached_property

from finance.osi import OSI
from finance.logging import Logging
from finance.enumerations import Spread, Instrument, Position
from support.custom import DateRange

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ProspectCalculator", "Prospect", "Costing", "Slippage", "Scenario"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True, slots=True)
class Slippage: entry: float = 0.25; exit: float = 0.35

@dataclass(frozen=True, slots=True)
class Costing: slippage: Slippage | float; commissions: float = 0.65

@dataclass(frozen=True, slots=True)
class Greeks: delta: float; gamma: float; theta: float; vega: float

@dataclass(frozen=True, slots=True)
class Risk: greeks: Greeks; underlying: float; volatility: float

@dataclass(frozen=True, slots=True)
class Scenario:
    cdays: int; tdays: int; vpts: float; sigmas: float
    odds: Optional[float] = None
    name: Optional[str] = None


class Prospect(ABC):
    def __init__(self, spread, securities, /, costing, scenarios):
        assert isinstance(scenarios, list) and all([isinstance(scenario, Scenario) for scenario in scenarios])
        assert isinstance(costing, Costing)
        assert isinstance(securities, pd.DataFrame)
        assert len(securities["ticker"].unique()) == 1
        assert len(securities["underlying"].unique()) == 1
        assert len(securities["volatility"].unique()) == 1
        assert spread in list(Spread)
        self.__ticker = securities["ticker"].unique()[0]
        self.__expires = DateRange(securities["expire"].to_list())
        self.__underlying = securities["underlying"].unique()[0]
        self.__volatility = securities["volatility"].unique()[0]
        self.__securities = securities
        self.__scenarios = scenarios
        self.__costing = costing
        self.__spread = spread

    def __iter__(self):
        for osi, position, quantity in zip(self.osi, self.positions, self.quantities):
            yield SimpleNamespace(osi=osi, position=position, quantity=quantity)

    @cached_property
    def zspread(self):
        if self.spread is Spread.FLY:
            left, center, right = self.securities["zscore"].to_numpy()
            return center - (left + right) / 2
        elif self.spread is Spread.CALENDAR:
            near, far = self.securities["zscore"].to_numpy()
            return far - near
        else: raise ValueError(self.spread)

    @property
    def greeks(self): return Greeks(delta=self.delta, gamma=self.gamma, theta=self.theta, vega=self.vega)
    @property
    def risk(self): return Risk(greeks=self.greeks, underlying=self.underlying, volatility=self.volatility)

    @cached_property
    def forecast(self): return (self.securities["forecast"] * self.positions.map(int) * self.quantities).sum()
    @cached_property
    def market(self): return (self.securities["median"] * self.positions.map(int) * self.quantities).sum()
    @cached_property
    def position(self): return Position((self.edge > 0) - (self.edge < 0))

    @cached_property
    def edge(self): return (self.forecast - self.market) * int(self.position)
    @cached_property
    def cost(self): return self.commissions + self.slippage

    @property
    def signature(self): return tuple((str(record.osi), int(record.position), int(record.quantity)) for record in self)
    @property
    def osi(self): return self.securities[["ticker", "expire", "option", "strike"]].apply(OSI, axis=1)

    @property
    def delta(self): return (self.securities["delta"] * self.positions.map(int) * self.quantities).sum()
    @property
    def gamma(self): return (self.securities["gamma"] * self.positions.map(int) * self.quantities).sum()
    @property
    def theta(self): return (self.securities["theta"] * self.positions.map(int) * self.quantities).sum()
    @property
    def vega(self): return (self.securities["vega"] * self.positions.map(int) * self.quantities).sum()

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
    @abstractmethod
    def commissions(self): pass
    @property
    @abstractmethod
    def slippage(self): pass
    @property
    @abstractmethod
    def intent(self): pass

    @property
    def securities(self): return self.__securities
    @property
    def underlying(self): return self.__underlying
    @property
    def volatility(self): return self.__volatility
    @property
    def scenarios(self): return self.__scenarios
    @property
    def costing(self): return self.__costing
    @property
    def spread(self): return self.__spread
    @property
    def expires(self): return self.__expires
    @property
    def ticker(self): return self.__ticker


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



