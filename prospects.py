# -*- coding: utf-8 -*-
"""
Created on Sat May 16 2026
@name:   Option Spread Objects
@author: Jack Kirby Cook

"""

import math
import pandas as pd
from typing import Optional
from dataclasses import dataclass
from types import SimpleNamespace
from functools import cached_property

from finance.osi import OSI
from finance.logging import Logging
from finance.enumerations import Spread, Instrument, Position
from support.custom import DateRange

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ProspectCalculator", "Prospect"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True, slots=True)
class Greeks: delta: float; gamma: float; theta: float; vega: float

@dataclass(frozen=True, slots=True)
class Scenario:
    cdays: int; tdays: int; vpts: float; sigmas: float
    probability: Optional[float] = None
    name: Optional[str] = None

@dataclass(frozen=True, slots=True)
class Risk:
    greeks: Greeks; underlying: float; volatility: float

    def __call__(self, scenario):
        shock = self.underlying * self.volatility * scenario.sigmas * math.sqrt(scenario.tdays / 252)
        delta = self.greeks.delta * shock
        gamma = self.greeks.gamma * (shock ** 2) / 2
        theta = self.greeks.theta * (scenario.cdays / 365)
        vega = self.greeks.vega * (scenario.vpts / 100)
        return delta + gamma + theta + vega


class Prospect(object):
    def __init__(self, spread, securities):
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
        self.__spread = spread

    def __str__(self):
        securities = [f"{str(record.osi)}={int(record.position) * int(record.quantity):.0f}" for record in self]
        valuation = f"{str(self.spread).title()} @ ${self.edge}"
        return "\n".join([valuation] + securities)

    def __iter__(self):
        for osi, position, quantity in zip(self.osi, self.positions, self.quantities):
            yield SimpleNamespace(osi=osi, position=position, quantity=quantity)

#    def pnl(self, scenario): return ((self.edge > 0) - (self.edge < 0)) * self.risk(scenario)
#    def var(self, scenarios): return max((max(0.0, -self.pnl(scenario)) for scenario in scenarios), default=0.0)
#    def ear(self, scenarios): return self.var(scenarios) / max(abs(self.edge), 1e-12)

#    @cached_property
#    def risk(self):
#        assert len(self.securities["underlying"].unique()) == 1
#        underlying = self.securities["underlying"].values[0]
#        volatility = self.securities["implied"].mean()
#        greeks = Greeks(**self.greeks)
#        return Risk(greeks, underlying, volatility)

    @cached_property
    def zspread(self):
        if self.spread is Spread.FLY:
            left, center, right = self.securities["zscore"].to_numpy()
            return center - (left + right) / 2
        elif self.spread is Spread.CALENDAR:
            near, far = self.securities["zscore"].to_numpy()
            return far - near
        else: raise ValueError(self.spread)

    @cached_property
    def forcast(self): return (self.securities["forecast"] * self.positions.map(int) * self.quantities).sum()
    @cached_property
    def market(self): return (self.securities["median"] * self.positions.map(int) * self.quantities).sum()
    @cached_property
    def position(self): return Position((self.edge > 0) - (self.edge < 0))
    @cached_property
    def edge(self): return self.forcast - self.market

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
    def greeks(self): return dict(delta=self.delta, gamma=self.gamma, theta=self.theta, vega=self.vega)

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



