# -*- coding: utf-8 -*-
"""
Created on Sat May 16 2026
@name:   Option Spread Objects
@author: Jack Kirby Cook

"""

import math
import pandas as pd
from abc import ABC
from typing import Optional
from itertools import product
from dataclasses import dataclass
from types import SimpleNamespace

from finance.osi import OSI
from finance.enumerations import Spread, Position
from support.custom import DateRange, NumRange

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Prospect"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True, slots=True)
class Scenario: days: int; vols: int; sigma: int; probability: Optional[float] = None

@dataclass(frozen=True, slots=True)
class Scenarios:
    days: NumRange; vols: NumRange; sigma: NumRange

    def __iter__(self):
        generator = product(self.days, self.vols, self.sigma)
        for days, vols, sigma in generator:
            yield Scenario(days=days, vols=vols, sigma=sigma)


@dataclass(frozen=True, slots=True)
class Greeks: delta: float; gamma: float; theta: float; vega: float

@dataclass(frozen=True, slots=True)
class Risk:
    greeks: Greeks; edge: float; underlying: float; volatility: float

    def __call__(self, scenario):
        delta = self.delta(scenario.days, scenario.sigmas)
        gamma = self.gamma(scenario.days, scenario.sigmas)
        theta = self.theta(scenario.days)
        vega = self.vega(scenario.vols)
        return delta + gamma + theta + vega

    def shock(self, days, sigma):
        movement = self.underlying * self.volatility * sigma
        return movement * math.sqrt(days / 365)

    def delta(self, days, sigma):
        pnl = self.greeks.delta * self.shock(days, sigma)
        return pnl / max(abs(self.edge), 1e-12)

    def gamma(self, days, sigma):
        pnl = 0.5 * self.greeks.gamma * self.shock(days, sigma) ** 2
        return pnl / max(abs(self.edge), 1e-12)

    def theta(self, days):
        pnl = self.greeks.theta * (days / 365)
        return pnl / max(abs(self.edge), 1e-12)

    def vega(self, vols):
        pnl = self.greeks.vega * (vols / 100)
        return pnl / max(abs(self.edge), 1e-12)


class Prospect(ABC):
    def __init__(self, spread, securities, scenarios):
        assert isinstance(securities, pd.DataFrame)
        assert len(securities["ticker"].unique()) == 1
        assert len(securities["underlying"].unique()) == 1
        assert len(securities["volatility"].unique()) == 1
        assert spread in list(Spread)
        self.__ticker = securities["ticker"].unique()[0]
        self.__expires = DateRange.create(securities["expire"].to_list())
        self.__securities = securities
        self.__scenarios = scenarios
        self.__spread = spread

    def __iter__(self):
        for osi, position, quantity in zip(self.osi, self.positions, self.quantities):
            yield SimpleNamespace(osi=osi, position=position, quantity=quantity)

    @property
    def var(self):
        position = lambda drift: Position((drift > 0) - (drift < 0))
        drifts = [self.risk(scenario) * self.edge for scenario in self.scenarios]
        unfavorable = [abs(drift) for drift in drifts if position(drift) != self.position]
        var = max(unfavorable)
        return var

    @property
    def risk(self):
        assert len(self.securities["underlying"].unique()) == 1
        underlying = self.securities["underlying"].values[0]
        volatility = self.securities["implied"].mean()
        greeks = Greeks(**self.greeks)
        return Risk(greeks, self.edge, underlying, volatility)

    @property
    def zscore(self):
        if self.spread is Spread.FLY:
            left, center, right = self.securities["zscore"].to_numpy()
            return center - (left + right) / 2
        elif self.spread is Spread.CALENDAR:
            near, far = self.securities["zscore"].to_numpy()
            return far - near
        else: raise ValueError(self.spread)

    @property
    def signature(self): return tuple((str(record.osi), int(record.position), int(record.quantity)) for record in self)
    @property
    def osi(self): return self.securities[["ticker", "expire", "option", "strike"]].apply(OSI, axis=1)

    @property
    def forcast(self): return (self.securities["forecast"] * self.positions.map(int) * self.quantities).sum()
    @property
    def market(self): return (self.securities["median"] * self.positions.map(int) * self.quantities).sum()
    @property
    def position(self): return Position((self.edge > 0) - (self.edge < 0))
    @property
    def edge(self): return self.forcast - self.market

    @property
    def gamma(self): return (self.securities["gamma"] * self.positions.map(int) * self.quantities).sum()
    @property
    def theta(self): return (self.securities["theta"] * self.positions.map(int) * self.quantities).sum()
    @property
    def vega(self): return (self.securities["vega"] * self.positions.map(int) * self.quantities).sum()
    @property
    def greeks(self): return dict(gamma=self.gamma, theta=self.theta, vega=self.vega)

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
    def scenarios(self): return self.__scenarios
    @property
    def spread(self): return self.__spread
    @property
    def expires(self): return self.__expires
    @property
    def ticker(self): return self.__ticker



