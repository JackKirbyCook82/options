# -*- coding: utf-8 -*-
"""
Created on Sat May 16 2026
@name:   Option Spread Objects
@author: Jack Kirby Cook

"""

import math
import pandas as pd
from abc import ABC
from dataclasses import dataclass
from types import SimpleNamespace

from finance.osi import OSI
from finance.enumerations import Spread
from support.custom import DateRange, NumRange

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Prospect"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True, slots=True)
class Scenario: sigmas: int; days: int; vols: int

@dataclass(frozen=True, slots=True)
class Greeks: delta: float; gamma: float; theta: float; vega: float

@dataclass(frozen=True, slots=True)
class Risk:
    greeks: Greeks; edge: float; underlying: float; volatility: float

    def __call__(self, days: int = 1, vols: int = 1, sigmas: int = 1):
        delta = self.delta(sigmas, days)
        gamma = self.gamma(sigmas, days)
        theta = self.theta(days)
        vega = self.vega(vols)
        return delta + gamma + theta + vega

    def delta(self, sigmas, days):
        shock = self.underlying * self.volatility * sigmas
        pnl = self.greeks.delta * shock * math.sqrt(days / 252)
        return pnl / max(abs(self.edge), 1e-12)

    def gamma(self, sigmas, days):
        shock = self.underlying * self.volatility * sigmas
        pnl = 0.5 * self.greeks.gamma * shock * math.sqrt(days / 252)
        return pnl / max(abs(self.edge), 1e-12)

    def theta(self, days):
        pnl = self.greeks.theta * (days / 365)
        return pnl / max(abs(self.edge), 1e-12)

    def vega(self, vols):
        pnl = self.greeks.vega * (vols / 100)
        return pnl / max(abs(self.edge), 1e-12)


class Prospect(ABC):
    def __init__(self, spread, securities):
        assert isinstance(securities, pd.DataFrame)
        assert len(securities["ticker"].unique()) == 1
        assert len(securities["underlying"].unique()) == 1
        assert len(securities["volatility"].unique()) == 1
        assert spread in list(Spread)
        self.__ticker = securities["ticker"].unique()[0]
        self.__expires = DateRange.create(securities["expire"].to_list())
        self.__securities = securities
        self.__spread = spread

    def __iter__(self):
        for osi, position, quantity in zip(self.osi, self.position, self.quantity):
            yield SimpleNamespace(osi=osi, position=position, quantity=quantity)

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
    def forcast(self): return (self.securities["forecast"] * self.position.map(int) * self.quantity).sum()
    @property
    def market(self): return (self.securities["median"] * self.position.map(int) * self.quantity).sum()
    @property
    def edge(self): return self.forcast - self.market

    @property
    def gamma(self): return (self.securities["gamma"] * self.position.map(int) * self.quantity).sum()
    @property
    def theta(self): return (self.securities["theta"] * self.position.map(int) * self.quantity).sum()
    @property
    def vega(self): return (self.securities["vega"] * self.position.map(int) * self.quantity).sum()
    @property
    def greeks(self): return dict(gamma=self.gamma, theta=self.theta, vega=self.vega)

    @property
    def gap(self): return (self.securities["gap"] * self.quantity).sum()
    @property
    def tightness(self): return self.securities["tightness"].max()
    @property
    def moneyness(self): return self.securities["moneyness"].max()
    @property
    def activity(self): return self.securities["activity"].min()

    @property
    def position(self): return self.securities["position"]
    @property
    def quantity(self): return self.securities["quantity"]

    @property
    def securities(self): return self.__securities
    @property
    def spread(self): return self.__spread
    @property
    def expires(self): return self.__expires
    @property
    def ticker(self): return self.__ticker



