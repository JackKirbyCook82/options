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

from finance.enumerations import Strategy
from finance.osi import OSI
from support.custom import DateRange

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Spread"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True)
class Greeks: delta: float; gamma: float; theta: float; vega: float; theta: float

@dataclass(frozen=True)
class Risk:
    greeks: Greeks; edge: float; underlying: float; volatility: float

    def __call__(self, movement, /, days: int = 1, vols: int = 1):
        delta = self.delta(movement)
        gamma = self.gamma()
        theta = self.theta(days)
        vega = self.vega(vols)
        return delta + gamma + theta + vega

    def delta(self, movement):
        underlying = self.underlying * self.volatility / math.sqrt(252)
        delta = self.greeks.delta * underlying
        return int(movement) * delta / max(abs(self.edge), 1e-12)

    def gamma(self):
        underlying = self.underlying * self.volatility / math.sqrt(252)
        gamma = 0.5 * self.greeks.gamma * underlying ** 2
        return gamma / max(abs(self.edge), 1e-12)

    def theta(self, days):
        theta = self.greeks.theta * (days / 252)
        return theta / max(abs(self.edge), 1e-12)

    def vega(self, vols):
        vega = self.greeks.vega * (vols / 100)
        return vega / max(abs(self.edge), 1e-12)


class Spread(ABC):
    def __init__(self, strategy, securities):
        assert isinstance(securities, pd.DataFrame)
        assert len(securities["ticker"].unique()) == 1
        assert len(securities["underlying"].unique()) == 1
        assert len(securities["volatility"].unique()) == 1
        assert strategy in list(Strategy)
        self.__ticker = securities["ticker"].unique()[0]
        self.__expires = DateRange.create(securities["expire"].to_list())
        self.__securities = securities
        self.__strategy = strategy

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
        if self.strategy is Strategy.FLY:
            left, center, right = self.securities["zscore"].to_numpy()
            return center - (left + right) / 2
        elif self.strategy is Strategy.CALENDAR:
            near, far = self.securities["zscore"].to_numpy()
            return far - near
        else: raise ValueError(self.strategy)

    @property
    def signature(self): return tuple((str(record.osi), int(record.position), int(record.quantity)) for record in self)
    @property
    def osi(self): return self.securities[["ticker", "expire", "option", "strike"]].apply(OSI, axis=1)

#    WRONG VALUE
#    @property
#    def value(self): return (self.securities["value"] * self.position.map(int) * self.quantity).sum()

    @property
    def market(self): return (self.securities["median"] * self.position.map(int) * self.quantity).sum()

#    WRONG VALUE
#    @property
#    def edge(self): return self.value - self.market

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
    def strategy(self): return self.__strategy
    @property
    def expires(self): return self.__expires
    @property
    def ticker(self): return self.__ticker



