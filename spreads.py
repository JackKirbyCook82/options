# -*- coding: utf-8 -*-
"""
Created on Sat May 16 2026
@name:   Option Spread Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import total_ordering

from support.finance import Concepts, Alerting
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SpreadCalculator", "Metrics", "Ratios"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@total_ordering
@dataclass(frozen=True)
class Profit:
    valuation: float; market: float

    def __lt__(self, other): return float(self) < float(other)
    def __float__(self): return self.valuation - self.market

@total_ordering
@dataclass(frozen=True)
class Quality:
    zscore: float; gap: float; profit: Profit

    def __lt__(self, other): return float(self) < float(other)
    def __float__(self): return np.abs(self.zscore) * float(self.profit) / max(self.gap, 1e-12)

@total_ordering
@dataclass(frozen=True)
class Risk:
    gamma: float; theta: float; vega: float

    def __lt__(self, other): return float(self) < float(other)
    def __float__(self): return abs(self.gamma) + abs(self.vega) + max(0.0, -self.theta)

@total_ordering
@dataclass(frozen=True)
class Score:
    profit: Profit; quality: Quality; risk: Risk

    def __lt__(self, other): return float(self) < float(other)
    def __float__(self):
        zscore = abs(self.quality.zscore)
        edge = max(float(self.profit), 1e-12)
        gap = float(self.quality.gap) / edge
        theta = abs(self.risk.theta) / edge
        gamma = abs(self.risk.gamma) / edge
        vega = abs(self.risk.vega) / edge
        return zscore - 1.5 * gap + 0.5 * theta - 0.5 * gamma - 0.25 * vega


@dataclass(frozen=True)
class Ratios:
    gamma: Optional[float] = None; theta: Optional[float] = None; vega: Optional[float] = None
    gap: Optional[float] = None

@dataclass(frozen=True)
class Metrics: ratios: Ratios; zscore: float; edge: float


class SpreadMeta(RegistryMeta): pass
class Spread(ABC, metaclass=SpreadMeta):
    def __init__(self, legs):
        assert isinstance(legs, pd.DataFrame)
        tickers = "|".join(list(legs["ticker"].unique()))
        assert len(tickers) == 1
        self.__ticker = str(tickers[0])
        self.__legs = legs

    @property
    def score(self): return Score(self.profit, self.quality, self.risk)
    @property
    def profit(self): return Profit(self.valuation, self.market)
    @property
    def quality(self): return Quality(self.zscore, self.gap, self.profit)
    @property
    def risk(self): return Risk(self.gamma, self.theta, self.vega)

    @property
    def gamma(self): return (self.legs["gamma"] * self.position * self.quantity).sum()
    @property
    def theta(self): return (self.legs["theta"] * self.position * self.quantity).sum()
    @property
    def vega(self): return (self.legs["vega"] * self.position * self.quantity).sum()

    @property
    def valuation(self): return (self.legs["value"] * self.position * self.quantity).sum()
    @property
    def market(self): return (self.legs["median"] * self.position * self.quantity).sum()
    @property
    def gap(self): return (self.legs["gap"] * self.quantity).sum()

    @property
    def position(self): return self.legs["position"].map(int)
    @property
    def quantity(self): return self.legs["quantity"]

    @property
    @abstractmethod
    def zscore(self): pass

    @property
    def ticker(self): return self.__ticker
    @property
    def legs(self): return self.__legs


class FlySpread(Spread, register=Concepts.Strategies.Spread.FLY):
    @property
    def zscore(self):
        left, center, right = self.legs["zscore"].to_numpy()
        return center - (left + right) / 2


class CalenderSpread(Spread, register=Concepts.Strategies.Spread.CALENDAR):
    @property
    def zscore(self):
        near, far = self.legs["zscore"].to_numpy()
        return far - near


class SpreadGenerator(ABC, metaclass=RegistryMeta):
    def __init__(self, limit=1): self.limit = limit
    def __call__(self, options):
        assert isinstance(options, pd.DataFrame)
        securities = self.securities(options)
        organized = self.organizer(securities)
        for security, dataframe in organized:
            limit, length = int(self.limit), len(dataframe.index)
            locators = self.locators(limit, length)
            for locator in locators:
                located = dataframe.iloc[locator]
                selected = self.selector(security, located)
                yield selected

    @staticmethod
    def securities(options):
        for position in iter(Concepts.Securities.Position):
            for option in iter(Concepts.Securities.Option):
                security = [Concepts.Securities.Instrument.OPTION, option, position]
                security = Concepts.Securities.Security(security)
                dataframe = options[options["option"].eq(option)]
                yield security, dataframe

    @staticmethod
    @abstractmethod
    def organizer(securities): pass
    @staticmethod
    @abstractmethod
    def locators(limit, length): pass
    @staticmethod
    @abstractmethod
    def selector(security, located): pass


class FlyGenerator(SpreadGenerator, register=Concepts.Strategies.Spread.FLY):
    @staticmethod
    def organizer(securities):
        for security, dataframes in securities:
            for dte, dataframe in dataframes.groupby("dte"):
                dataframe = dataframe.sort_values("strike")
                yield security, dataframe

    @staticmethod
    def locators(limit, length):
        for section in range(1, limit + 1):
            for left in range(length - 2 * section):
                center = left + section
                right = left + section * 2
                yield [left, center, right]

    @staticmethod
    def selector(security, located):
        position = security.position
        hedge = Concepts.Securities.Position(-int(position))
        located["spread"] = Concepts.Strategies.Spread.FLY
        located["position"] = [hedge, position, hedge]
        located["quantity"] = [1, 2, 1]
        spread = Spread[Concepts.Strategies.Spread.FLY](located)
        yield spread


class CalendarGenerator(SpreadGenerator, register=Concepts.Strategies.Spread.CALENDAR):
    @staticmethod
    def organizer(securities):
        for security, dataframes in securities:
            for strike, dataframe in dataframes.groupby("strike"):
                dataframe = dataframe.sort_values("dte")
                yield security, dataframe

    @staticmethod
    def locators(limit, length):
        for section in range(1, limit + 1):
            for near in range(length - section):
                far = near + section
                yield [near, far]

    @staticmethod
    def selector(security, located):
        position = security.position
        hedge = Concepts.Securities.Position(-int(position))
        located["spread"] = Concepts.Strategies.Spread.CALENDAR
        located["position"] = [hedge, position]
        located["quantity"] = [1, 1]
        spread = Spread[Concepts.Strategies.Spread.CALENDAR](located)
        yield spread


class SpreadCalculator(Alerting):
    def __init__(self, *args, spreads, limit=1, **kwargs):
        assert isinstance(limit, int) and limit > 0
        super().__init__(*args, **kwargs)
        spreads = [Concepts.Strategies.Spread[str(spread).upper()] for spread in spreads]
        spreads = {spread: SpreadGenerator[spread](limit=limit) for spread in spreads}
        self.__spreads = spreads

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        generator = self.generator(options, *args, **kwargs)
        spreads = list(generator)
        sizes = dict(previous=len(options), post=len(spreads))
        self.alert(spreads, title="Calculator", instrument=Concepts.Securities.Instrument.OPTION, **sizes)
        return spreads

    def generator(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        for spreads in self.spreads.values():
            for spread in spreads(options):
                yield spread

    @property
    def spreads(self): return self.__spreads



