# -*- coding: utf-8 -*-
"""
Created on Sat May 16 2026
@name:   Option Spread Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from typing import Optional
from abc import ABC, abstractmethod
from functools import total_ordering
from dataclasses import dataclass, fields

from finance.variables import Enumerations, Specifications
from finance.logging import Logging
from finance.osi import OSI
from support.custom import DateRange
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SpreadCalculator", "SpreadMetrics"]
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
        theta = self.risk.theta / edge
        gamma = abs(self.risk.gamma) / edge
        vega = abs(self.risk.vega) / edge
        return zscore - 1.5 * gap + 0.5 * theta - 0.5 * gamma - 0.25 * vega


@dataclass(frozen=True)
class Ratios:
    gamma: Optional[float] = None; theta: Optional[float] = None; vega: Optional[float] = None
    gap: Optional[float] = None

@dataclass(frozen=True)
class SpreadMetrics:
    ratios: Ratios; zscore: float; profit: float

    @classmethod
    def create(cls, /, ratios, zscore, profit):
        assert isinstance(ratios, dict) and isinstance(zscore, float) and isinstance(profit, float)
        ratios = {field.name: ratios.get(field.name, None) for field in fields(Ratios)}
        ratios = Ratios(**ratios)
        return cls(ratios, zscore, profit)


class SpreadMeta(RegistryMeta): pass
class Spread(ABC, metaclass=SpreadMeta):
    def __init__(self, legs):
        assert isinstance(legs, pd.DataFrame)
        tickers = list(legs["ticker"].unique())
        types = list(legs["spread"].unique())
        assert len(tickers) == 1 and len(types) == 1
        self.__ticker = str(tickers[0])
        self.__type = types[0]
        self.__legs = legs

    def __str__(self):
        pass

    @property
    def osi(self): return self.legs[["ticker", "expire", "option", "strike"]].apply(OSI, axis=1)
    @property
    def cost(self): return (self.legs["median"] * self.position.map(int) * self.quantity).sum()

    @property
    def score(self): return Score(self.profit, self.quality, self.risk)
    @property
    def profit(self): return Profit(self.valuation, self.market)
    @property
    def quality(self): return Quality(self.zscore, self.gap, self.profit)
    @property
    def risk(self): return Risk(self.gamma, self.theta, self.vega)

    @property
    def gamma(self): return (self.legs["gamma"] * self.position.map(int) * self.quantity).sum()
    @property
    def theta(self): return (self.legs["theta"] * self.position.map(int) * self.quantity).sum()
    @property
    def vega(self): return (self.legs["vega"] * self.position.map(int) * self.quantity).sum()

    @property
    def valuation(self): return (self.legs["value"] * self.position.map(int) * self.quantity).sum()
    @property
    def market(self): return (self.legs["median"] * self.position.map(int) * self.quantity).sum()
    @property
    def gap(self): return (self.legs["gap"] * self.quantity).sum()

    @property
    def ratios(self):
        gamma = self.gamma / max(float(self.profit), 1e-12)
        theta = self.theta / max(float(self.profit), 1e-12)
        vega = self.vega / max(float(self.profit), 1e-12)
        gap = self.gap / max(float(self.profit), 1e-12)
        return Ratios(gamma=gamma, theta=theta, vega=vega, gap=gap)

    @property
    def tightness(self): return self.legs["tightness"].max()
    @property
    def moneyness(self): return self.legs["moneyness"].max()
    @property
    def activity(self): return self.legs["activity"].min()

    @property
    def expires(self):
        expires = self.legs["expire"].to_list()
        return DateRange.create(expires)

    @property
    def position(self): return self.legs["position"]
    @property
    def quantity(self): return self.legs["quantity"]

    @property
    @abstractmethod
    def zscore(self): pass

    @property
    def ticker(self): return self.__ticker
    @property
    def legs(self): return self.__legs
    @property
    def type(self): return self.__type


class FlySpread(Spread, register=Enumerations.Spread.FLY):
    @property
    def zscore(self):
        left, center, right = self.legs["zscore"].to_numpy()
        return center - (left + right) / 2


class CalenderSpread(Spread, register=Enumerations.Spread.CALENDAR):
    @property
    def zscore(self):
        near, far = self.legs["zscore"].to_numpy()
        return far - near


class SpreadCreator(ABC, metaclass=RegistryMeta):
    def __init__(self, limit=1): self.limit = limit
    def __call__(self, options):
        assert isinstance(options, pd.DataFrame)
        securities = self.securities(options)
        organized = self.organizer(securities)
        for security, dataframe in organized:
            limit, length = int(self.limit), len(dataframe.index)
            locators = self.locators(limit, length)
            for locator in locators:
                located = dataframe.iloc[locator].copy()
                selector = self.selector(security, located)
                for selected in selector:
                    yield self.create(selected)

    @staticmethod
    def securities(options):
        for position in iter(Enumerations.Position):
            for option in iter(Enumerations.Option):
                if option is Enumerations.Option.EMPTY: continue
                if position is Enumerations.Position.EMPTY: continue
                security = [Enumerations.Instrument.OPTION, option, position]
                security = Specifications.Securities(tuple(security))
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
    @staticmethod
    @abstractmethod
    def create(selected): pass


class FlyCreator(SpreadCreator, register=Enumerations.Spread.FLY):
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
        hedge = Enumerations.Position(-int(position))
        located["spread"] = Enumerations.Spread.FLY
        located["position"] = [hedge, position, hedge]
        located["quantity"] = [1, 2, 1]
        yield located

    @staticmethod
    def create(located): return Spread[Enumerations.Spread.FLY](located)


class CalendarCreator(SpreadCreator, register=Enumerations.Spread.CALENDAR):
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
        hedge = Enumerations.Position(-int(position))
        located["spread"] = Enumerations.Spread.CALENDAR
        located["position"] = [hedge, position]
        located["quantity"] = [1, 1]
        yield located

    @staticmethod
    def create(located): return Spread[Enumerations.Spread.CALENDAR](located)


class SpreadCalculator(Logging):
    def __init__(self, *args, spreads, limit=1, **kwargs):
        assert isinstance(limit, int) and limit > 0
        super().__init__(*args, **kwargs)
        creators = {Enumerations.Spread(spread): SpreadCreator[spread](limit=limit) for spread in spreads}
        self.__creators = creators

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        generator = self.calculator(options, *args, **kwargs)
        spreads = list(generator)
        self.results(spreads, title="Calculator", instrument=Enumerations.Instrument.SPREAD)
        return spreads

    def calculator(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        for creator in self.creators.values():
            for spread in creator(options):
                yield spread

    @property
    def creators(self): return self.__creators



