# -*- coding: utf-8 -*-
"""
Created on Mon Jul 6 2026
@name:   Option Acquisition Objects
@author: Jack Kirby Cook

"""

import math
import pandas as pd
from itertools import product
from abc import ABC, abstractmethod
from functools import cached_property
from dataclasses import dataclass, field, astuple

from options.prospects import Prospect, Scenario
from finance.enumerations import Spread, Instrument, Option, Position, Intent
from finance.specifications import Securities
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["AcquisitionCreators", "Metrics", "Weights", "Targets", "Priority"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True, slots=True)
class Measures: zspread: float; multiple: float; ratio: float
class Weights(Measures): pass
class Targets(Measures): pass
class Metrics(Measures):
    def __post_init__(self):
        assert self.zspread > 0
        assert self.multiple > 0
        assert self.ratio > 0

    def __call__(self, acquisition):
        assert isinstance(acquisition, Acquisition)
        if acquisition.zspread <= self.zspread: return False
        if acquisition.multiple <= self.multiple: return False
        if acquisition.ratio <= self.ratio: return False
        return True


@dataclass(frozen=True, slots=True)
class Priority:
    targets: Measures = field(default_factory=lambda: Measures(zspread=3.0, multiple=4.0, ratio=20.0))
    weights: Measures = field(default_factory=lambda: Measures(zspread=0.20, multiple=0.30, ratio=0.35))

    def __call__(self, acquisition):
        assert isinstance(acquisition, Acquisition)
        values = Measures(zspread=acquisition.zspread, multiple=acquisition.multiple, ratio=acquisition.ratio)
        generator = zip(astuple(values), astuple(self.targets), astuple(self.weights))
        function = lambda value, target, weight: weight * math.log(max(value / (value + target), 1e-12))
        return math.exp(sum([function(*arguments) for arguments in generator]))


class Acquisition(Prospect):
    @property
    def slippage(self): return (self.costing.slippage.entry + self.costing.slippage.exit) * self.gap
    @property
    def commissions(self): return self.costing.commissions * self.quantities.sum() * 2
    @property
    def intent(self): return Intent.OPEN

    @cached_property
    def multiple(self): return self.edge / self.cost
    @cached_property
    def ratio(self): return self.gain / self.loss

    @cached_property
    def edge(self): return self.forecast - self.market
    @cached_property
    def gain(self): return self.edge - self.cost

    @cached_property
    def loss(self):
        generator = product(range(-1, 2), range(-1, 2))
        scenarios = (Scenario(zscore=zscore, vpts=vpts, tdays=1, cdays=1) for zscore, vpts in generator)
        worse = min([self.risk(scenario) for scenario in scenarios]) - self.cost
        return max(self.cost, - worse)


class AcquisitionCreators(object):
    def __new__(cls, *args, spreads, **kwargs):
        spreads = [spread for spread in spreads if spread != Spread.EMPTY]
        instances = {spread: AcquisitionCreator[spread](*args, **kwargs) for spread in spreads}
        return instances


class AcquisitionCreator(ABC, metaclass=RegistryMeta):
    def __init__(self, *args, costing, limit=1, **kwargs):
        assert isinstance(limit, int) and limit > 0
        self.__costing = costing
        self.__limit = limit

    def __call__(self, options, /, **kwargs):
        assert isinstance(options, pd.DataFrame)
        securities = self.securities(options)
        organized = self.organize(securities)
        for security, dataframe in organized:
            locators = self.locators(dataframe)
            for locator in locators:
                located = dataframe.iloc[locator].copy()
                prospect = self.creator(security, located)
                yield prospect

    @staticmethod
    def securities(options):
        for position in iter(Position):
            for option in iter(Option):
                if option is Option.EMPTY: continue
                if position is Position.EMPTY: continue
                security = [Instrument.OPTION, option, position]
                security = Securities(tuple(security))
                dataframe = options[options["option"].eq(option)]
                yield security, dataframe

    @staticmethod
    @abstractmethod
    def organize(securities): pass
    @abstractmethod
    def locators(self, securities): pass
    @abstractmethod
    def creator(self, security, securities): pass

    @property
    def costing(self): return self.__costing
    @property
    def limit(self): return self.__limit


class FlyAcquisitionCreator(AcquisitionCreator, register=Spread.FLY):
    @staticmethod
    def organize(securities):
        for security, dataframes in securities:
            for dte, dataframe in dataframes.groupby("dte"):
                dataframe = dataframe.sort_values("strike")
                yield security, dataframe

    def locators(self, securities):
        for section in range(1, self.limit + 1):
            for index in range(len(securities) - 2 * section):
                yield [index, index + section, index + section * 2]

    def creator(self, security, securities):
        body, wing = security.position, Position(-int(security.position))
        securities["spread"] = Spread.FLY
        securities["position"] = [wing, body, wing]
        securities["quantity"] = [1, 2, 1]
        parameters = dict(costing=self.costing)
        prospect = Acquisition(Spread.FLY, securities, **parameters)
        return prospect


class CalendarAcquisitionCreator(AcquisitionCreator, register=Spread.CALENDAR):
    @staticmethod
    def organize(securities):
        for security, dataframes in securities:
            for strike, dataframe in dataframes.groupby("strike"):
                dataframe = dataframe.sort_values("dte")
                yield security, dataframe

    def locators(self, securities):
        for section in range(1, self.limit + 1):
            for index in range(len(securities) - section):
                yield [index, index + section]

    def creator(self, security, securities):
        far, near = security.position, Position(-int(security.position))
        securities["spread"] = Spread.CALENDAR
        securities["position"] = [near, far]
        securities["quantity"] = [1, 1]
        parameters = dict(costing=self.costing)
        prospect = Acquisition(Spread.CALENDAR, securities, **parameters)
        return prospect




