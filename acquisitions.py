# -*- coding: utf-8 -*-
"""
Created on Mon Jul 6 2026
@name:   Option Acquisition Objects
@author: Jack Kirby Cook

"""

import pandas as pd
from abc import ABC, abstractmethod

from options.prospects import Prospect
from finance.enumerations import Spread, Instrument, Option, Position
from finance.specifications import Securities
from finance.logging import Logging
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["AcquisitionCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class Acquisition(Prospect):
    pass


class AcquisitionCreator(ABC, metaclass=RegistryMeta):
    def __init__(self, *args, limit=1, **kwargs):
        assert isinstance(limit, int) and limit > 0
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
        position = security.position
        hedge = Position(-int(position))
        securities["spread"] = Spread.FLY
        securities["position"] = [hedge, position, hedge]
        securities["quantity"] = [1, 2, 1]
        prospect = Acquisition(Spread.FLY, securities)
        yield prospect


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
        position = security.position
        hedge = Position(-int(position))
        securities["spread"] = Spread.CALENDAR
        securities["position"] = [hedge, position]
        securities["quantity"] = [1, 1]
        prospect = Acquisition(Spread.CALENDAR, securities)
        yield prospect


class AcquisitionCalculator(Logging):
    def __init__(self, *args, spreads, metrics, **kwargs):
        super().__init__(*args, **kwargs)
        creators = {spread: AcquisitionCreator[spread](*args, **kwargs) for spread in spreads}
        self.__creators = creators
        self.__metrics = metrics

    def __call__(self, options, /, **kwargs):
        assert isinstance(options, pd.DataFrame)
        prospects = self.calculator(options, **kwargs)
        prospects = list(prospects)
        self.results(prospects, title="Calculator", instrument=Instrument.SPREAD)
        return prospects

    def calculator(self, options, **kwargs):
        assert isinstance(options, pd.DataFrame)
        for spread, creator in self.creators.items():
            for prospect in creator(options, **kwargs):
                if not self.metrics(prospect): continue
                yield prospect

    @property
    def creators(self): return self.__creators
    @property
    def metrics(self): return self.__metrics




