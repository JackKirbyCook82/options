# -*- coding: utf-8 -*-
"""
Created on Sat May 16 2026
@name:   Option Prospect Objects
@author: Jack Kirby Cook
@file:   options/prospects.py

"""

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from types import SimpleNamespace
from abc import ABC, abstractmethod
from functools import cached_property
from datetime import date as Date

from finance.osi import OSI
from finance.reporting import Results
from finance.enumerations import Spread, Instrument, Position, Option
from finance.specifications import Securities
from support.meta import RegistryMeta
from support.custom import DateRange
from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ProspectMarketCalculator", "ProspectPortfolioCalculator", "Prospect"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True, slots=True)
class Greeks: delta: float; gamma: float; theta: float; vega: float

@dataclass(frozen=True, slots=True)
class Risk:
    greeks: Greeks; underlying: float; volatility: float

    def __call__(self, scenario):
        shock = self.shock(scenario.zscore, scenario.tdays)
        delta = self.delta(shock)
        gamma = self.gamma(shock)
        theta = self.theta(scenario.cdays)
        vega = self.vega(scenario.vpts)
        return delta + gamma + theta + vega

    def shock(self, zscore, tdays): return zscore * self.underlying * self.volatility * math.sqrt(tdays / 252)
    def delta(self, shock): return self.greeks.delta * (shock ** 1) / 1
    def gamma(self, shock): return self.greeks.gamma * (shock ** 2) / 2
    def theta(self, cdays): return self.greeks.theta * (cdays / 365)
    def vega(self, vpts): return self.greeks.vega * (vpts / 100)


class ProspectError(Exception): pass
class ProspectTickerError(ProspectError): pass
class ProspectUnderlyingError(ProspectError): pass
class ProspectVolatilityError(ProspectError): pass
class ProspectColumnError(ProspectError): pass


class ProspectMeta(type):
    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        existing = getattr(cls, "columns", [])
        updated = kwargs.get("columns", [])
        if isinstance(updated, str): updated = str(updated).split(" ")
        elif isinstance(updated, list): updated = list(updated)
        else: raise TypeError(type(updated))
        cls.columns = existing + updated

    def __call__(cls, spread, securities):
        assert spread in list(Spread)
        assert isinstance(securities, pd.DataFrame)
        for column in cls.columns:
            if securities[column].isna().any(): raise ProspectColumnError()
        instance = super().__call__(spread, securities)
        return instance


class Prospect(object, metaclass=ProspectMeta, columns=["ticker expire underlying volatility forecast market zscore delta gamma theta vega gap tightness moneyness activity quantity position"]):
    def __new__(cls, spread, securities, *args, **kwargs):
        assert spread in list(Spread)
        assert isinstance(securities, pd.DataFrame)
        if not len(securities["ticker"].unique()) == 1: raise ProspectTickerError()
        if not len(securities["underlying"].unique()) == 1: raise ProspectUnderlyingError()
        if not len(securities["volatility"].unique()) == 1: raise ProspectVolatilityError()
        instance = super().__new__(cls)
        return instance

    def __init__(self, spread, securities, *args, **kwargs):
        self.__ticker = securities["ticker"].unique()[0]
        self.__expires = DateRange(securities["expire"].to_list())
        self.__underlying = securities["underlying"].unique()[0]
        self.__volatility = securities["volatility"].unique()[0]
        self.__securities = securities
        self.__spread = spread
        super().__init__()

    def __iter__(self):
        for osi, position, quantity in zip(self.osi, self.positions, self.quantities):
            yield SimpleNamespace(osi=osi, position=position, quantity=quantity)

    @property
    def signature(self): return tuple((str(record.osi), int(record.position), int(record.quantity)) for record in self)
    @cached_property
    def dte(self): return (self.expires.minimum - Date.today()).days
    @property
    def osi(self):
        try: return self.securities["osi"]
        except KeyError: return self.securities[["ticker", "expire", "option", "strike"]].apply(OSI, axis=1)

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
    def greeks(self): return Greeks(delta=self.delta, gamma=self.gamma, theta=self.theta, vega=self.vega)
    @cached_property
    def risk(self): return Risk(greeks=self.greeks, underlying=self.underlying, volatility=self.volatility)

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
    def expires(self): return self.__expires
    @property
    def ticker(self): return self.__ticker
    @property
    def spread(self): return self.__spread


class ProspectCreator(ABC, metaclass=RegistryMeta):
    def __call__(self, options, /, limit=1, **kwargs):
        assert isinstance(options, pd.DataFrame)
        securities = self.securities(options)
        organized = self.organize(securities)
        for security, dataframe in organized:
            locators = self.locators(dataframe, limit)
            for locator in locators:
                located = dataframe.iloc[locator].copy()
                prospect = self.create(security, located)
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
    @staticmethod
    @abstractmethod
    def locators(securities, limit): pass
    @staticmethod
    @abstractmethod
    def create(security, securities): pass


class FlyProspectCreator(ProspectCreator, register=Spread.FLY):
    @staticmethod
    def organize(securities):
        for security, dataframes in securities:
            for dte, dataframe in dataframes.groupby("dte"):
                dataframe = dataframe.sort_values("strike")
                yield security, dataframe

    @staticmethod
    def locators(securities, limit):
        for section in range(1, limit + 1):
            for index in range(len(securities) - 2 * section):
                yield [index, index + section, index + section * 2]

    @staticmethod
    def create(security, securities):
        body, wing = security.position, Position(-int(security.position))
        securities["spread"] = Spread.FLY
        securities["position"] = [wing, body, wing]
        securities["quantity"] = [1, 2, 1]
        prospect = Prospect(Spread.FLY, securities)
        return prospect


class CalendarProspectCreator(ProspectCreator, register=Spread.CALENDAR):
    @staticmethod
    def organize(securities):
        for security, dataframes in securities:
            for strike, dataframe in dataframes.groupby("strike"):
                dataframe = dataframe.sort_values("dte")
                yield security, dataframe

    @staticmethod
    def locators(securities, limit):
        for section in range(1, limit + 1):
            for index in range(len(securities) - section):
                yield [index, index + section]

    @staticmethod
    def create(security, securities):
        far, near = security.position, Position(-int(security.position))
        securities["spread"] = Spread.CALENDAR
        securities["position"] = [near, far]
        securities["quantity"] = [1, 1]
        prospect = Prospect(Spread.CALENDAR, securities)
        return prospect


class ProspectPortfolioCalculator(Results, Logging):
    def __call__(self, holdings, /, **kwargs):
        assert isinstance(holdings, pd.DataFrame)
        scope = self.scope(holdings, instrument=Instrument.OPTION)
        generator = self.calculator(holdings, **kwargs)
        prospects = list(generator)
        results = self.results(scope=scope, size=len(prospects))
        self.console("Calculated", results)
        return prospects

    @staticmethod
    def calculator(holdings, /, **kwargs):
        for (order, spread), securities in holdings.groupby(["order", "spread"]):
            yield Prospect(spread, securities)


class ProspectMarketCalculator(Results, Logging):
    def __init__(self, *args, spreads, limit=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.__creators = [ProspectCreator[spread](*args, **kwargs) for spread in spreads]
        self.__limit = int(limit)

    def __call__(self, options, /, **kwargs):
        assert isinstance(options, pd.DataFrame)
        scope = self.scope(options, instrument=Instrument.OPTION)
        generator = self.calculator(options, **kwargs)
        prospects = list(generator)
        results = self.results(scope=scope, size=len(prospects))
        self.console("Calculated", results)
        return prospects

    def calculator(self, options, /, **kwargs):
        parameters = dict(limit=self.limit)
        for creator in self.creators:
            for prospect in creator(options, **parameters, **kwargs):
                yield prospect

    @property
    def creators(self): return self.__creators
    @property
    def limit(self): return self.__limit



