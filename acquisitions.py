# -*- coding: utf-8 -*-
"""
Created on Mon Jul 6 2026
@name:   Option Acquisition Objects
@author: Jack Kirby Cook
@file:   options/acquisitions.py

"""

import math
import pandas as pd
from abc import ABC, abstractmethod
from functools import cached_property
from dataclasses import dataclass, astuple

from options.prospects import Prospect
from finance.enumerations import Spread, Instrument, Option, Position, Intent
from finance.specifications import Securities
from finance.logging import Logging
from support.custom import NumberRange
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["AcquisitionCalculator", "AcquisitionMetric"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True, slots=True)
class Measure:
    zspread: float | NumberRange
    multiple: float | NumberRange
    ratio: float | NumberRange

class Metric(Measure):
    def __post_init__(self):
        assert self.zspread > 0
        assert self.multiple > 0
        assert self.ratio > 0

    def __call__(self, measure):
        assert isinstance(measure, Measure)
        if measure.zspread <= self.zspread: return False
        if measure.multiple <= self.multiple: return False
        if measure.ratio <= self.ratio: return False
        return True


@dataclass(frozen=True, slots=True)
class Priority:
    targets: Measure; weights: Measure

    def __call__(self, acquisition):
        assert isinstance(acquisition, Acquisition)
        values = Measure(zspread=acquisition.zspread, multiple=acquisition.multiple, ratio=acquisition.ratio)
        weights, total = astuple(self.weights), sum(astuple(self.weights))
        weights = (weight / total for weight in weights)
        generator = zip(astuple(values), astuple(self.targets), weights)
        function = lambda value, target, weight: weight * math.log(max(value / (value + target), 1e-12))
        return math.exp(sum([function(*arguments) for arguments in generator]))


class Acquisition(Prospect):
    @property
    def slippage(self): return (self.costing.slippage.entry + self.costing.slippage.exit) * self.gap
    @property
    def commissions(self): return self.costing.commissions * self.quantities.sum() * 2
    @property
    def intent(self): return Intent.OPEN

    @property
    def measure(self): return Measure(self.zspread, self.multiple, self.ratio)
    @property
    def priority(self):
        targets = Measure(zspread=3.0, multiple=5.0, ratio=20.0)
        weights = Measure(zspread=0.30, multiple=0.30, ratio=0.40)
        priority = Priority(targets=targets, weights=weights)
        return priority(self)

    @cached_property
    def multiple(self): return self.edge / self.cost
    @cached_property
    def ratio(self): return self.pnl / self.var

    @cached_property
    def edge(self): return self.forecast - self.market
    @cached_property
    def pnl(self): return self.edge - self.cost


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


class AcquisitionMetric(Metric): pass
class AcquisitionCalculator(Logging):
    def __init__(self, *args, spreads, metric, **kwargs):
        super().__init__(*args, **kwargs)
        self.__creators = {spread: AcquisitionCreator[spread](*args, **kwargs) for spread in spreads}
        self.__metric = metric

    def __call__(self, options, /, **kwargs):
        assert isinstance(options, pd.DataFrame)
        scope = self.scope(options, instrument=Instrument.OPTION)
        prospects = [prospect for spread, creator in self.creators.items() for prospect in creator(options, **kwargs)]
        acquisitions = [prospect for prospect in prospects if self.metrics(prospect.measure)]
        acquisitions.sort(key=lambda prospect: prospect.priority, reverse=True)
        size = (len(prospects), len(acquisitions))
        breakdown = self.breakdown(prospects)
        self.results(scope=scope, size=size, pre=breakdown, title="Calculated")
        return prospects

    def breakdown(self, prospects):
        boundary = self.boundary(prospects)
        survival = self.survival(prospects)
        zspread = f"ZSpread>={self.metric.zspread:.1f}[{survival.zspread:.0f}%, f{boundary.zspread.minimum:.1f}->f{boundary.zspread.maximum:.1f}]"
        multiple = f"Multiple>={self.metric.multiple:.1f}[{survival.multiple:.0f}%, f{boundary.multiple.minimum:.1f}->f{boundary.multiple.maximum:.1f}]"
        ratio = f"Ratio>={self.metric.ratio:.1f}[{survival.ratio:.0f}%, f{boundary.ratio.minimum:.1f}->f{boundary.ratio.maximum:.1f}]"
        return [zspread, multiple, ratio]

    def survival(self, prospects):
        zspreads = [prospect.measure.zspread >= self.metrics.zspread for prospect in prospects]
        multiples = [prospect.measure.multiple >= self.metrics.multiple for prospect in prospects]
        ratios = [prospect.measure.ratio >= self.metrics.ratio for prospect in prospects]
        zspreads = sum(zspreads) / len(zspreads) * 100
        multiples = sum(multiples) / len(multiples) * 100
        ratios = sum(ratios) / len(ratios) * 100
        return Measure(zspreads, multiples, ratios)

    @staticmethod
    def boundary(prospects):
        zspreads = [prospect.measure.zspread for prospect in prospects]
        multiples = [prospect.measure.multiple for prospect in prospects]
        ratios = [prospect.measure.ratio for prospect in prospects]
        zspreads = NumberRange([min(zspreads), max(zspreads)])
        multiples = NumberRange([min(multiples), max(multiples)])
        ratios = NumberRange([min(ratios), max(ratios)])
        return Measure(zspreads, multiples, ratios)

    @property
    def creators(self): return self.__creators
    @property
    def metric(self): return self.__metric



