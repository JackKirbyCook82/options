# -*- coding: utf-8 -*-
"""
Created on Tues May 12 2026
@name:   Option Prospect Objects
@author: Jack Kirby Cook

"""

import pandas as pd
from abc import ABC, abstractmethod

from support.finance import Concepts, Alerting
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ProspectCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class Scanner(ABC, metaclass=RegistryMeta):
    def __init__(self, *args, metrics, proximity=1, **kwargs):
        self.__proximity = int(proximity)
        self.__metrics = metrics

    def __call__(self, options, *args, **kwargs):
        for spread in self.generator(options):
            prospects = spread(self.metrics, *args, **kwargs)
            if bool(prospects.empty): continue
            yield prospects

    @abstractmethod
    def selector(self, length): pass
    @abstractmethod
    def generator(self, options): pass

    @property
    def proximity(self): return self.__proximity
    @property
    def metrics(self): return self.__metrics


class FlyScanner(Scanner, register=Concepts.Strategies.Spread.FLY):
    def selector(self, length):
        for width in range(1, self.proximity + 1):
            for left in range(length - 2 * width):
                center = left + width
                right = left + width * 2
                yield [left, center, right]

    def generator(self, options):
        for position in iter(Concepts.Securities.Position):
            hedge = Concepts.Securities.Position(-int(position))
            for option in iter(Concepts.Securities.Option):
                dataframes = options[options["option"].eq(option)]
                for dte, dataframe in dataframes.groupby("dte"):
                    dataframe = dataframe.sort_values("strike")
                    for index in self.selector(len(dataframe)):
                        legs = dataframe.iloc[index]
                        legs["spread"] = Concepts.Strategies.Spread.FLY
                        legs["position"] = [hedge, position, hedge]
                        legs["quantity"] = [1, 2, 1]
                        yield legs


class CalenderScanner(Scanner, Concepts.Strategies.Spread.CALENDER):
    def selector(self, length):
        for width in range(1, self.proximity + 1):
            for near in range(length - width):
                far = near + width
                yield [near, far]

    def generator(self, options):
        for position in iter(Concepts.Securities.Position):
            hedge = Concepts.Securities.Position(-int(position))
            for option in iter(Concepts.Securities.Option):
                dataframes = options[options["option"].eq(option)]
                for strike, dataframe in dataframes.groupby("strike"):
                    dataframe = dataframe.sort_values("dte")
                    for index in self.selector(len(dataframe)):
                        legs = dataframe.iloc[index]
                        legs["spread"] = Concepts.Strategies.Spread.CALENDAR
                        legs["position"] = [hedge, position]
                        legs["quantity"] = [1, 1]
                        yield legs


class ProspectCalculator(Alerting):
    def __init__(self, *args, metrics, proximity=1, **kwargs):
        assert isinstance(proximity, int) and proximity >= 1
        super().__init__(*args, **kwargs)
        metrics = {Concepts.Strategies.Spread[str(key).upper()]: value for key, value in metrics.items()}
        scanners = [Scanner[key](proximity=proximity, metrics=value) for key, value in metrics.items()]
        self.__proximity = int(proximity)
        self.__scanners = scanners

    def __call__(self, options, *args, **kwargs):
        pass

#    def __call__(self, options, *args, **kwargs):
#        assert isinstance(options, pd.DataFrame)
#        prospects = self.scanner(options, *args, **kwargs)
#        prospects = list(prospects)
#        if bool(prospects): prospects = pd.concat(prospects, axis=0)
#        else: prospects = pd.DataFrame(columns=options.columns)
#        prospects = prospects.sort_values(by=["identity"], ascending=True, inplace=False)
#        prospects = prospects.reset_index(drop=True, inplace=False)
#        sizes = dict(previous=len(options), post=len(prospects))
#        self.alert(options, title="Calculated", instrument=Concepts.Securities.Instrument.OPTION, **sizes)
#        return prospects
#
#    def scanner(self, options, *args, **kwargs):
#        for scanner in self.scanners:
#            generator = scanner(options, *args, **kwargs)
#            yield from generator

    @property
    def proximity(self): return self.__proximity
    @property
    def scanners(self): return self.__scanners






