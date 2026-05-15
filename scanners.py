# -*- coding: utf-8 -*-
"""
Created on Tues May 12 2026
@name:   Option Scanner Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass
from types import SimpleNamespace
from abc import ABC, abstractmethod

from support.finance import Concepts, Alerting
from support.meta import CounterMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["CalenderScanner", "FlyScanner", "Metrics", "Ratios"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True)
class Ratios: gap: float; theta: Optional[float] = None

@dataclass(frozen=True)
class Metrics:
    gamma: Optional[float]; theta: Optional[float]; vega: Optional[float]
    ratios: Ratios; zscore: float; edge: float


class Spread(ABC, metaclass=CounterMeta):
    def __init__(self, strategy, legs, *args, **kwargs):
        assert isinstance(strategy, SimpleNamespace) and isinstance(legs, pd.DataFrame)
        legs = self.create(legs, *args, **kwargs)
        self.__strategy = strategy
        self.__legs = legs

    def __call__(self, metrics, *args, **kwargs):
        if not self.qualify(metrics): return None
        strategy = dict(vars(self.legs))
        identity = type(self).counter
        parameters = strategy | dict(identity=identity)
        prospects = self.legs.update(parameters)
        return prospects

    def qualify(self, metrics):
        assert isinstance(metrics, Metrics)
        if self.edge < metrics.edge: return False
        ratios = all([self.ratios.gap <= metrics.ratios.gap, self.ratios.theta >= metrics.ratios.theta])
        zscore = abs(self.zscore) >= abs(metrics.zscore)
        gamma = True if metrics.gamma is None else abs(self.gamma) <= abs(metrics.gamma)
        theta = True if metrics.theta is None else self.theta >= metrics.theta
        vega = True if metrics.vega is None else self.vega > metrics.vega
        return all([ratios, zscore, gamma, theta, vega])

    @property
    def gamma(self): return (self.legs["gamma"] * self.position * self.quantity).sum()
    @property
    def theta(self): return (self.legs["theta"] * self.position * self.quantity).sum()
    @property
    def vega(self): return (self.legs["vega"] * self.position * self.quantity).sum()
    @property
    def valuation(self): return (self.legs["value"] * self.position * self.quantity).sum()
    @property
    def market(self): return (self.legs["medium"] * self.position * self.quantity).sum()
    @property
    def efficiency(self): return np.abs(self.zscore) * self.edge / max(self.gap, 1e-12)
    @property
    def gap(self): return (self.legs["gap"] * self.quantity).sum()
    @property
    def edge(self): return self.valuation - self.market
    @property
    def position(self): return self.legs["position"].map(int)
    @property
    def quantity(self): return self.legs["quantity"]

    @property
    def ratios(self):
        gap = self.gap / max(self.edge, 1e-12)
        theta = self.theta / max(self.edge, 1e-12)
        return Ratios(gap=gap, theta=theta)

    @property
    @abstractmethod
    def zscore(self): pass
    @abstractmethod
    def create(self, dataframe, *args, position, quantity=1, **kwargs): pass

    @property
    def strategy(self): return self.__strategy
    @property
    def legs(self): return self.__legs


class Fly(Spread):
    def create(self, legs, *args, position, quantity=1, **kwargs):
        legs = legs.sort_values("strike")
        hedge = Concepts.Securities.Position[-int(position)]
        legs["quantity"] = list(map(lambda x: x * quantity, [1, 2, 1]))
        legs["position"] = [hedge, position, hedge]
        legs["spread"] = Concepts.Strategies.Spread.FLY
        return legs

    @property
    def zscore(self):
        left, center, right = self.legs["zscore"].to_numpy()
        return center - (left + right) / 2


class Calender(Spread):
    def create(self, legs, *args, position, quantity=1, **kwargs):
        legs = legs.sort_values("dte")
        hedge = Concepts.Securities.Position[-int(position)]
        legs["quantity"] = list(map(lambda x: x * quantity, [1, 1]))
        legs["position"] = [hedge, position]
        legs["spread"] = Concepts.Strategies.Spread.CALENDAR
        return legs

    @property
    def zscore(self):
        near, far = self.legs["zscore"].to_numpy()
        return far - near


class Scanner(Alerting, ABC):
    def __init__(self, *args, metrics, proximity=1, **kwargs):
        assert isinstance(proximity, int) and proximity >= 1
        assert isinstance(metrics, Metrics)
        super().__init__(*args, **kwargs)
        self.__proximity = int(proximity)
        self.__metrics = metrics

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        prospects = self.scanner(options, *args, **kwargs)
        prospects = list(prospects)
        if bool(options): prospects = pd.concat(prospects, axis=0)
        else: prospects = pd.DataFrame(columns=options.columns)
        options = options.sort_values(by=["identity"], ascending=[True, True], inplace=False)
        options = options.reset_index(drop=True, inplace=False)
        sizes = dict(previous=len(options), post=len(prospects))
        self.alert(prospects, title="Calculated", instrument=Concepts.Securities.Instrument.OPTION, **sizes)
        return options

    def scanner(self, options, *args, **kwargs):
        for spread in self.generator(options, *args, **kwargs):
            yield spread(self.metrics, *args, **kwargs)

    @abstractmethod
    def generator(self, options, *args, **kwargs): pass
    @abstractmethod
    def selector(self, length): pass

    @property
    def proximity(self): return self.__proximity
    @property
    def metrics(self): return self.__metrics


class FlyScanner(Scanner):
    def selector(self, length):
        for width in range(1, self.proximity + 1):
            for left in range(length - 2 * width):
                center = left + width
                right = left + width * 2
                yield [left, center, right]

    def generator(self, options, *args, **kwargs):
        for position in iter(Concepts.Securities.Position):
            for option in iter(Concepts.Securities.Option):
                dataframes = options[options["option"].eq(option)]
                for dte, dataframe in dataframes.groupby("dte"):
                    dataframe = dataframe.sort_values("strike")
                    for index in self.selector(len(dataframe)):
                        strategy = SimpleNamespace(position=position, option=option, dte=dte)
                        spread = dataframe.iloc[index]
                        spread = Fly(strategy, spread, position=position, quantity=1)
                        yield spread


class CalenderScanner(Scanner):
    def selector(self, length):
        for width in range(1, self.proximity + 1):
            for near in range(length - width):
                far = near + width
                yield [near, far]

    def generator(self, options, *args, **kwargs):
        for position in iter(Concepts.Securities.Position):
            for option in iter(Concepts.Securities.Option):
                dataframes = options[options["option"].eq(option)]
                for strike, dataframe in dataframes.groupby("strike"):
                    dataframe = dataframe.sort_values("dte")
                    for index in self.selector(len(dataframe)):
                        strategy = SimpleNamespace(position=position, option=option, strike=strike)
                        spread = dataframe.iloc[index]
                        spread = Calender(strategy, spread, position=position, quantity=1)
                        yield spread



