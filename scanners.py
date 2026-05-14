# -*- coding: utf-8 -*-
"""
Created on Tues May 12 2026
@name:   Option Scanner Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from types import SimpleNamespace
from abc import ABC, abstractmethod

from support.finance import Concepts, Alerting
from support.meta import CounterMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["CalenderScanner", "FlyScanner"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class Spread(ABC):
    def __init__(self, strategy, spread):
        self.__strategy = strategy
        self.__spread = spread

    @property
    def gamma(self): return (self.spread["gamma"] * self.position * self.quantity).sum()
    @property
    def theta(self): return (self.spread["theta"] * self.position * self.quantity).sum()
    @property
    def vega(self): return (self.spread["vega"] * self.position * self.quantity).sum()
    @property
    def valuation(self): return (self.spread["value"] * self.position * self.quantity).sum()
    @property
    def market(self): return (self.spread["medium"] * self.position * self.quantity).sum()
    @property
    def efficiency(self): return np.abs(self.zscore) * self.edge / max(self.friction, 1e-12)
    @property
    def gap(self): return (self.spread["gap"] * self.quantity).sum()
    @property
    def edge(self): return self.valuation - self.market
    @property
    def friction(self): return self.gap / self.edge
    @property
    def position(self): return self.spread["position"].map(int)
    @property
    def quantity(self): return self.spread["quantity"]

    @property
    @abstractmethod
    def zscore(self): pass
    @property
    def spread(self): return self.__spread


class Fly(Spread):
    @property
    def zscore(self):
        left, center, right = self.spread["zscore"].to_numpy()
        return center - (left + right) / 2

class Calender(Spread):
    @property
    def zscore(self):
        far, near = self.spread["zscore"].to_numpy()
        return far - near


class Scanner(Alerting, ABC, metaclass=CounterMeta):
    def __init__(self, *args, proximity=1, **kwargs):
        assert isinstance(proximity, int) and proximity >= 1
        super().__init__(*args, **kwargs)
        self.__proximity = int(proximity)

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        previous = len(options.index)
        options = self.scanner(options, *args, **kwargs)
        options = pd.concat(list(options), axis=0)
        post = len(options.index)
        options = options.sort_values(by=["identity"], ascending=[True, True], inplace=False)
        options = options.reset_index(drop=True, inplace=False)
        sizes = dict(previous=previous, post=post)
        self.alert(options, title="Calculated", instrument=Concepts.Securities.Instrument.OPTION, **sizes)
        return options

    def scanner(self, options, *args, **kwargs):
        for strategy, spread in self.generator(options, *args, **kwargs):
            pass

    @abstractmethod
    def generator(self, options, *args, **kwargs): pass
    @abstractmethod
    def selector(self, length): pass

    @property
    def proximity(self): return self.__proximity


class CalenderScanner(Scanner):
    def selector(self, length):
        for width in range(1, self.proximity + 1):
            for near in range(length - width):
                far = near + width
                yield [near, far]

    def generator(self, options, *args, **kwargs):
        for position, hedge in zip(iter(Concepts.Securities.Position), reversed(Concepts.Securities.Position)):
            for option in iter(Concepts.Securities.Option):
                dataframes = options[options["option"].eq(option)]
                for strike, dataframe in dataframes.groupby("strike"):
                    dataframe = dataframe.sort_values("dte")
                    for index in self.selector(len(dataframe)):
                        strategy = SimpleNamespace(position=position, option=option, strike=strike)
                        spread = dataframe.iloc[index]
                        spread["position"] = [hedge, position]
                        spread["quantity"] = [1, 1]
                        yield strategy, spread


class FlyScanner(Scanner):
    def selector(self, length):
        for width in range(1, self.proximity + 1):
            for left in range(length - 2 * width):
                center = left + width
                right = left + width * 2
                yield [left, center, right]

    def generator(self, options, *args, **kwargs):
        for position, hedge in zip(iter(Concepts.Securities.Position), reversed(Concepts.Securities.Position)):
            for option in iter(Concepts.Securities.Option):
                dataframes = options[options["option"].eq(option)]
                for dte, dataframe in dataframes.groupby("dte"):
                    dataframe = dataframe.sort_values("strike")
                    for index in self.selector(len(dataframe)):
                        strategy = SimpleNamespace(position=position, option=option, dte=dte)
                        spread = dataframe.iloc[index]
                        spread["position"] = [hedge, position, hedge]
                        spread["quantity"] = [1, 2, 1]
                        yield strategy, spread



