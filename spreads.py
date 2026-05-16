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

from support.meta import CounterMeta, RegistryMeta
from support.finance import Concepts

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Spread", "Metrics", "Ratios"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True)
class Ratios: gap: float; theta: Optional[float] = None

@dataclass(frozen=True)
class Metrics:
    gamma: Optional[float]; theta: Optional[float]; vega: Optional[float]
    ratios: Ratios; zscore: float; profit: float; quality: float


class SpreadMeta(CounterMeta, RegistryMeta): pass
class Spread(ABC, metaclass=SpreadMeta):
    def __init__(self, legs, *args, **kwargs):
        assert isinstance(legs, pd.DataFrame)
        self.__legs = self.create(legs, *args, **kwargs)
        self.__identity = type(self).counter

#    def __call__(self, metrics, *args, **kwargs):
#        columns = ["identity"] + list(self.legs.columns)
#        if not self.qualify(metrics): return pd.DataFrame(columns=columns)
#        identity = type(self).counter
#        prospects = self.legs.assign(identity=identity)
#        return prospects

    def qualify(self, metrics):
        assert isinstance(metrics, Metrics)
        if self.profit < metrics.profit: return False
        ratios = all([self.ratios.gap <= metrics.ratios.gap, self.ratios.theta >= metrics.ratios.theta])
        zscore = abs(self.zscore) >= abs(metrics.zscore)
        quality = self.quality >= metrics.quality
        gamma = True if metrics.gamma is None else abs(self.gamma) <= abs(metrics.gamma)
        theta = True if metrics.theta is None else self.theta >= metrics.theta
        vega = True if metrics.vega is None else self.vega > metrics.vega
        return all([ratios, zscore, quality, gamma, theta, vega])

    @property
    def profit(self): return self.valuation - self.market
    @property
    def quality(self): return np.abs(self.zscore) * self.profit / max(self.gap, 1e-12)
    @property
    def risk(self): return abs(self.gamma) + abs(self.vega) + max(0, -self.theta)

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
    def ratios(self):
        gap = self.gap / max(self.profit, 1e-12)
        theta = self.theta / max(self.profit, 1e-12)
        return Ratios(gap=gap, theta=theta)

    @property
    @abstractmethod
    def zscore(self): pass
    @abstractmethod
    def create(self, dataframe, *args, position, quantity=1, **kwargs): pass

    @property
    def identity(self): return self.__identity
    @property
    def legs(self): return self.__legs


class Fly(Spread, register=Concepts.Strategies.Spread.FLY):
    def create(self, legs, *args, position, quantity=1, **kwargs):
        assert len(legs) == 3
        legs = legs.sort_values("strike")
        hedge = Concepts.Securities.Position(-int(position))
        legs["quantity"] = list(map(lambda x: x * quantity, [1, 2, 1]))
        legs["position"] = [hedge, position, hedge]
        legs["spread"] = Concepts.Strategies.Spread.FLY
        return legs

    @property
    def zscore(self):
        left, center, right = self.legs["zscore"].to_numpy()
        return center - (left + right) / 2


class Calender(Spread, register=Concepts.Strategies.Spread.CALENDAR):
    def create(self, legs, *args, position, quantity=1, **kwargs):
        assert len(legs) == 2
        legs = legs.sort_values("dte")
        hedge = Concepts.Securities.Position(-int(position))
        legs["quantity"] = list(map(lambda x: x * quantity, [1, 1]))
        legs["position"] = [hedge, position]
        legs["spread"] = Concepts.Strategies.Spread.CALENDAR
        return legs

    @property
    def zscore(self):
        near, far = self.legs["zscore"].to_numpy()
        return far - near



