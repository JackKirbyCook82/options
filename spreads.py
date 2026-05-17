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

from support.meta import CounterMeta, RegistryMeta
from support.finance import Concepts

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Spread", "Metrics", "Ratios"]
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


class SpreadMeta(CounterMeta, RegistryMeta): pass
class Spread(ABC, metaclass=SpreadMeta):
    def __init__(self, legs, *args, **kwargs):
        assert isinstance(legs, pd.DataFrame)
        self.__identity = type(self).counter
        self.__legs = legs

#    def __call__(self, metrics, *args, **kwargs):
#        columns = ["identity"] + list(self.legs.columns)
#        if not self.qualify(metrics): return pd.DataFrame(columns=columns)
#        identity = type(self).counter
#        prospects = self.legs.assign(identity=identity)
#        return prospects

#    def qualify(self, metrics):
#        assert isinstance(metrics, Metrics)
#        if self.profit < metrics.profit: return False
#        ratios = all([self.ratios.gap <= metrics.ratios.gap, self.ratios.theta >= metrics.ratios.theta])
#        zscore = abs(self.zscore) >= abs(metrics.zscore)
#        quality = self.quality >= metrics.quality
#        gamma = True if metrics.gamma is None else abs(self.gamma) <= abs(metrics.gamma)
#        theta = True if metrics.theta is None else self.theta >= metrics.theta
#        vega = True if metrics.vega is None else self.vega > metrics.vega
#        return all([ratios, zscore, quality, gamma, theta, vega])

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
    def identity(self): return self.__identity
    @property
    def legs(self): return self.__legs


class Fly(Spread, register=Concepts.Strategies.Spread.FLY):
    @property
    def zscore(self):
        left, center, right = self.legs["zscore"].to_numpy()
        return center - (left + right) / 2


class Calender(Spread, register=Concepts.Strategies.Spread.CALENDAR):
    @property
    def zscore(self):
        near, far = self.legs["zscore"].to_numpy()
        return far - near



