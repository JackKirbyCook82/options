# -*- coding: utf-8 -*-
"""
Created on Tues May 12 2026
@name:   Option Acquisition Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from typing import Optional
from functools import total_ordering
from dataclasses import dataclass, fields

from finance.enumerations import Strategy
from finance.logging import Logging
from options.spreads import Spread

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["AcquisitionCalculator", "Acquisition", "Metrics"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@total_ordering
@dataclass(frozen=True)
class Profit:
    value: float; cost: float

    def __lt__(self, other): return float(self) < float(other)
    def __float__(self): return self.value - self.cost

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
class Metrics:
    ratios: Ratios; zscore: float; profit: float

    @classmethod
    def create(cls, /, ratios, zscore, profit):
        assert isinstance(ratios, dict) and isinstance(zscore, float) and isinstance(profit, float)
        ratios = {field.name: ratios.get(field.name, None) for field in fields(Ratios)}
        ratios = Ratios(**ratios)
        return cls(ratios, zscore, profit)


class Acquisition(Spread):
    @property
    def score(self): return Score(self.profit, self.quality, self.risk)
    @property
    def profit(self): return Profit(self.value, self.cost)
    @property
    def quality(self): return Quality(self.zscore, self.gap, self.profit)
    @property
    def risk(self): return Risk(self.gamma, self.theta, self.vega)

    @property
    def ratios(self):
        gamma = self.gamma / max(float(self.profit), 1e-12)
        theta = self.theta / max(float(self.profit), 1e-12)
        vega = self.vega / max(float(self.profit), 1e-12)
        gap = self.gap / max(float(self.profit), 1e-12)
        return Ratios(gamma=gamma, theta=theta, vega=vega, gap=gap)

    @property
    def zscore(self):
        if self.strategy is Strategy.FLY:
            left, center, right = self.securities["zscore"].to_numpy()
            return center - (left + right) / 2
        elif self.strategy is Strategy.CALENDAR:
            near, far = self.securities["zscore"].to_numpy()
            return far - near
        else: raise ValueError(self.strategy)


class AcquisitionCreator(object):
    pass


class AcquisitionCalculator(Logging):
    def __init__(self, *args, strategies, proximity=1, **kwargs):
        assert isinstance(proximity, int) and proximity > 0
        assert [Strategy(strategy) for strategy in strategies]
        super().__init__(*args, **kwargs)
        self.__strategies = strategies
        self.__proximity = proximity

    def __call__(self, options, **kwargs):
        assert isinstance(options, pd.DataFrame)

    @property
    def strategies(self): return self.__strategies
    @property
    def proximity(self): return self.__proximity


