# -*- coding: utf-8 -*-
"""
Created on Sat May 16 2026
@name:   Option Spread Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from typing import Optional
from types import SimpleNamespace
from functools import total_ordering
from dataclasses import dataclass, fields

from finance.enumerations import Spread
from finance.osi import OSI
from support.custom import DateRange

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Spread", "Metrics"]
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


class Spread(object):
    def __init__(self, legs): self.legs = legs
    def __new__(cls, legs):
        assert isinstance(legs, pd.DataFrame)
        assert len(legs["ticker"].unique()) == 1
        assert len(legs["type"].unique()) == 1
        instance = super().__new__(cls)
        return instance

    @property
    def signature(self): return tuple((str(record.osi), int(record.position), int(record.quantity)) for record in self.records)
    @property
    def records(self):
        function = lambda value: (str(value[0]), int(value[1]), int(value[2]))
        keys, values = ["osi", "position", "quantity"], list(zip(self.osi, self.position, self.quantity))
        records = [dict(zip(keys, value)) for value in sorted(values, key=function)]
        return [SimpleNamespace(**record) for record in records]

    @property
    def osi(self): return self.legs[["ticker", "expire", "option", "strike"]].apply(OSI, axis=1)
    @property
    def cost(self): return (self.legs["median"] * self.position.map(int) * self.quantity).sum()

    @property
    def score(self): return Score(self.profit, self.quality, self.risk)
    @property
    def profit(self): return Profit(self.valuation, self.market)
    @property
    def quality(self): return Quality(self.zscore, self.gap, self.profit)
    @property
    def risk(self): return Risk(self.gamma, self.theta, self.vega)

    @property
    def gamma(self): return (self.legs["gamma"] * self.position.map(int) * self.quantity).sum()
    @property
    def theta(self): return (self.legs["theta"] * self.position.map(int) * self.quantity).sum()
    @property
    def vega(self): return (self.legs["vega"] * self.position.map(int) * self.quantity).sum()

    @property
    def valuation(self): return (self.legs["value"] * self.position.map(int) * self.quantity).sum()
    @property
    def market(self): return (self.legs["median"] * self.position.map(int) * self.quantity).sum()
    @property
    def gap(self): return (self.legs["gap"] * self.quantity).sum()

    @property
    def ratios(self):
        gamma = self.gamma / max(float(self.profit), 1e-12)
        theta = self.theta / max(float(self.profit), 1e-12)
        vega = self.vega / max(float(self.profit), 1e-12)
        gap = self.gap / max(float(self.profit), 1e-12)
        return Ratios(gamma=gamma, theta=theta, vega=vega, gap=gap)

    @property
    def zscore(self):
        if self.type is Spread.FLY:
            left, center, right = self.legs["zscore"].to_numpy()
            return center - (left + right) / 2
        elif self.type is Spread.CALENDAR:
            near, far = self.legs["zscore"].to_numpy()
            return far - near
        else: raise ValueError(self.type)

    @property
    def tightness(self): return self.legs["tightness"].max()
    @property
    def moneyness(self): return self.legs["moneyness"].max()
    @property
    def activity(self): return self.legs["activity"].min()

    @property
    def position(self): return self.legs["position"]
    @property
    def quantity(self): return self.legs["quantity"]

    @property
    def ticker(self): return str(list(self.legs["ticker"].unique()[0]))
    @property
    def expires(self): return DateRange.create(self.legs["expire"].to_list())
    @property
    def type(self): return list(self.legs["type"].unique())[0]



