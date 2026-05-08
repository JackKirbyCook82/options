# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 2026
@name:   Option Dataset Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

from support.concepts import NumRange, DateRange
from support.equations import Equations
from support.finance import Alerting
from support.surface import Surface

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DatasetCalculator", "Dataset"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True)
class Variables: tau: float; mae: float; tiv: float = None

@dataclass(frozen=False)
class Dataset:
    scatter: pd.DataFrame = None
    center: Variables = None
    radius: Variables = None
    surface: Surface = None

    def __bool__(self): return self.scatter is not None and not self.scatter.empty
    def __len__(self): return len(self.scatter)

    def __str__(self):
        tickers = "|".join(list(self.scatter["ticker"].unique()))
        expires = DateRange.create(list(self.scatter["expire"].unique()))
        expires = f"{expires.minimum.strftime('%Y%m%d')}->{expires.maximum.strftime('%Y%m%d')}"
        inner, outer = self.inner, self.outer
        variables = [self.string(axis, inner, outer) for axis in list("xy")]
        return "\n".join([f"{tickers}|{expires}[{len(self):.0f}]"] + variables)

    def __post_init__(self):
        mask = (self.scatter["tau"].notna() & self.scatter["mae"].notna() & self.scatter["tiv"].notna())
        center = self.scatter.attrs.get("center", None)
        radius = self.scatter.attrs.get("radius", None)
        scatter = self.scatter.loc[mask].copy()
        self.scatter = scatter
        self.center = center
        self.radius = radius

    @staticmethod
    def string(axis, inner, outer):
        inner, outer = getattr(inner, axis), getattr(outer, axis)
        inner = f"({inner.minimum:.03f}, {inner.maximum:.03f})"
        outer = f"({outer.minimum:.03f}, {outer.maximum:.03f})"
        string = f"{str(axis).upper()}[{inner} ∈ {outer}]"
        return string

    @staticmethod
    def average(axis, decimals): return round((axis.minimum + axis.maximum) / 2, decimals)
    @staticmethod
    def distance(axis, decimals): return round((axis.maximum - axis.minimum) / 2, decimals)

    @property
    def inner(self):
        tau = NumRange.create([self.scatter["tau"].min(), self.scatter["tau"].max()])
        mae = NumRange.create([self.scatter["mae"].min(), self.scatter["mae"].max()])
        tiv = NumRange.create([self.scatter["tiv"].min(), self.scatter["tiv"].max()])
        return Variables(tau=tau, mae=mae, tiv=tiv)

    @property
    def outer(self):
        tau = NumRange.create([self.center.tau - self.radius.tau, self.center.tau + self.radius.tau])
        mae = NumRange.create([self.center.mae - self.radius.mae, self.center.mae + self.radius.mae])
        return Variables(tau=tau, mae=mae)


class DatasetCalculator(Equations, Alerting):
    def __call__(self, dataset, *args, **kwargs):
        assert isinstance(dataset, Dataset)
        function = lambda row: dataset.surface(row["tau"], row["mae"])
        hat = dataset.scatter[["tau", "mae"]].apply(function, axis=1)





