# -*- coding: utf-8 -*-
"""
Created on Tues Mar 24 2026
@name:   Forward Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from typing import Callable
from itertools import product
from dataclasses import dataclass

from support.concepts import DateRange
from support.finance import Concepts
from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ForwardCalculator", "ForwardHyperparams"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


def calculation(c, p, k, w):
    y, x, w = (c - p), k, np.sqrt(w)
    yw, xw = y * w, np.column_stack([np.ones_like(x), x]) * w[:, None]
    (a, b), *_ = np.linalg.lstsq(xw, yw, rcond=None)
    ε = np.sqrt(np.average((y - (a + b * x)) ** 2, weights=w))
    return - a / b, ε


@dataclass(frozen=True)
class MethodHyperParam:
    name: str; epsilon: float; function: Callable

    def __str__(self): return str(self.name)
    def __call__(self, spread, supply, demand):
        arguments = [spread, supply, demand]
        parameters = dict(epsilon=self.epsilon)
        return self.function(*arguments, **parameters)


class ForwardHyperparams:
    class Weights:
        SQRT = MethodHyperParam("sqrt", 1e-6, lambda spread, supply, demand, /, epsilon: np.sqrt((supply + demand).clip(lower=0.0)) / spread.clip(lower=epsilon))
        TOTAL = MethodHyperParam("total", 1e-6, lambda spread, supply, demand, /, epsilon: (supply + demand) / spread.clip(lower=epsilon))
        UNSIZED = MethodHyperParam("unsized", 1e-6, lambda spread, supply, demand, /, epsilon: 1.0 / spread.clip(lower=epsilon))


class ForwardCalculator(Logging):
    def __init__(self, *args, weights, **kwargs):
        super().__init__(*args, **kwargs)
        self.__weights = weights

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if bool(options.empty): return options
        pairs = options.pivot_table(index=["ticker", "expire", "strike"], columns="option", values=["median", "spread", "supply", "demand", "discount"], sort=False).sort_index()
        validity = [pairs[index].notna() for index in list(product(["median", "spread"], list(Concepts.Securities.Option)))]
        pairs = pairs[np.logical_and.reduce(validity)]
        strikes = pairs.index.get_level_values("strike").astype(float)
        spread = pairs["spread", Concepts.Securities.Option.CALL] + pairs["spread", Concepts.Securities.Option.PUT]
        supply = pairs["supply", Concepts.Securities.Option.CALL] + pairs["supply", Concepts.Securities.Option.PUT]
        demand = pairs["demand", Concepts.Securities.Option.CALL] + pairs["demand", Concepts.Securities.Option.PUT]
        weights = self.weights(spread, supply, demand)
        forward, error = calculation(pairs["median", Concepts.Securities.Option.CALL], pairs["median", Concepts.Securities.Option.PUT], strikes, weights)
        options["forward"] = forward
        options["error"] = error
        self.alert(options)
        return options

    def alert(self, dataframe):
        instrument = str(Concepts.Securities.Instrument.OPTION).title()
        tickers = "|".join(list(dataframe["ticker"].unique()))
        expires = DateRange.create(list(dataframe["expire"].unique()))
        expires = f"{expires.minimum.strftime('%Y%m%d')}->{expires.maximum.strftime('%Y%m%d')}"
        self.console("Calculated", f"{str(instrument)}[{str(tickers)}, {str(expires)}, {len(dataframe):.0f}]")

    @property
    def weights(self): return self.__weights



