# -*- coding: utf-8 -*-
"""
Created on Tues Mar 24 2026
@name:   Forward Objects
@author: Jack Kirby Cook

"""

import inspect
import numpy as np
import pandas as pd
from typing import Callable
from itertools import product
from dataclasses import dataclass

from support.finance import Concepts
from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ForwardCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class ForwardCalculator(Logging):
    def __init__(self, *args, weights, **kwargs):
        assert callable(weights)
        signature = inspect.signature(weights).parameters.items()
        arguments = [variable for variable, details in signature if details.kind in (details.POSITIONAL_OR_KEYWORD, details.POSITIONAL_OR_KEYWORD)]
        assert arguments == ["spread", "supply", "demand"]
        super().__init__(*args, **kwargs)
        self.__weights = weights

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if bool(options.empty): return options
        calculated = self.calculator(options, *args, **kwargs)
        calculated = pd.concat(list(calculated), axis=0)
        calculated = calculated.sort_values(by=["ticker", "expire", "strike"], ascending=[True, True, True], inplace=False)
        return calculated

    def calculator(self, options, *args, **kwargs):
        for (ticker, expire), dataframe in options.groupby(["ticker", "expire"]):
            pairwise = dataframe.pivot_table(index=["ticker", "expire", "strike"], columns="option", values=["median", "spread", "supply", "demand"], sort=False).sort_index()
            validity = [pairwise[index].notna() for index in list(product(["median", "spread"], list(Concepts.Securities.Option)))]
            pairwise = pairwise[np.logical_and.reduce(validity)]
            calculated = self.calculate(pairwise, *args, **kwargs)
            calculated = dataframe.assign(**calculated)
            self.alert(ticker, expire, len(calculated), len(pairwise))
            yield calculated

    def calculate(self, options, *args, **kwargs):
        options = options.reset_index(drop=False, inplace=False)
        options = options.drop(columns=["ticker", "expire"], inplace=False)
        difference = (options["median", Concepts.Securities.Option.CALL] - options["median", Concepts.Securities.Option.PUT]).rename("difference")
        spread = (options["spread", Concepts.Securities.Option.CALL] + options["spread", Concepts.Securities.Option.PUT]).rename("spread")
        supply = (options["supply", Concepts.Securities.Option.CALL] + options["supply", Concepts.Securities.Option.PUT]).rename("supply")
        demand = (options["demand", Concepts.Securities.Option.CALL] + options["demand", Concepts.Securities.Option.PUT]).rename("demand")
        weights = self.weights(supply, demand, spread)
        forward, discount, error = self.regression(difference, options["strike"], weights)
        return dict(forward=forward, discount=discount, error=error)

    def alert(self, ticker, expire, size, sample):
        instrument = str(Concepts.Securities.Instrument.OPTION).title()
        self.console("Calculated", f"{str(instrument)}[{str(ticker)}, {expire.strftime('%Y%m%d')}, {int(size):.0f}|{int(sample):.0f}]")

    @staticmethod
    def regression(y, x, w):
        w = np.sqrt(w)
        yw, xw = y * w, np.column_stack([np.ones_like(x), x]) * w[:, None]
        (a, b), *_ = np.linalg.lstsq(xw, yw, rcond=None)
        ε = np.sqrt(np.average((y - (a + b * x)) ** 2, weights=w))
        return - a / b, - b, ε

    @property
    def weights(self): return self.__weights


