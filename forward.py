# -*- coding: utf-8 -*-
"""
Created on Tues Mar 24 2026
@name:   Forward Objects
@author: Jack Kirby Cook

"""

import inspect
import numpy as np
import pandas as pd
from itertools import product

from support.finance import Concepts
from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ForwardCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class ForwardCalculator(Logging):
    def __init__(self, *args, weights, spreads, **kwargs):
        assert callable(weights) and callable(spreads)
        assert self.arguments(weights) == ["spread", "supply", "demand"]
        assert self.arguments(spreads) == ["spread", "spot"]
        super().__init__(*args, **kwargs)
        self.__weights = weights
        self.__spreads = spreads

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if bool(options.empty): return options
        forward = self.calculator(options, *args, **kwargs)
        forward = pd.concat(list(forward), axis=0)
        forward = forward.sort_values(by=["ticker", "expire", "strike"], ascending=[True, True, True], inplace=False)
        forward = forward.reset_index(drop=True, inplace=False)
        return forward

    def calculator(self, options, *args, **kwargs):
        for (ticker, expire), settlements in options.groupby(["ticker", "expire"]):
            selection = self.selection(settlements, *args, **kwargs)
            forward = self.calculate(selection, *args, **kwargs)
            forward = settlements.assign(**forward)
            self.alert(ticker, expire, len(forward), len(selection))
            yield forward

    def calculate(self, selection, *args, **kwargs):
        if len(selection) <= 2: return dict(forward=np.NaN, discount=np.NaN, error=np.NaN)
        selection = selection.reset_index(drop=False, inplace=False).drop(columns=["ticker", "expire"], inplace=False)
        spreads = self.spreads(selection["spread"], selection["spot"]).squeeze()
        selection = selection.where(spreads).dropna(how="all", inplace=False)
        weights = self.weights(selection["supply"], selection["demand"], selection["spread"]).to_numpy()
        difference = selection["difference"].to_numpy()
        strikes = selection["strike"].to_numpy()
        forward, discount, error = self.regression(difference, strikes, weights)
        return dict(forward=forward, discount=discount, error=error)

    def alert(self, ticker, expire, size, sample):
        instrument = str(Concepts.Securities.Instrument.OPTION).title()
        self.console("Calculated", f"{str(instrument)}[{str(ticker)}, {expire.strftime('%Y%m%d')}, {int(size):.0f}|{int(sample):.0f}]")

    @staticmethod
    def selection(settlements, *args, **kwargs):
        selection = settlements.pivot_table(index=["ticker", "expire", "strike"], columns="option", values=["median", "spread", "supply", "demand", "spot"], sort=False).sort_index()
        if set(Concepts.Securities.Option) - set(selection.columns.get_level_values("option")): return pd.DataFrame(columns=selection.columns)
        validity = [selection[index].notna() for index in list(product(["median", "spread"], list(Concepts.Securities.Option)))]
        selection = selection[np.logical_and.reduce(validity)]
        if bool(selection.empty): return pd.DataFrame(columns=selection.columns)
        difference = (selection["median", Concepts.Securities.Option.CALL] - selection["median", Concepts.Securities.Option.PUT]).rename("difference")
        spread = (selection["spread", Concepts.Securities.Option.CALL] + selection["spread", Concepts.Securities.Option.PUT]).rename("spread")
        supply = (selection["supply", Concepts.Securities.Option.CALL] + selection["supply", Concepts.Securities.Option.PUT]).rename("supply")
        demand = (selection["demand", Concepts.Securities.Option.CALL] + selection["demand", Concepts.Securities.Option.PUT]).rename("demand")
        spot = (selection["spot", Concepts.Securities.Option.CALL] + selection["spot", Concepts.Securities.Option.PUT]).rename("spot") / 2
        selection = pd.concat([difference, spread, supply, demand, spot], axis=1)
        return selection

    @staticmethod
    def regression(y, x, w):
        yw, xw = y * w, np.column_stack([np.ones_like(x), x]) * w[:, None]
        (a, b), *_ = np.linalg.lstsq(xw, yw, rcond=None)
        ε = np.sqrt(np.average((y - (a + b * x)) ** 2, weights=w))
        return - a / b, - b, ε

    @staticmethod
    def arguments(function):
        signature = inspect.signature(function).parameters.items()
        return [variable for variable, details in signature if details.kind in (details.POSITIONAL_OR_KEYWORD, details.POSITIONAL_OR_KEYWORD)]

    @property
    def weights(self): return self.__weights
    @property
    def spreads(self): return self.__spreads


