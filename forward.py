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

from support.concepts import DateRange
from support.finance import Concepts
from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ForwardCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class ForwardError(Exception): pass
class PrimaryForwardError(ForwardError): pass
class SecondaryForwardError(ForwardError): pass


class ForwardCalculator(Logging):
    def __init__(self, *args, weights, spreads, samplesize=5, **kwargs):
        assert callable(weights) and callable(spreads)
        assert self.arguments(weights) == ["spread", "supply", "demand"]
        assert self.arguments(spreads) == ["spread", "spot"]
        super().__init__(*args, **kwargs)
        self.__samplesize = int(samplesize)
        self.__weights = weights
        self.__spreads = spreads

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if bool(options.empty): return options
        forward = self.calculator(options, *args, **kwargs)
        forward = pd.concat(list(forward), axis=0)
        forward = forward.sort_values(by=["ticker", "expire", "strike"], ascending=[True, True, True], inplace=False)
        forward = forward.reset_index(drop=True, inplace=False)
        self.alert(forward)
        return forward

    def calculator(self, options, *args, **kwargs):
        for _, options in options.groupby(["ticker", "expire"]):
            try: options = self.primary(options, *args, **kwargs)
            except PrimaryForwardError:
                try: options = self.secondary(options, *args, **kwargs)
                except SecondaryForwardError:
                    options = self.tertiary(options, *args, **kwargs)
            yield options

    def alert(self, dataframe):
        instrument = str(Concepts.Securities.Instrument.OPTION).title()
        tickers = "|".join(list(dataframe["ticker"].unique()))
        expires = DateRange.create(list(dataframe["expire"].unique()))
        expires = f"{expires.minimum.strftime('%Y%m%d')}->{expires.maximum.strftime('%Y%m%d')}"
        self.console("Calculated", f"{str(instrument)}[{str(tickers)}, {str(expires)}, {len(dataframe):.0f}]")

    def primary(self, options, *args, **kwargs):
        samples = self.samples(options, *args, **kwargs)
        if len(samples) <= self.samplesize: raise PrimaryForwardError()
        samples = samples.reset_index(drop=False, inplace=False).drop(columns=["ticker", "expire"], inplace=False)
        spreads = self.spreads(samples["spread"], samples["spot"]).squeeze()
        samples = samples.where(spreads).dropna(how="all", inplace=False)
        weights = self.weights(samples["supply"], samples["demand"], samples["spread"]).to_numpy()
        difference = samples["difference"].to_numpy()
        strikes = samples["strike"].to_numpy()
        forward, discount, error = self.regression(difference, strikes, weights)
        options = options.assign(forward=forward, discount=discount, error=error)
        return options

    @staticmethod
    def secondary(options, *args, interest=np.NaN, dividends=np.NaN, **kwargs):
        if interest is np.NaN or dividends is np.NaN: raise SecondaryForwardError()
        rate = interest - dividends
        discount = np.exp(rate * options["tau"])
        forward = options["spot"] * discount
        options = pd.concat([options, forward, discount], axis=1)
        options["error"] = np.NaN
        return options

    @staticmethod
    def tertiary(options, *args, **kwargs):
        options = options.assign(forward=np.NaN, discount=np.NaN, error=np.NaN)
        return options

    @staticmethod
    def samples(options, *args, **kwargs):
        samples = options.pivot_table(index=["ticker", "expire", "strike"], columns="option", values=["median", "spread", "supply", "demand", "spot"], sort=False).sort_index()
        if set(Concepts.Securities.Option) - set(samples.columns.get_level_values("option")): return pd.DataFrame(columns=samples.columns)
        validity = [samples[index].notna() for index in list(product(["median", "spread"], list(Concepts.Securities.Option)))]
        samples = samples[np.logical_and.reduce(validity)]
        if bool(samples.empty): return pd.DataFrame(columns=samples.columns)
        difference = (samples["median", Concepts.Securities.Option.CALL] - samples["median", Concepts.Securities.Option.PUT]).rename("difference")
        spread = (samples["spread", Concepts.Securities.Option.CALL] + samples["spread", Concepts.Securities.Option.PUT]).rename("spread")
        supply = (samples["supply", Concepts.Securities.Option.CALL] + samples["supply", Concepts.Securities.Option.PUT]).rename("supply")
        demand = (samples["demand", Concepts.Securities.Option.CALL] + samples["demand", Concepts.Securities.Option.PUT]).rename("demand")
        spot = (samples["spot", Concepts.Securities.Option.CALL] + samples["spot", Concepts.Securities.Option.PUT]).rename("spot") / 2
        samples = pd.concat([difference, spread, supply, demand, spot], axis=1)
        return samples

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
    def samplesize(self): return self.__samplesize
    @property
    def weights(self): return self.__weights
    @property
    def spreads(self): return self.__spreads


