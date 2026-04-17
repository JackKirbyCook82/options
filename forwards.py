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

from support.finance import Concepts, Alerting
from support.calculations import Generator

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ForwardCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class ForwardError(Exception): pass
class ForwardSampleError(ForwardError): pass


class ForwardCalculator(Generator, Alerting):
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
        forward = self.generate(options, *args, **kwargs)
        forward = forward.sort_values(by=["ticker", "expire", "strike"], ascending=[True, True, True], inplace=False)
        forward = forward.reset_index(drop=True, inplace=False)
        self.alert(forward, title="Calculated", instrument=Concepts.Securities.Instrument.OPTION)
        return forward

    def generator(self, options, *args, **kwargs):
        for (ticker, expire), options in options.groupby(["ticker", "expire"]):
            spot = options["spot"].dropna(inplace=False).to_numpy()
            tau = options["tau"].dropna(inplace=False).to_numpy()
            instrument = str(Concepts.Securities.Instrument.OPTION).title()
            constants = dict(spot=spot[0], tau=tau[0])
            assert (tau[0] == tau).all() and (spot[0] == spot).all()
            try:
                samples = self.samples(options, *args, **kwargs)
                spreads = self.spreads(samples["spread"], spot[0]).squeeze()
                samples = samples.where(spreads).dropna(how="all", inplace=False)
                weights = self.weights(samples["supply"], samples["demand"], samples["spread"])
                if len(samples) >= self.samplesize:
                    forwards = self.primary(samples, weights, *args, **constants, **kwargs)
                    options = options.assign(**forwards)
                    self.console("Regression", f"{instrument}[{ticker}, {expire.strftime('%Y%m%d')}, {len(options.index)}]")
                    yield options
                else:
                    forwards = self.secondary(samples, weights, *args, **constants, **kwargs)
                    options = options.assign(**forwards)
                    self.console("AverageCarry", f"{instrument}[{ticker}, {expire.strftime('%Y%m%d')}, {len(options.index)}]")
                    yield options
            except ForwardSampleError:
                forwards = self.tertiary(*args, **constants, **kwargs)
                options = options.assign(**forwards)
                self.console("SingleCarry", f"{instrument}[{ticker}, {expire.strftime('%Y%m%d')}, {len(options.index)}]")
                yield options

    def primary(self, samples, weights, *args, **kwargs):
        difference = samples["difference"].to_numpy()
        strikes = samples["strike"].to_numpy()
        weights = weights.to_numpy()
        forward, discount, error = self.regression(difference, strikes, weights)
        return dict(forward=forward, discount=discount, error=error)

    @staticmethod
    def secondary(samples, weights, *args, tau, interest, dividends, **kwargs):
        discount = np.exp(tau * (interest - dividends))
        forwards = (samples["strike"] + samples["difference"] / discount).to_numpy()
        forward = np.average(forwards, weights=weights)
        return dict(forward=forward, discount=discount, error=np.NaN)

    @staticmethod
    def tertiary(*args, spot, tau, interest, dividends, **kwargs):
        discount = np.exp(tau[0] * (interest - dividends))
        forward = spot[0] * discount
        return dict(forward=forward, discount=discount, error=np.NaN)

    @staticmethod
    def samples(options, *args, **kwargs):
        samples = options.pivot_table(index=["ticker", "expire", "strike"], columns="option", values=["median", "spread", "supply", "demand"], sort=False).sort_index()
        if set(Concepts.Securities.Option) - set(samples.columns.get_level_values("option")): raise ForwardSampleError()
        validity = [samples[index].notna() for index in list(product(["median", "spread"], list(Concepts.Securities.Option)))]
        samples = samples[np.logical_and.reduce(validity)]
        difference = (samples["median", Concepts.Securities.Option.CALL] - samples["median", Concepts.Securities.Option.PUT]).rename("difference")
        spread = (samples["spread", Concepts.Securities.Option.CALL] + samples["spread", Concepts.Securities.Option.PUT]).rename("spread")
        supply = (samples["supply", Concepts.Securities.Option.CALL] + samples["supply", Concepts.Securities.Option.PUT]).rename("supply")
        demand = (samples["demand", Concepts.Securities.Option.CALL] + samples["demand", Concepts.Securities.Option.PUT]).rename("demand")
        strike = samples.index.get_level_values("strike").to_series(index=samples.index)
        samples = pd.concat([strike, difference, spread, supply, demand], axis=1)
        samples = samples.reset_index(drop=True, inplace=False)
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



