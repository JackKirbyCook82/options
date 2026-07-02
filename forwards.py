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

from finance.variables import Enumerations
from finance.logging import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ForwardCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class ForwardError(Exception): pass
class ForwardSampleError(ForwardError): pass


class ForwardCalculator(Logging):
    def __init__(self, *args, tight=0.05, samplesize=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.__samplesize = int(samplesize)
        self.__tightness = float(tight)

    def __call__(self, options, *args, include=False, **kwargs):
        assert isinstance(options, pd.DataFrame)
        forward = self.generate(options, *args, **kwargs)
        forward = forward.sort_index(inplace=False)
        self.results(forward, title="Calculated", instrument=Enumerations.Instrument.OPTION)
        return forward

    def generate(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        generator = self.generator(options, *args, **kwargs)
        forwards = list(generator)
        if bool(forwards): forward = pd.concat(forwards, axis=0)
        else: forward = pd.DataFrame(columns=options.columns)
        return forward

    def generator(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        for (ticker, expire), options in options.groupby(["ticker", "expire"]):
            spot = options["spot"].dropna(inplace=False).to_numpy()
            tau = options["tau"].dropna(inplace=False).to_numpy()
            constants = dict(spot=spot[0], tau=tau[0])
            assert (tau[0] == tau).all() and (spot[0] == spot).all()
            try:
                samples = self.samples(options, *args, **kwargs)
                mask = samples["gap"] / samples["median"] <= self.tightness
                samples = samples.where(mask).dropna(how="all", inplace=False)
                weights = self.weights(samples, *args, **kwargs)
                if len(samples) >= self.samplesize:
                    forwards = self.primary(samples, weights, *args, **constants, **kwargs)
                    options = options.assign(**forwards)
                    self.console("Regression", f"Options[{ticker}, {expire.strftime('%Y%m%d')}, {len(options.index)}]")
                    yield options
                else:
                    forwards = self.secondary(samples, weights, *args, **constants, **kwargs)
                    options = options.assign(**forwards)
                    self.console("AverageCarry", f"Options[{ticker}, {expire.strftime('%Y%m%d')}, {len(options.index)}]")
                    yield options
            except ForwardSampleError:
                forwards = self.tertiary(*args, **constants, **kwargs)
                options = options.assign(**forwards)
                self.console("SingleCarry", f"Options[{ticker}, {expire.strftime('%Y%m%d')}, {len(options.index)}]")
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
        try: forward = np.average(forwards, weights=weights)
        except ZeroDivisionError: raise ForwardSampleError()
        return dict(forward=forward, discount=discount, error=np.NaN)

    @staticmethod
    def tertiary(*args, spot, tau, interest, dividends, **kwargs):
        discount = np.exp(tau * (interest - dividends))
        forward = spot * discount
        return dict(forward=forward, discount=discount, error=np.NaN)

    @staticmethod
    def samples(options, *args, **kwargs):
        samples = options.pivot_table(index=["ticker", "expire", "strike"], columns="option", values=["median", "gap", "supply", "demand"], sort=False).sort_index()
        if set(Enumerations.Option) - set(samples.columns.get_level_values("option")): raise ForwardSampleError()
        validity = [samples[index].notna() for index in list(product(["median", "gap"], list(Enumerations.Option)))]
        samples = samples[np.logical_and.reduce(validity)]
        difference = (samples["median", Enumerations.Option.CALL] - samples["median", Enumerations.Option.PUT]).rename("difference")
        supply = (samples["supply", Enumerations.Option.CALL] + samples["supply", Enumerations.Option.PUT]).rename("supply")
        demand = (samples["demand", Enumerations.Option.CALL] + samples["demand", Enumerations.Option.PUT]).rename("demand")
        median = (samples["median", Enumerations.Option.CALL] + samples["median", Enumerations.Option.PUT]).rename("median")
        gap = (samples["gap", Enumerations.Option.CALL] + samples["gap", Enumerations.Option.PUT]).rename("gap")
        strike = samples.index.get_level_values("strike").to_series(index=samples.index)
        samples = pd.concat([strike, difference, median, gap, supply, demand], axis=1)
        samples = samples.reset_index(drop=True, inplace=False)
        return samples

    @staticmethod
    def weights(samples, *args, **kwargs):
        activity = np.sqrt((samples["supply"] + samples["demand"]).clip(lower=0.0))
        weights = activity / samples["gap"].clip(lower=1e-6)
        return weights

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
    def tightness(self): return self.__tightness




