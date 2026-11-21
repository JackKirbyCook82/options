# -*- coding: utf-8 -*-
"""
Created on Tues Mar 24 2026
@name:   Forward Objects
@author: Jack Kirby Cook
@file:   options/forwards.py

"""

import numpy as np
import pandas as pd
from types import SimpleNamespace
from dataclasses import dataclass
from itertools import product as iterprod

from finance.variables import Collection, Variables
from finance.console import LoggingDecorators
from finance.enumerations import Enumerations
from support.mixins import LoggingMixin
from support.pipeline import Pipelines

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ForwardCalculator", "Forwarding"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class ForwardError(Exception): pass
class AverageCarryError(ForwardError): pass
class SimpleCarryError(ForwardError): pass
class RegressionError(ForwardError): pass
class LinearAlgebraError(RegressionError): pass
class CoefficientError(RegressionError): pass
class RankError(RegressionError): pass

class SampleError(ForwardError): pass
class BalanceError(SampleError): pass
class SizeError(SampleError): pass

class ValidationError(ForwardError): pass
class ArgumentError(ValidationError): pass
class ParameterError(ValidationError): pass
class ToleranceError(ValidationError): pass


@dataclass(frozen=True, slots=True)
class Forwarding: forward: float; discount: float; error: float


class ForwardCalculator(LoggingMixin):
    def __init__(self, *args, forwarding, tightness=0.15, samplesize=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.__samplesize = int(samplesize)
        self.__tightness = float(tightness)
        self.__forwarding = forwarding

    @Pipelines.Function(locator=0)
    @LoggingDecorators.Function(Collection[Variables.Ticker], Collection[Variables.Expire], title="Calculated")
    def __call__(self, options, /, **kwargs):
        if options is None: return None
        assert isinstance(options, pd.DataFrame)
        if options.empty: return None
        generator = self.calculator(options, **kwargs)
        calculated = list(generator)
        if not calculated: return None
        calculated = pd.concat(calculated, axis=0)
        calculated = calculated.sort_index(inplace=False)
        return calculated

    def calculator(self, options, /, **kwargs):
        assert isinstance(options, pd.DataFrame)
        for (ticker, expire), options in options.groupby(["ticker", "expire"], sort=False, dropna=False):
            underlying = options["underlying"].dropna(inplace=False).to_numpy()
            try: tau = options["tau"].dropna(inplace=False).to_numpy()
            except KeyError:
                try: tau = options["trading"].dropna(inplace=False).to_numpy() / 252
                except KeyError: tau = options["calender"].dropna(inplace=False).to_numpy() / 365
            constants = dict(underlying=underlying[0], tau=tau[0])
            assert (tau[0] == tau).all() and (underlying[0] == underlying).all()

            try:
                samples = self.samples(options, **kwargs)
                finite = np.isfinite(samples["market"]) & np.isfinite(samples["gap"])
                positive = (samples["market"] > 0) & (samples["gap"] > 0)
                tightness = (samples["gap"] / samples["market"] <= self.tightness)
                mask = finite & positive & tightness
                samples = samples.loc[mask]
                weights = self.weights(samples)
                if len(samples) >= self.samplesize:
                    try:
                        forwarding = self.primary(samples, weights, **constants, **kwargs)
                        self.console(f"{ticker}, {expire.strftime('%Y/%m/%d')}", title="Regression")
                    except ForwardError:
                        forwarding = self.secondary(samples, weights, **constants, **kwargs)
                        self.console(f"{ticker}, {expire.strftime('%Y/%m/%d')}", title="AverageCarray")

                else:
                    forwarding = self.secondary(samples, weights, **constants, **kwargs)
                    self.console(f"{ticker}, {expire.strftime('%Y/%m/%d')}", title="AverageCarray")

            except ForwardError:
                forwarding = self.tertiary(**constants, **kwargs)
                self.console(f"{ticker}, {expire.strftime('%Y/%m/%d')}", title="SimpleCarray")

            options = options.assign(forward=forwarding.forward, discount=forwarding.discount)
            yield options

    def validate(self, samples, forwarding, /, underlying, tau, interest, dividends, **kwargs):
        if not np.isfinite([forwarding.forward, forwarding.discount]): raise ArgumentError()
        if forwarding.forward <= 0 or forwarding.discount <= 0: raise ArgumentError()
        if not np.isfinite([underlying, tau, interest, dividends]): raise ParameterError()
        if underlying <= 0 or tau <= 0: raise ParameterError()

        forward = underlying * np.exp((interest - dividends) * tau)
        discount = np.exp(-interest * tau)
        carry = Forwarding(forward, discount, np.nan)
        if not np.isfinite([carry.forward, carry.discount]).all(): raise ArgumentError()
        if carry.forward <= 0 or carry.discount <= 0: raise ArgumentError()

        forward = abs(forward / carry.forward - 1.0)
        discount = abs(discount / carry.discount - 1.0)
        error = SimpleNamespace(forward=forward, discount=discount)
        if error.forward > self.forwarding.forward: raise ToleranceError()
        if error.discount > self.forwarding.discount: raise ToleranceError()

        if np.isfinite(forwarding.error):
            gaps = samples["gap"].to_numpy(dtype=float)
            gaps = gaps[np.isfinite(gaps) & (gaps > 0)]
            if len(gaps) == 0: raise ValidationError()
            if forwarding.error > max(self.forwarding.error, np.median(gaps)): raise ToleranceError()

    def primary(self, samples, weights, /, **kwargs):
        difference = samples["difference"].to_numpy()
        strikes = samples["strike"].to_numpy()
        weights = weights.to_numpy()
        regression = self.regression(difference, strikes, weights)

        try: self.validate(samples, regression, **kwargs)
        except ValidationError: raise RegressionError()
        return regression

    def secondary(self, samples, weights, /, tau, interest, **kwargs):
        discount = np.exp(- interest * tau)
        forwards = (samples["strike"] + samples["difference"] / discount).to_numpy()
        weights = weights.to_numpy(dtype=float)
        valid = (np.isfinite(forwards) & np.isfinite(weights) & (weights > 0))
        if not valid: raise AverageCarryError()
        forward = np.average(forwards[valid], weights=weights[valid])
        carry = Forwarding(forward, discount, np.nan)
        parameters = dict(tau=tau, interest=interest)

        try: self.validate(samples, carry, **parameters, **kwargs)
        except ValidationError: raise AverageCarryError()
        return carry

    @staticmethod
    def tertiary(*, /, underlying, tau, interest, dividends, **kwargs):
        discount = np.exp(- interest * tau)
        forward = underlying * np.exp((interest - dividends) * tau)
        carry = Forwarding(forward, discount, np.nan)
        return carry

    @staticmethod
    def samples(options, /, **kwargs):
        samples = options.pivot_table(index=["ticker", "expire", "strike"], columns="right", values=["market", "gap", "supply", "demand"], sort=False).sort_index()
        if set(Enumerations.Right) - set(samples.columns.get_level_values("right")): raise BalanceError()
        validity = [samples[index].notna() for index in list(iterprod(["market", "gap"], list(Enumerations.Right)))]
        samples = samples[np.logical_and.reduce(validity)]

        difference = (samples["market", Enumerations.Right.CALL] - samples["market", Enumerations.Right.PUT]).rename("difference")
        supply = (samples["supply", Enumerations.Right.CALL] + samples["supply", Enumerations.Right.PUT]).rename("supply")
        demand = (samples["demand", Enumerations.Right.CALL] + samples["demand", Enumerations.Right.PUT]).rename("demand")
        market = (samples["market", Enumerations.Right.CALL] + samples["market", Enumerations.Right.PUT]).rename("market")
        gap = (samples["gap", Enumerations.Right.CALL] + samples["gap", Enumerations.Right.PUT]).rename("gap")
        strike = samples.index.get_level_values("strike").to_series(index=samples.index)

        samples = pd.concat([strike, difference, market, gap, supply, demand], axis=1)
        samples = samples.reset_index(drop=True, inplace=False)
        return samples

    @staticmethod
    def regression(y, x, w):
        y = np.asarray(y, dtype=float), np.asarray(x, dtype=float), np.asarray(w, dtype=float)
        valid = (np.isfinite(y) & np.isfinite(x) & np.isfinite(w) & (w > 0))
        y, x, w = y[valid], x[valid], w[valid]
        if len(x) < 2 or np.unique(x).size < 2: raise SizeError()

        yw = y * np.sqrt(w)
        xw = np.column_stack([np.ones_like(x), x]) * np.sqrt(w)[:, None]
        try: coefficients, residuals, rank, singular = (np.linalg.lstsq(xw, yw, rcond=None))
        except np.linalg.LinAlgError as error: raise LinearAlgebraError() from error
        if rank < 2: raise RankError()

        a, b = coefficients
        if not np.isfinite([a, b]).all() or abs(b) < 1e-10: raise CoefficientError()
        forward, discount = - a / b, b
        error = np.sqrt(np.average((y - (a + b * x)) ** 2, weights=w))
        return Forwarding(forward, discount, error)

    @staticmethod
    def weights(samples):
        activity = np.sqrt((samples["supply"] + samples["demand"]).clip(lower=0.0))
        weights = activity / samples["gap"].clip(lower=1e-6)
        return weights

    @property
    def samplesize(self): return self.__samplesize
    @property
    def tightness(self): return self.__tightness
    @property
    def forwarding(self): return self.__forwarding



