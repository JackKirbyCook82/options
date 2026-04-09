# -*- coding: utf-8 -*-
"""
Created on Tues Mar 24 2026
@name:   Valuation Objects
@author: Jack Kirby Cook

"""

import math
import numpy as np
import pandas as pd
from numba import njit

from support.concepts import DateRange
from support.finance import Concepts
from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ValuationCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@njit(cache=True, inline="always")
def normcdf(z): return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

@njit(cache=True, inline="always")
def normpdf(z): return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)

@njit(cache=True, inline="always")
def discount(r, τ): return math.exp(-r * τ)

@njit(cache=True, inline="always")
def zitm(x, k, τ, σ, r):
    if x <= 0.0 or k <= 0.0 or σ <= 0.0 or τ <= 0.0: return math.nan
    fτσ = σ * math.sqrt(τ)
    return (math.log(x / k) + (r + 0.5 * σ * σ) * τ) / fτσ

@njit(cache=True, inline="always")
def zotm(x, k, τ, σ, r):
    if x <= 0.0 or k <= 0.0 or σ <= 0.0 or τ <= 0.0: return math.nan
    fτσ = σ * math.sqrt(τ)
    return zitm(x, k, τ, σ, r) - fτσ

@njit(cache=True, inline="always")
def valid(x, k, τ, i):
    positive = (x > 0.0 and k > 0.0 and τ > 0.0)
    option = (i == 1 or i == -1)
    finite = (math.isfinite(x) and math.isfinite(k) and math.isfinite(τ))
    return positive and option and finite

@njit(cache=True, inline="always")
def blackscholes(x, k, τ, σ, i, r):
    if not valid(x, k, τ, i) or σ <= 0.0 or not math.isfinite(σ): return math.nan
    zx = zitm(x, k, τ, σ, r)
    zk = zx - σ * math.sqrt(τ)
    dcf = discount(r, τ)
    return i * (x * normcdf(i * zx) - k * dcf * normcdf(i * zk))


@njit(cache=True)
def calculation(x, k, τ, σ, i, r):
    y = np.empty(len(x), dtype=np.float64)
    for idx in range(len(x)):
        y[idx] = blackscholes(x[idx], k[idx], τ[idx], σ[idx], i[idx], r)
    return y


class ValuationCalculator(Logging):
    def __call__(self, options, *args, interest, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if bool(options.empty): return options
        x = options["spot"].to_numpy(np.float64)
        k = options["strike"].to_numpy(np.float64)
        τ = options["tau"].to_numpy(np.float64)
        σ = options["volatility"].to_numpy(np.float64)
        i = options["option"].apply(int).to_numpy(np.int8)
        options["value"] = calculation(x, k, τ, σ, i, float(interest))
        self.alert(options)
        return options

    def alert(self, dataframe):
        instrument = str(Concepts.Securities.Instrument.OPTION).title()
        tickers = "|".join(list(dataframe["ticker"].unique()))
        expires = DateRange.create(list(dataframe["expire"].unique()))
        expires = f"{expires.minimum.strftime('%Y%m%d')}->{expires.maximum.strftime('%Y%m%d')}"
        self.console("Calculated", f"{str(instrument)}[{str(tickers)}, {str(expires)}, {len(dataframe):.0f}]")



