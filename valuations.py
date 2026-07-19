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

from finance.enumerations import Instrument
from finance.logging import Logging

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
def discount(ρ, τ): return math.exp(-ρ * τ)

@njit(cache=True, inline="always")
def zitm(x, k, τ, σ, r, q):
    if x <= 0.0 or k <= 0.0 or σ <= 0.0 or τ <= 0.0: return math.nan
    fτσ = σ * math.sqrt(τ)
    return (math.log(x / k) + (r - q + 0.5 * σ * σ) * τ) / fτσ

@njit(cache=True, inline="always")
def zotm(x, k, τ, σ, r, q):
    if x <= 0.0 or k <= 0.0 or σ <= 0.0 or τ <= 0.0: return math.nan
    fτσ = σ * math.sqrt(τ)
    return zitm(x, k, τ, σ, r, q) - fτσ

@njit(cache=True, inline="always")
def valid(x, k, τ, i, r, q):
    positive = (x > 0.0 and k > 0.0 and τ > 0.0)
    option = (i == 1 or i == -1)
    finite = (math.isfinite(x) and math.isfinite(k) and math.isfinite(τ) and math.isfinite(r) and math.isfinite(q))
    return positive and option and finite

@njit(cache=True, inline="always")
def value(x, k, τ, σ, i, r, q):
    if not valid(x, k, τ, i, r, q) or σ <= 0.0 or not math.isfinite(σ): return math.nan
    zx = zitm(x, k, τ, σ, r, q)
    zk = zx - σ * math.sqrt(τ)
    return i * (x * discount(q, τ) * normcdf(i * zx) - k * discount(r, τ) * normcdf(i * zk))


@njit(cache=True)
def calculation(x, k, τ, σ, i, r, q):
    y = np.empty(len(x), dtype=np.float64)
    for idx in range(len(x)):
        y[idx] = value(x[idx], k[idx], τ[idx], σ[idx], i[idx], r, q)
    return y


class ValuationCalculator(Logging):
    def __call__(self, options, /, interest, dividends, include=False, **kwargs):
        assert isinstance(options, pd.DataFrame)
        x = options["spot"].to_numpy(np.float64)
        k = options["strike"].to_numpy(np.float64)
        τ = options["tau"].to_numpy(np.float64) / 365
        σ = options["volatility"].to_numpy(np.float64)
        i = options["option"].apply(int).to_numpy(np.int8)
        valuations = calculation(x, k, τ, σ, i, float(interest), float(dividends))
        valuations = pd.Series(valuations, name="bsm", index=options.index)
        options = pd.concat([options, valuations], axis=1)
        self.results(options, title="Calculated", instrument=Instrument.OPTION)
        return options



