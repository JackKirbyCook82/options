# -*- coding: utf-8 -*-
"""
Created on Tues Mar 24 2026
@name:   Volatility Objects
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
__all__ = ["VolatilityCalculator"]
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
def blackscholes(x, k, τ, σ, i, r):
    if not valid(x, k, τ, i) or σ <= 0.0 or not math.isfinite(σ): return math.nan
    zx = zitm(x, k, τ, σ, r)
    zk = zx - σ * math.sqrt(τ)
    dcf = discount(r, τ)
    return i * (x * normcdf(i * zx) - k * dcf * normcdf(i * zk))

@njit(cache=True, inline="always")
def error(y, x, k, τ, σ, i, r): return blackscholes(x, k, τ, σ, i, r) - y

@njit(cache=True, inline="always")
def intrinsic(x, k, τ, i, r):
    dcf = discount(r, τ)
    if i == +1: yτ = max(0.0, x - k * dcf)
    elif i == -1: yτ = max(0.0, k * dcf - x)
    return yτ

@njit(cache=True, inline="always")
def boundary(x, k, τ, i, r):
    assert i == +1 or i == -1
    dcf = discount(r, τ)
    if i == +1: yl = max(0.0, x - k * dcf); yh = x
    elif i == -1: yl = max(0.0, k * dcf - x); yh = k * dcf
    return yl, yh

@njit(cache=True, inline="always")
def valid(x, k, τ, i):
    positive = (x > 0.0 and k > 0.0 and τ > 0.0)
    option = (i == 1 or i == -1)
    finite = (math.isfinite(x) and math.isfinite(k) and math.isfinite(τ))
    return positive and option and finite

@njit(cache=True, inline="always")
def vega(x, k, τ, σ, i, r):
    """dy/dσ"""
    if x <= 0.0 or k <= 0.0 or τ <= 0.0 or σ <= 0.0: return 0.0
    zx = zitm(x, k, τ, σ, r)
    return x * normpdf(zx) * math.sqrt(τ)

@njit(cache=True, inline="always")
def adaptive(x, k, τ):
    xk = abs(math.log(x / k))
    τ = max(τ, 1.0 / 365.0)
    return min(math.sqrt((10.0 + 2.0 * xk) / τ), 20.0)

@njit(cache=True, inline="always")
def brenner(y, x, k, τ, i, r, /, low, high):
    yτ = intrinsic(x, k, τ, i, r)
    dy = max(y - yτ, 1e-12)
    x = max(x, 1e-12)
    σ = math.sqrt(2.0 * math.pi / max(τ, 1e-12)) * dy / x
    if not math.isfinite(σ): σ = 0.2
    if σ < low: σ = low
    if σ > high: σ = high
    return σ

@njit(cache=True)
def newton(y, x, k, τ, i, r, /, low, high, tol, iters):
    σ = brenner(y, x, k, τ, i, r, low, high)
    for _ in range(iters):
        err = error(y, x, k, τ, σ, i, r)
        if abs(err) <= tol: return σ
        dydσ = vega(x, k, τ, σ, i, r)
        if not math.isfinite(dydσ) or dydσ <= 1e-12: return math.nan
        step = err / dydσ
        limit = 0.5 * max(σ, 0.10)
        if step > limit: step = limit
        elif step < -limit: step = -limit
        update = σ - step
        if not math.isfinite(update): return math.nan
        if update <= low: update = 0.5 * (σ + low)
        elif update >= high: update = 0.5 * (σ + high)
        if abs(update - σ) <= 1e-10: return update
        σ = update
    return math.nan

@njit(cache=True)
def bisection(y, x, k, τ, i, r, /, low, high, tol, iters):
    yl = error(y, x, k, τ, low, i, r)
    yh = error(y, x, k, τ, high, i, r)
    if not math.isfinite(yl) or not math.isfinite(yh): return math.nan
    if yl == 0.0: return low
    if yh == 0.0: return high
    if yl * yh > 0.0: return math.nan
    a, b = low, high
    fa, fb = yl, yh
    for _ in range(iters):
        m = 0.5 * (a + b)
        fm = error(y, x, k, τ, m, i, r)
        if not math.isfinite(fm): return math.nan
        if abs(fm) <= tol or (b - a) <= 1e-10: return m
        if fa * fm <= 0.0: b, fb = m, fm
        else: a, fa = m, fm
    return 0.5 * (a + b)

@njit(cache=True)
def fitting(y, x, k, τ, i, r, /, low, high, tol, iters):
    assert i == 1 or i == -1
    σl, σh = low, high
    for _ in range(iters):
        σml = σl + (σh - σl) / 3.0
        σmh = σh - (σh - σl) / 3.0
        yml = abs(error(y, x, k, τ, σml, i, r))
        ymh = abs(error(y, x, k, τ, σmh, i, r))
        if yml < ymh: σh = σmh
        else: σl = σml
        if (σh - σl) < tol: break
    return 0.5 * (σl + σh)

@njit(cache=True)
def implied(y, x, k, τ, i, r, /, low, high, tol, iters):
    if not valid(x, k, τ, i) or low <= 0.0 or high <= low: return math.nan
    yl, yh = boundary(x, k, τ, i, r)
    if y < yl - tol or y > yh + tol: return math.nan
    σh = min(high, adaptive(x, k, τ))
    if σh <= low: σh = high
    σ = newton(y, x, k, τ, i, r, low=low, high=σh, tol=tol, iters=iters)
    if math.isfinite(σ): return σ
    return bisection(y, x, k, τ, i, r, low=low, high=σh, tol=tol, iters=max(100, iters))


@njit(cache=True)
def calculation(y, x, k, τ, i, r, /, low, high, tol, iters):
    σ = np.empty(len(y), dtype=np.float64)
    for idx in range(len(y)):
        σ[idx] = implied(y[idx], x[idx], k[idx], τ[idx], i[idx], r[idx], low=low, high=high, tol=tol, iters=iters)
    return σ


class VolatilityCalculator(Logging):
    def __init__(self, *args, low=1e-4, high=5.0, tol=1e-10, iters=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.__hyperparams = dict(low=low, high=high, tol=tol, iters=iters)

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if bool(options.empty): return options
        y = options["median"].to_numpy(np.float64)
        x = options["underlying"].to_numpy(np.float64)
        k = options["strike"].to_numpy(np.float64)
        τ = options["tau"].to_numpy(np.float64)
        i = options["option"].apply(int).to_numpy(np.int8)
        r = options["interest"].to_numpy(np.float64)
        options["implied"] = calculation(y, x, k, τ, i, r, **self.hyperparams)
        self.alert(options)
        return options

    def alert(self, dataframe):
        instrument = str(Concepts.Securities.Instrument.OPTION).title()
        tickers = "|".join(list(dataframe["ticker"].unique()))
        expires = DateRange.create(list(dataframe["expire"].unique()))
        expires = f"{expires.minimum.strftime('%Y%m%d')}->{expires.maximum.strftime('%Y%m%d')}"
        self.console("Calculated", f"{str(instrument)}[{str(tickers)}, {str(expires)}, {len(dataframe):.0f}]")

    @property
    def hyperparams(self): return self.__hyperparams



