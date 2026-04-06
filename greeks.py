# -*- coding: utf-8 -*-
"""
Created on Tues Mar 24 2026
@name:   Greek Objects
@author: Jack Kirby Cook

"""

import math
import numpy as np
import pandas as pd
from numba import njit
from types import SimpleNamespace
from collections import OrderedDict as ODict

from support.concepts import DateRange
from support.finance import Concepts
from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["GreekCalculator"]
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
def delta(x, k, τ, σ, i, r):
    """dy/dx"""
    if not valid(x, k, τ, i) or σ <= 0.0: return math.nan
    zx = zitm(x, k, τ, σ, r)
    return i * normcdf(i * zx)

@njit(cache=True, inline="always")
def gamma(x, k, τ, σ, i, r):
    """d²y/dx²"""
    if x <= 0.0 or k <= 0.0 or τ <= 0.0 or σ <= 0.0: return math.nan
    zx = zitm(x, k, τ, σ, r)
    return normpdf(zx) / (x * σ * math.sqrt(τ))

@njit(cache=True, inline="always")
def theta(x, k, τ, σ, i, r):
    """dy/dτ"""
    if not valid(x, k, τ, i) or σ <= 0.0: return math.nan
    zx = zitm(x, k, τ, σ, r)
    zk = zx - σ * math.sqrt(τ)
    dcf = discount(r, τ)
    fx = -x * normpdf(zx) * σ / (2.0 * math.sqrt(τ))
    fk = -i * r * k * dcf * normcdf(i * zk)
    return fx + fk

@njit(cache=True, inline="always")
def rho(x, k, τ, σ, i, r):
    """dy/dr"""
    if not valid(x, k, τ, i) or σ <= 0.0: return math.nan
    zk = zotm(x, k, τ, σ, r)
    dcf = discount(r, τ)
    return i * k * τ * dcf * normcdf(i * zk)

@njit(cache=True, inline="always")
def vega(x, k, τ, σ, i, r):
    """dy/dσ"""
    if x <= 0.0 or k <= 0.0 or τ <= 0.0 or σ <= 0.0: return 0.0
    zx = zitm(x, k, τ, σ, r)
    return x * normpdf(zx) * math.sqrt(τ)

@njit(cache=True, inline="always")
def vomma(x, k, τ, σ, i, r):
    """d²y/dσ²"""
    if x <= 0.0 or k <= 0.0 or τ <= 0.0 or σ <= 0.0: return math.nan
    zx = zitm(x, k, τ, σ, r)
    zk = zx - σ * math.sqrt(τ)
    dydσ = vega(x, k, τ, σ, i, r)
    return dydσ * zx * zk / max(σ, 1e-12)

@njit(cache=True, inline="always")
def vanna(x, k, τ, σ, i, r):
    """d²y/dx*dσ"""
    if x <= 0.0 or k <= 0.0 or τ <= 0.0 or σ <= 0.0: return math.nan
    zx = zitm(x, k, τ, σ, r)
    zk = zx - σ * math.sqrt(τ)
    return -normpdf(zx) * zk / max(σ, 1e-12)

@njit(cache=True, inline="always")
def charm(x, k, τ, σ, i, r):
    """d²y/dx*dτ"""
    if not valid(x, k, τ, i) or σ <= 0.0: return math.nan
    zx = zitm(x, k, τ, σ, r)
    zk = zx - σ * math.sqrt(τ)
    fk = 2.0 * r * τ - zk * σ * math.sqrt(τ)
    fτσ = 2.0 * τ * σ * math.sqrt(τ)
    return -i * normpdf(zx) * fk / fτσ


@njit(cache=True)
def calculation(x, k, τ, σ, i, r):
    Δ = np.empty(len(x), dtype=np.float64)  # Delta, Δ = dy/dx
    Γ = np.empty(len(x), dtype=np.float64)  # Gamma, Γ = d²y/dx²
    Θ = np.empty(len(x), dtype=np.float64)  # Theta, Θ = dy/dτ
    Ρ = np.empty(len(x), dtype=np.float64)  # Rho, Ρ = dy/dr
    V = np.empty(len(x), dtype=np.float64)  # Vega, V = dy/dσ
    Φ = np.empty(len(x), dtype=np.float64)  # Vomma, Φ = d²y/dσ²
    Ψ = np.empty(len(x), dtype=np.float64)  # Vanna, Ψ = d²y/dx*dσ
    Χ = np.empty(len(x), dtype=np.float64)  # Charm, Χ = d²y/dx*dτ

    for idx in range(len(x)):
        Δ[idx] = delta(x[idx], k[idx], τ[idx], σ[idx], i[idx], r)
        Γ[idx] = gamma(x[idx], k[idx], τ[idx], σ[idx], i[idx], r)
        Θ[idx] = theta(x[idx], k[idx], τ[idx], σ[idx], i[idx], r)
        Ρ[idx] = rho(x[idx], k[idx], τ[idx], σ[idx], i[idx], r)
        V[idx] = vega(x[idx], k[idx], τ[idx], σ[idx], i[idx], r)
        Φ[idx] = vomma(x[idx], k[idx], τ[idx], σ[idx], i[idx], r)
        Ψ[idx] = vanna(x[idx], k[idx], τ[idx], σ[idx], i[idx], r)
        Χ[idx] = charm(x[idx], k[idx], τ[idx], σ[idx], i[idx], r)
    return Δ, Γ, Θ, Ρ, V, Φ, Ψ, Χ


class GreekCalculator(Logging):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        inlet = ODict(x="underlying", k="strike", τ="tau", σ="volatility", i="option", r="interest")
        outlet = ODict(Δ="delta", Γ="gamma", Θ="theta", Ρ="rho", V="vega", Φ="vomma", Ψ="vanna", Χ="charm")
        variables = SimpleNamespace(inlet=inlet, outlet=outlet)
        self.__variables = variables

    def __call__(self, options, *args, interest, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if bool(options.empty): return options
        x = options["underlying"].to_numpy(np.float64)
        k = options["strike"].to_numpy(np.float64)
        τ = options["tau"].to_numpy(np.float64)
        σ = options["implied"].to_numpy(np.float64)
        i = options["option"].apply(int).to_numpy(np.int8)
        greeks = list(calculation(x, k, τ, σ, i, float(interest)))
        greeks = dict(zip(self.variables.outlet.values(), greeks))
        options = pd.concat([options,  pd.DataFrame(greeks)], axis=1)
        self.alert(options)
        return options

    def alert(self, dataframe):
        instrument = str(Concepts.Securities.Instrument.OPTION).title()
        tickers = "|".join(list(dataframe["ticker"].unique()))
        expires = DateRange.create(list(dataframe["expire"].unique()))
        expires = f"{expires.minimum.strftime('%Y%m%d')}->{expires.maximum.strftime('%Y%m%d')}"
        self.console("Calculated", f"{str(instrument)}[{str(tickers)}, {str(expires)}, {len(dataframe):.0f}]")

    @property
    def variables(self): return self.__variables



