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

from support.finance import Concepts, Alerting

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
def valid(x, k, τ, i):
    positive = (x > 0.0 and k > 0.0 and τ > 0.0)
    option = (i == 1 or i == -1)
    finite = (math.isfinite(x) and math.isfinite(k) and math.isfinite(τ))
    return positive and option and finite

@njit(cache=True, inline="always")
def delta(x, k, τ, σ, i, r, q):
    """dy/dx"""
    if not valid(x, k, τ, i) or σ <= 0.0 or not math.isfinite(r) or not math.isfinite(q): return math.nan
    zx = zitm(x, k, τ, σ, r, q)
    return i * discount(q, τ) * normcdf(i * zx)

@njit(cache=True, inline="always")
def gamma(x, k, τ, σ, i, r, q):
    """d²y/dx²"""
    if x <= 0.0 or k <= 0.0 or τ <= 0.0 or σ <= 0.0 or not math.isfinite(r) or not math.isfinite(q): return math.nan
    zx = zitm(x, k, τ, σ, r, q)
    return discount(q, τ) * normpdf(zx) / (x * σ * math.sqrt(τ))

@njit(cache=True, inline="always")
def theta(x, k, τ, σ, i, r, q):
    """dy/dτ"""
    if not valid(x, k, τ, i) or σ <= 0.0 or not math.isfinite(r) or not math.isfinite(q): return math.nan
    zx = zitm(x, k, τ, σ, r, q)
    zk = zx - σ * math.sqrt(τ)
    fx = -x * discount(q, τ) * normpdf(zx) * σ / (2.0 * math.sqrt(τ))
    fq = i * q * x * discount(q, τ) * normcdf(i * zx)
    fk = -i * r * k * discount(r, τ) * normcdf(i * zk)
    return fx + fq + fk

@njit(cache=True, inline="always")
def rho(x, k, τ, σ, i, r, q):
    """dy/dr"""
    if not valid(x, k, τ, i) or σ <= 0.0 or not math.isfinite(r) or not math.isfinite(q): return math.nan
    zk = zotm(x, k, τ, σ, r, q)
    return i * k * τ * discount(r, τ) * normcdf(i * zk)

@njit(cache=True, inline="always")
def vega(x, k, τ, σ, i, r, q):
    """dy/dσ"""
    if x <= 0.0 or k <= 0.0 or τ <= 0.0 or σ <= 0.0 or not math.isfinite(r) or not math.isfinite(q): return 0.0
    zx = zitm(x, k, τ, σ, r, q)
    return x * discount(q, τ) * normpdf(zx) * math.sqrt(τ)

@njit(cache=True, inline="always")
def vomma(x, k, τ, σ, i, r, q):
    """d²y/dσ²"""
    if x <= 0.0 or k <= 0.0 or τ <= 0.0 or σ <= 0.0 or not math.isfinite(r) or not math.isfinite(q): return math.nan
    zx = zitm(x, k, τ, σ, r, q)
    zk = zx - σ * math.sqrt(τ)
    dydσ = vega(x, k, τ, σ, i, r, q)
    return dydσ * zx * zk / max(σ, 1e-12)

@njit(cache=True, inline="always")
def vanna(x, k, τ, σ, i, r, q):
    """d²y/dx*dσ"""
    if x <= 0.0 or k <= 0.0 or τ <= 0.0 or σ <= 0.0 or not math.isfinite(r) or not math.isfinite(q): return math.nan
    zx = zitm(x, k, τ, σ, r, q)
    zk = zx - σ * math.sqrt(τ)
    return - discount(q, τ) * normpdf(zx) * zk / max(σ, 1e-12)

@njit(cache=True, inline="always")
def charm(x, k, τ, σ, i, r, q):
    """d²y/dx*dτ"""
    if not valid(x, k, τ, i) or σ <= 0.0 or not math.isfinite(r) or not math.isfinite(q): return math.nan
    zx = zitm(x, k, τ, σ, r, q)
    zk = zx - σ * math.sqrt(τ)
    fk = 2.0 * (r - q) * τ - zk * σ * math.sqrt(τ)
    fτσ = 2.0 * τ * σ * math.sqrt(τ)
    return discount(q, τ) * normpdf(zx) * fk / fτσ - i * q * discount(q, τ) * normcdf(i * zx)


@njit(cache=True)
def calculation(x, k, τ, σ, i, r, q):
    Δ = np.empty(len(x), dtype=np.float64)  # Delta, Δ = dy/dx
    Γ = np.empty(len(x), dtype=np.float64)  # Gamma, Γ = d²y/dx²
    Θ = np.empty(len(x), dtype=np.float64)  # Theta, Θ = dy/dτ
    Ρ = np.empty(len(x), dtype=np.float64)  # Rho, Ρ = dy/dr
    V = np.empty(len(x), dtype=np.float64)  # Vega, V = dy/dσ
    Φ = np.empty(len(x), dtype=np.float64)  # Vomma, Φ = d²y/dσ²
    Ψ = np.empty(len(x), dtype=np.float64)  # Vanna, Ψ = d²y/dx*dσ
    Χ = np.empty(len(x), dtype=np.float64)  # Charm, Χ = d²y/dx*dτ

    for idx in range(len(x)):
        Δ[idx] = delta(x[idx], k[idx], τ[idx], σ[idx], i[idx], r, q)
        Γ[idx] = gamma(x[idx], k[idx], τ[idx], σ[idx], i[idx], r, q)
        Θ[idx] = theta(x[idx], k[idx], τ[idx], σ[idx], i[idx], r, q)
        Ρ[idx] = rho(x[idx], k[idx], τ[idx], σ[idx], i[idx], r, q)
        V[idx] = vega(x[idx], k[idx], τ[idx], σ[idx], i[idx], r, q)
        Φ[idx] = vomma(x[idx], k[idx], τ[idx], σ[idx], i[idx], r, q)
        Ψ[idx] = vanna(x[idx], k[idx], τ[idx], σ[idx], i[idx], r, q)
        Χ[idx] = charm(x[idx], k[idx], τ[idx], σ[idx], i[idx], r, q)
    return Δ, Γ, Θ, Ρ, V, Φ, Ψ, Χ


class GreekCalculator(Alerting):
    def __call__(self, options, *args, interest, dividends, **kwargs):
        assert isinstance(options, pd.DataFrame)
        spot = options["spot"].to_numpy(np.float64)
        strike = options["strike"].to_numpy(np.float64)
        tau = options["tau"].to_numpy(np.float64)
        implied = options["implied"].to_numpy(np.float64)
        option = options["option"].apply(int).to_numpy(np.int8)
        greeks = list(calculation(spot, strike, tau, implied, option, float(interest), float(dividends)))
        greeks = dict(zip(["delta", "gamma", "theta", "rho", "vega", "vomma", "vanna", "charm"], greeks))
        greeks = pd.DataFrame(greeks)
        greeks = pd.concat([options, greeks], axis=1)
        self.alert(options, title="Calculated", instrument=Concepts.Securities.Instrument.OPTION)
        return greeks



