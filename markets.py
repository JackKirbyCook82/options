# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 2026
@name:   Option Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from abc import ABC
from datetime import date as Date

from support.finance import Concepts, Alerting
from support.equations import Equations
from support.filters import Filter

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SanityFilter", "ViabilityFilter", "MarketCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class MarketFilter(Filter, Alerting, ABC):
    def __call__(self, markets, *args, **kwargs):
        assert isinstance(markets, pd.DataFrame)
        if bool(markets.empty): return markets
        previous = len(markets.index)
        markets = self.filter(markets, *args, **kwargs)
        post = len(markets.index)
        sizes = dict(previous=previous, post=post)
        self.alert(markets, instrument=Concepts.Securities.Instrument.OPTION, **sizes)
        return markets


class SanityFilter(MarketFilter, variables=["sanity"]):
    sanity = lambda supplied, demanded, bided, asked, realistic: np.logical_and.reduce([supplied, demanded, bided, asked, realistic])
    supplied = lambda supply: supply.notna() & (supply >= 1)
    demanded = lambda demand: demand.notna() & (demand >= 1)
    bided = lambda bid: bid.notna() & np.isfinite(bid) & (bid >= 0)
    asked = lambda ask: ask.notna() & np.isfinite(ask) & (ask >= 0)
    realistic = lambda bid, ask: ask > bid


class ViabilityFilter(MarketFilter, variables=["viability"], defaults={"size": 2, "money": 0.20, "tight": 0.20}):
    viability = lambda moneyed, tightened, supplied, demanded:  np.logical_and.reduce([moneyed, tightened, supplied, demanded])
    moneyed = lambda moneyness, *, money: abs(moneyness) <= float(money) if money is not None else pd.Series(True, index=moneyness.index)
    tightened = lambda tightness, *, tight: tightness <= float(tight) if tight is not None else pd.Series(True, index=tightness.index)
    supplied = lambda supply, *, size: supply >= int(size)
    demanded = lambda demand, *, size: demand >= int(size)


class MarketCalculator(Equations, Alerting):
    tau = lambda expire: (pd.to_datetime(expire) - pd.Timestamp(Date.today())).dt.days / 365
    moneyness = lambda spot, strike, option: np.log10(spot / strike) * option.astype(int)
    tightness = lambda bid, ask, median: (ask - bid) / median
    mean = lambda bid, ask, demand, supply: (bid * demand + ask * supply) / (demand + supply)
    median = lambda bid, ask: (bid + ask) / 2
    spread = lambda bid, ask: ask - bid

    def __call__(self, markets, *args, **kwargs):
        assert isinstance(markets, pd.DataFrame)
        if bool(markets.empty): return markets
        options = self.equate(markets, *args, **kwargs)
        self.alert(markets, instrument=Concepts.Securities.Instrument.OPTION)
        return options




