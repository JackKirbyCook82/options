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

from support.equations import Equations
from support.concepts import DateRange
from support.finance import Concepts
from support.filters import Filter
from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SanityFilter", "ViabilityFilter", "OptionCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class OptionFilter(Filter, Logging, ABC):
    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if bool(options.empty): return options
        previous = len(options.index)
        options = self.filter(options, *args, **kwargs)
        post = len(options.index)
        self.alert(options, int(previous), int(post))
        return options

    def alert(self, dataframe, previous, post):
        instrument = str(Concepts.Securities.Instrument.OPTION).title()
        tickers = "|".join(list(dataframe["ticker"].unique()))
        expires = DateRange.create(list(dataframe["expire"].unique()))
        expires = f"{expires.minimum.strftime('%Y%m%d')}->{expires.maximum.strftime('%Y%m%d')}"
        self.console("Filtered", f"{str(instrument)}[{str(tickers)}, {str(expires)}, {int(previous):.0f}|{int(post):.0f}]")


class SanityFilter(OptionFilter, variables=["sanity"]):
    sanity = lambda supplied, demanded, bided, asked, realistic: np.logical_and.reduce([supplied, demanded, bided, asked, realistic])
    supplied = lambda supply: supply.notna() & (supply >= 1)
    demanded = lambda demand: demand.notna() & (demand >= 1)
    bided = lambda bid: bid.notna() & np.isfinite(bid) & (bid >= 0)
    asked = lambda ask: ask.notna() & np.isfinite(ask) & (ask >= 0)
    realistic = lambda bid, ask: ask > bid


class ViabilityFilter(OptionFilter, variables=["viability"], defaults={"size": 2, "money": 0.20, "tight": 0.20}):
    viability = lambda moneyed, tightened, supplied, demanded:  np.logical_and.reduce([moneyed, tightened, supplied, demanded])
    moneyed = lambda moneyness, *, money: abs(moneyness) <= float(money) if money is not None else pd.Series(True, index=moneyness.index)
    tightened = lambda tightness, *, tight: tightness <= float(tight) if tight is not None else pd.Series(True, index=tightness.index)
    supplied = lambda supply, *, size: supply >= int(size)
    demanded = lambda demand, *, size: demand >= int(size)


class OptionCalculator(Equations, Logging):
    tau = lambda expire: (pd.to_datetime(expire) - pd.Timestamp(Date.today())).dt.days / 365
    moneyness = lambda spot, strike, option: np.log10(spot / strike) * option.astype(int)
    tightness = lambda bid, ask, median: (ask - bid) / median
    mean = lambda bid, ask, demand, supply: (bid * demand + ask * supply) / (demand + supply)
    median = lambda bid, ask: (bid + ask) / 2
    spread = lambda bid, ask: ask - bid

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if bool(options.empty): return options
        calculated = self.equate(options, *args, **kwargs)
        options = pd.concat([options, calculated], axis=1)
        self.alert(options)
        return options

    def alert(self, dataframe):
        instrument = str(Concepts.Securities.Instrument.OPTION).title()
        tickers = "|".join(list(dataframe["ticker"].unique()))
        expires = DateRange.create(list(dataframe["expire"].unique()))
        expires = f"{expires.minimum.strftime('%Y%m%d')}->{expires.maximum.strftime('%Y%m%d')}"
        self.console("Calculated", f"{str(instrument)}[{str(tickers)}, {str(expires)}, {len(dataframe):.0f}]")






