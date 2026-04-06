# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 2026
@name:   Market Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from abc import ABC
from datetime import date as Date

from support.calculations import Calculation
from support.concepts import DateRange
from support.finance import Concepts
from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SanityFilter", "ViabilityFilter", "MarketCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class MarketFilter(Calculation, Logging, ABC):
    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if bool(options.empty): return options
        mask = self.calculate(options, *args, **kwargs).squeeze()
        previous = len(options.index)
        options = options.where(mask)
        options = options.dropna(how="all", inplace=False)
        options = options.reset_index(drop=True, inplace=False)
        self.alert(options, int(previous), len(options))
        return options

    def alert(self, dataframe, previous, post):
        instrument = str(Concepts.Securities.Instrument.OPTION).title()
        tickers = "|".join(list(dataframe["ticker"].unique()))
        expires = DateRange.create(list(dataframe["expire"].unique()))
        expires = f"{expires.minimum.strftime('%Y%m%d')}->{expires.maximum.strftime('%Y%m%d')}"
        self.console("Filtered", f"{str(instrument)}[{str(tickers)}, {str(expires)}, {int(previous):.0f}|{int(post):.0f}]")


class SanityFilter(MarketFilter, variables=["sanity"]):
    sanity = lambda supplied, demanded, bided, asked, realistic: np.logical_and.reduce([supplied, demanded, bided, asked, realistic])
    supplied = lambda supply, *, size: supply.notna() & (supply >= 1)
    demanded = lambda demand, *, size: demand.notna() & (demand >= 1)
    bided = lambda bid: bid.notna() & np.isfinite(bid) & (bid >= 0)
    asked = lambda ask: ask.notna() & np.isfinite(ask) & (ask >= 0)
    realistic = lambda bid, ask: ask > bid


class ViabilityFilter(MarketFilter, variables=["viability"], defaults={"size": 2, "money": 0.10, "tight": 0.25}):
    viability = lambda money, tight, supplied, demanded:  np.logical_and.reduce([money, tight, supplied, demanded])
    money = lambda moneyness, /, money: moneyness >= float(money)
    tight = lambda tightness, /, tight: tightness <= float(tight)
    supplied = lambda supply, *, size: supply >= int(size)
    demanded = lambda demand, *, size: demand >= int(size)


class MarketCalculator(Calculation, Logging):
    tau = lambda expire: (pd.to_datetime(expire) - pd.Timestamp(Date.today())).dt.days / 365
    moneyness = lambda strike, underlying, option: np.log10(underlying / strike) * option.astype(int)
    tightness = lambda bid, ask: (ask - bid) * 2 / (ask + bid)
    mean = lambda bid, ask, demand, supply: (bid * demand + ask * supply) / (demand + supply)
    median = lambda bid, ask: (bid + ask) / 2
    spread = lambda bid, ask: ask - bid

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if bool(options.empty): return options
        market = self.calculate(options, *args, **kwargs)
        options = pd.concat([options, market], axis=1)
        self.alert(options)
        return options

    def alert(self, dataframe):
        instrument = str(Concepts.Securities.Instrument.OPTION).title()
        tickers = "|".join(list(dataframe["ticker"].unique()))
        expires = DateRange.create(list(dataframe["expire"].unique()))
        expires = f"{expires.minimum.strftime('%Y%m%d')}->{expires.maximum.strftime('%Y%m%d')}"
        self.console("Calculated", f"{str(instrument)}[{str(tickers)}, {str(expires)}, {len(dataframe):.0f}]")





