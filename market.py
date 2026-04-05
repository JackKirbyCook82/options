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
        mask = self.calculate(options, *args, **kwargs)
        previous = len(options.index)
        options = options.where(mask.squeeze())
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
    sanity = lambda realistic, supplied, demanded, bided, asked: np.logical_and.reduce([realistic, supplied, demanded, bided, asked])
    realistic = lambda bid, ask: ask > bid
    supplied = lambda supply: supply.notna() & (supply >= 1)
    demanded = lambda demand: demand.notna() & (demand >= 1)
    bided = lambda bid: bid.notna() & (bid >= 0)
    asked = lambda ask: ask.notna() & (ask >= 0)


class ViabilityFilter(MarketFilter, variables=["viability"], defaults={"spread": 0.25, "size": 2}):
    viability = lambda liquid, supplied, demanded:  np.logical_and.reduce([liquid, supplied, demanded])
    liquid = lambda bid, ask, *, spread=0.25: (ask - bid) * 2 / (ask + bid) <= float(spread)
    supplied = lambda supply, *, size=2: supply >= int(size)
    demanded = lambda demand, *, size=2: demand >= int(size)


class MarketCalculator(Calculation, Logging):
    tau = lambda expire: (pd.to_datetime(expire) - pd.Timestamp(Date.today())).dt.days / 365
    intrinsic = lambda strike, underlying, option: (np.maximum((underlying - strike) * option.astype(int), 0) * option.astype(int))
    moneyness = lambda strike, underlying: strike / underlying
    mean = lambda bid, ask, demand, supply: (bid * demand + ask * supply) / (demand + supply)
    median = lambda bid, ask: (bid + ask) / 2
    spread = lambda bid, ask: ask - bid

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if bool(options.empty): return options
        calculated = self.calculate(options, *args, **kwargs)
        options = pd.concat([options, calculated], axis=1)
        self.alert(options)
        return options

    def alert(self, dataframe):
        instrument = str(Concepts.Securities.Instrument.OPTION).title()
        tickers = "|".join(list(dataframe["ticker"].unique()))
        expires = DateRange.create(list(dataframe["expire"].unique()))
        expires = f"{expires.minimum.strftime('%Y%m%d')}->{expires.maximum.strftime('%Y%m%d')}"
        self.console("Calculated", f"{str(instrument)}[{str(tickers)}, {str(expires)}, {len(dataframe):.0f}]")





