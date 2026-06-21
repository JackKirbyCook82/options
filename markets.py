# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 2026
@name:   Option Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from datetime import date as Date

from finance.variables import Alerting, Enumerations
from support.equations import Equations

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SanityFilter", "ViabilityCalculator", "MarketCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


def moneyness_function(series, /, **parameters):
    try: abs(series) <= float(parameters["moneyness"])
    except (KeyError, TypeError): return pd.Series(True, index=series.index)

def tightness_function(series, /, **parameters):
    try: return series <= float(parameters["tightness"])
    except (KeyError, TypeError): return pd.Series(True, index=series.index)


class SanityFilter(Alerting, Equations, variables=["sanity"]):
    sanity = lambda supplied, demanded, bided, asked, realistic: np.logical_and.reduce([supplied, demanded, bided, asked, realistic])
    supplied = lambda supply: supply.notna() & (supply >= 0)
    demanded = lambda demand: demand.notna() & (demand >= 0)
    bided = lambda bid: bid.notna() & np.isfinite(bid) & (bid >= 0)
    asked = lambda ask: ask.notna() & np.isfinite(ask) & (ask >= 0)
    realistic = lambda bid, ask: ask > bid

    def __call__(self, markets, *args, **kwargs):
        assert isinstance(markets, pd.DataFrame)
        if bool(markets.empty): return markets
        previous = len(markets.index)
        markets = self.filter(markets, *args, **kwargs)
        post = len(markets.index)
        sizes = dict(previous=previous, post=post)
        self.alert(markets, title="Filtered", instrument=Enumerations.Instrument.OPTION, **sizes)
        return markets

    def filter(self, dataframe, *args, **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        if bool(dataframe.empty): return dataframe
        mask = self.execute(dataframe, *args, **kwargs)
        mask = mask.squeeze()
        dataframe = dataframe.where(mask)
        dataframe = dataframe.dropna(how="all", inplace=False)
        dataframe = dataframe.reset_index(drop=True, inplace=False)
        return dataframe


class ViabilityCalculator(Alerting, Equations, variables=["tightened", "moneyed", "sized"], parameters={"tightness": None, "moneyness": None, "size": 1}):
    viability = lambda moneyed, tightened, sized: np.logical_and.reduce([moneyed, tightened, sized])
    sized = lambda supply, demand, /, **params:  (supply >= int(params["size"])) & (demand >= int(params["size"]))
    tightened = lambda tightness, /, **params: tightness_function(tightness, **params)
    moneyed = lambda moneyness, /, **params: moneyness_function(moneyness, **params)

    def __call__(self, markets, *args, **kwargs):
        assert isinstance(markets, pd.DataFrame)
        viability = self.execute(markets, *args, **kwargs)
        viability = pd.concat([markets, viability], axis=1)
        self.alert(viability, title="Calculated", instrument=Enumerations.Instrument.OPTION)
        return viability


class MarketCalculator(Equations, Alerting):
    moneyness = lambda spot, strike, option: np.log(spot / strike.astype(float)) * option.astype(int)
    tau = lambda expire: (pd.to_datetime(expire) - pd.Timestamp(Date.today())).dt.days / 365
    dte = lambda expire: (pd.to_datetime(expire) - pd.Timestamp(Date.today())).dt.days
    quality = lambda activity, tightness: activity / (tightness ** 2 + 1e-6)
    activity = lambda supply, demand: np.sqrt(1 + demand + supply)
    tightness = lambda gap, median: gap / median
    median = lambda bid, ask: (bid + ask) / 2
    gap = lambda bid, ask: ask - bid

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        markets = self.execute(options, *args, **kwargs)
        markets = pd.concat([options, markets], axis=1)
        self.alert(markets, title="Calculated", instrument=Enumerations.Instrument.OPTION)
        return markets



