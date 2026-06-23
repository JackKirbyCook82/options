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


class SanityFilter(Alerting, Equations, variables=["sanity"]):
    sanity = lambda supplied, demanded, bided, asked, realistic: np.logical_and.reduce([supplied, demanded, bided, asked, realistic])
    supplied = lambda supply: supply.notna() & (supply >= 0)
    demanded = lambda demand: demand.notna() & (demand >= 0)
    bided = lambda bid: bid.notna() & np.isfinite(bid) & (bid >= 0)
    asked = lambda ask: ask.notna() & np.isfinite(ask) & (ask >= 0)
    realistic = lambda bid, ask: ask > bid

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if bool(options.empty): return options
        previous = len(options.index)
        mask = self.execute(options, *args, **kwargs).squeeze()
        options = options.where(mask).dropna(how="all", inplace=False)
        post = len(options.index)
        sizes = dict(previous=previous, post=post)
        self.alert(options, title="Filtered", instrument=Enumerations.Instrument.OPTION, **sizes)
        return options


class ViabilityCalculator(Alerting, Equations, parameters={"tight": None, "money": None, "size": 1}):
    viability = lambda moneyed, tightened, sized: np.logical_and.reduce([moneyed, tightened, sized])
    sized = lambda supply, demand, *, size:  (supply >= int(size)) & (demand >= int(size))
    tightened = lambda tightness, *, tight: tightness <= float(tight) if tight is not None else pd.Series(True, index=tightness.index)
    moneyed = lambda moneyness, *, money: abs(moneyness) <= float(money) if money is not None else pd.Series(True, index=moneyness.index)

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if bool(options.empty): return options
        masking = self.execute(options, *args, **kwargs)
        mask = masking["viability"]
        previous = len(options.index)
        options = options.where(mask).dropna(how="all", inplace=False)
        post = len(options.index)
        sizes = dict(previous=previous, post=post)
        self.alert(options, title="Filtered", instrument=Enumerations.Instrument.OPTION, **sizes)
        self.analysis(masking, title="Knockout", **sizes)
        return options

    def analysis(self, viability, *args, title, previous, post, **kwargs):
        tight = kwargs.get("tight", self.constants.get("tight", None))
        money = kwargs.get("money", self.constants.get("money", None))
        size = kwargs.get("size", self.constants.get("size", None))
        criteria = [f"{1 - int(post) / int(previous):.0f}% Total"]
        if size is not None: criteria.append(f"{(~viability['sized']).sum() / len(viability.index):.0f}% @ Size={int(size):.0f}")
        if money is not None: criteria.append(f"{(~viability['moneyed']).sum() / len(viability.index):.0f}% @ Money={float(money):.02f}")
        if tight is not None: criteria.append(f"{(~viability['tightened']).sum() / len(viability.index):.0f}% @ Tight={float(tight):.02f}")
        self.console(str(title), f"Options[{', '.join(reversed(criteria))}]")


class MarketCalculator(Alerting, Equations):
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



