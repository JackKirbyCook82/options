# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 2026
@name:   Option Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from datetime import date as Date

from finance.variables import Enumerations
from finance.logging import Logging
from support.equations import Equations
from support.custom import NumRange

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SanityFilter", "ViabilityCalculator", "MarketCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class SanityFilter(Logging, Equations, variables=["sanity"]):
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
        sanity = self.execute(options, *args, **kwargs).squeeze()
        options = options.where(sanity).dropna(how="all", inplace=False)
        post = len(options.index)
        sizes = dict(previous=previous, post=post)
        self.results(options, title="Filtered", instrument=Enumerations.Instrument.OPTION, **sizes)
        return options


class ViabilityCalculator(Logging, Equations, parameters={"tight": None, "money": None, "size": 1}):
    viability = lambda moneyed, tightened, sized: np.logical_and.reduce([moneyed, tightened, sized])
    sized = lambda supply, demand, *, size:  (supply >= int(size)) & (demand >= int(size))
    tightened = lambda tightness, *, tight: tightness <= float(tight) if tight is not None else pd.Series(True, index=tightness.index)
    moneyed = lambda moneyness, *, money: abs(moneyness) <= float(money) if money is not None else pd.Series(True, index=moneyness.index)

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if bool(options.empty): return options
        viabilities = self.execute(options, *args, **kwargs)
        previous = len(options.index)
        options = options.where(viabilities["viability"]).dropna(how="all", inplace=False)
        post = len(options.index)
        sizes = dict(previous=previous, post=post)
        self.results(options, viabilities, title="Filtered", instrument=Enumerations.Instrument.OPTION, **sizes)
        return options

    def results(self, options, viabilities, *args, **kwargs):
        tightness = (viabilities['sized']).sum() / len(viabilities.index) * 100
        moneyness = (viabilities['moneyed']).sum() / len(viabilities.index) * 100
        sizing = (viabilities['tightened']).sum() / len(viabilities.index) * 100
        strings = list()
        try: strings.append(f"Tightness<={self.constants['tight']:.2f}: {tightness:.0f}%")
        except KeyError: pass
        try: strings.append(f"Moneyness<={self.constants['money']:.2f}: {moneyness:.0f}%")
        except KeyError: pass
        try: strings.append(f"Sizing>={self.constants['size']:.0f}: {sizing:.0f}%")
        except KeyError: pass
        super().results(options, *args, **kwargs)
        self.console("Filtered", f"Options[{', '.join(strings)}]")


class ViabilityAnalyzer(Logging):
    def __init__(self, *args, tight, money, size, gridsize=25, **kwargs):
        assert isinstance(tight, NumRange) and isinstance(money, NumRange) and isinstance(size, int)
        super().__init__(*args, **kwargs)
        self.__tight = np.linspace(tight.minimum, tight.maximum, gridsize)
        self.__money = np.linspace(money.minimum, money.maximum, gridsize)
        self.__size = int(size)

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if bool(options.empty): return

        # CHATGPT CODE WILL GO HERE

    @property
    def tight(self): return self.__tight
    @property
    def money(self): return self.__money
    @property
    def size(self): return self.__size


class MarketCalculator(Logging, Equations):
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
        self.results(markets, title="Calculated", instrument=Enumerations.Instrument.OPTION)
        return markets



