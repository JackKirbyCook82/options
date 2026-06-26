# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 2026
@name:   Option Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from itertools import product
from datetime import date as Date
from dataclasses import dataclass

from finance.variables import Enumerations
from finance.logging import Logging
from support.equations import Equations
from support.custom import NumRange

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SanityFilter", "SurvivalCalculator", "ViabilityCalculator", "MarketCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True)
class Viability:
    tight: float | NumRange; money: float | NumRange; size: int | NumRange

    def __iter__(self): yield self.tight; yield self.money; yield self.size
    def __call__(self, gridsize):
        function = lambda variable: np.linspace(variable.minimum, variable.maximum, gridsize) if isinstance(variable, NumRange) else np.array([variable])
        variables = [function(variable) for variable in iter(self)]
        yield from product(*variables)


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


class SurvivalCalculator(Logging):
    def __init__(self, *args, tight, money, size, gridsize=25, **kwargs):
        assert isinstance(tight, (float, NumRange)) and isinstance(money, (float, NumRange)) and isinstance(size, (int, NumRange))
        assert isinstance(gridsize, int)
        super().__init__(*args, **kwargs)
        self.__viability = Viability(tight, money, size)
        self.__gridsize = gridsize

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        survivals = self.generate(options, *args, **kwargs)
        self.results(options, title="Analyzed", instrument=Enumerations.Instrument.OPTION)
        return survivals

    def generate(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        generator = self.generator(options, *args, **kwargs)
        survivals = list(generator)
        survivals = pd.DataFrame(survivals)
        return survivals

    def generator(self, options, *args, **kwargs):
        moneyness = np.abs(pd.to_numeric(options["moneyness"], errors="coerce"))
        tightness = pd.to_numeric(options["tightness"], errors="coerce")
        supply = pd.to_numeric(options["supply"], errors="coerce")
        demand = pd.to_numeric(options["demand"], errors="coerce")
        gridsize = int(self.gridsize)
        for tight, money, size in self.viability(gridsize):
            sized = (supply >= size) & (demand >= size)
            tightened = tightness <= tight
            moneyed = moneyness <= money
            viable = sized & tightened & moneyed
            survival = int(viable.sum())
            yield dict(tightness=tight, moneyness=money, sizing=size, survival=survival)

    @property
    def viability(self): return self.__viability
    @property
    def gridsize(self): return self.__gridsize


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



