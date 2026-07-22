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

from finance.enumerations import Instrument
from finance.logging import Logging
from support.equations import Equations
from support.custom import NumberRange

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["OptionCalculator", "SurvivalCalculator", "SanityFilter", "ViabilityFilter"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class OptionCalculator(Logging, Equations, variables=["moneyness", "tightness", "activity", "median", "gap", "dte"]):
    dte = lambda expire: (pd.to_datetime(expire) - pd.Timestamp(Date.today())).dt.days
    moneyness = lambda spot, strike, option: np.log(spot / strike.astype(float)) * option.astype(int)
    activity = lambda supply, demand: np.minimum(supply, demand) / (np.maximum(supply, demand) + 10)
    tightness = lambda gap, median: gap / median
    mean = lambda bid, ask, supply, demand: ((bid * demand) + (ask * supply)) / (demand + supply)
    median = lambda bid, ask: (bid + ask) / 2
    gap = lambda bid, ask: ask - bid

    def __call__(self, options, **kwargs):
        assert isinstance(options, pd.DataFrame)
        calculated = self.execute(options, **kwargs)
        options = pd.concat([options, calculated], axis=1)
        self.results(options, title="Calculated", instrument=Instrument.OPTION)
        return options


class SurvivalCalculator(Logging):
    def __init__(self, *args, tight, money, active, gridsize=25, **kwargs):
        assert isinstance(tight, (float, NumberRange)) and isinstance(money, (float, NumberRange)) and isinstance(active, (float, NumberRange))
        assert isinstance(gridsize, int)
        super().__init__(*args, **kwargs)
        self.__viability = Viability(tight, money, active)
        self.__gridsize = gridsize

    def __call__(self, options, **kwargs):
        assert isinstance(options, pd.DataFrame)
        survivals = self.generate(options, **kwargs)
        self.results(options, title="Calculated", instrument=Instrument.OPTION)
        return survivals

    def generate(self, options, **kwargs):
        assert isinstance(options, pd.DataFrame)
        generator = self.generator(options, **kwargs)
        survivals = list(generator)
        survivals = pd.DataFrame(survivals)
        return survivals

    def generator(self, options, **kwargs):
        moneyness = np.abs(pd.to_numeric(options["moneyness"], errors="coerce"))
        tightness = pd.to_numeric(options["tightness"], errors="coerce")
        activity = pd.to_numeric(options["activity"], errors="coerce")
        gridsize = int(self.gridsize)
        for tight, money, active in self.viability(gridsize):
            activated = activity >= active
            tightened = tightness <= tight
            moneyed = moneyness <= money
            viable = activated & tightened & moneyed
            survival = int(viable.sum())
            yield dict(tightness=tight, moneyness=money, activity=activity, survival=survival)

    @property
    def viability(self): return self.__viability
    @property
    def gridsize(self): return self.__gridsize


class SanityFilter(Logging, Equations, parameters={"size": 1}):
    sanity = lambda supplied, demanded, bided, asked, realistic: np.logical_and.reduce([supplied, demanded, bided, asked, realistic])
    supplied = lambda supply, *, size: supply.notna() & (supply >= size)
    demanded = lambda demand, *, size: demand.notna() & (demand >= size)
    bided = lambda bid: bid.notna() & np.isfinite(bid) & (bid >= 0)
    asked = lambda ask: ask.notna() & np.isfinite(ask) & (ask >= 0)
    realistic = lambda bid, ask: ask > bid

    def __call__(self, options, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if bool(options.empty): return options
        sanity = self.execute(options, **kwargs).squeeze()
        self.results(options, title="Calculated", instrument=Instrument.OPTION)
        previous = len(options.index)
        options = options.where(sanity["sanity"]).dropna(how="all", inplace=False)
        post = len(options.index)
        sizes = dict(previous=previous, post=post)
        self.results(options, title="Filtered", instrument=Instrument.OPTION, **sizes)
        return options


@dataclass(frozen=True)
class Viability:
    tight: float | NumberRange; money: float | NumberRange; active: int | NumberRange

    def __iter__(self): yield self.tight; yield self.money; yield self.active
    def __call__(self, gridsize):
        function = lambda variable: np.linspace(variable.minimum, variable.maximum, gridsize) if isinstance(variable, NumberRange) else np.array([variable])
        variables = [function(variable) for variable in iter(self)]
        yield from product(*variables)


class ViabilityFilter(Logging, Equations, parameters={"tight": None, "money": None, "active": None}):
    viability = lambda moneyed, tightened, activated: np.logical_and.reduce([moneyed, tightened, activated])
    tightened = lambda tightness, *, tight: tightness <= float(tight) if tight is not None else pd.Series(True, index=tightness.index)
    moneyed = lambda moneyness, *, money: abs(moneyness) <= float(money) if money is not None else pd.Series(True, index=moneyness.index)
    activated = lambda activity, *, active: activity >= float(active) if active is not None else pd.Series(True, index=activity.index)

    def __call__(self, options, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if bool(options.empty): return options
        viability = self.execute(options, **kwargs)
        self.results(options, title="Calculated", instrument=Instrument.OPTION)
        previous = len(options.index)
        options = options.where(viability["viability"]).dropna(how="all", inplace=False)
        post = len(options.index)
        sizes = dict(previous=previous, post=post)
        self.results(options, title="Filtered", instrument=Instrument.OPTION, **sizes)
        self.breakdown(viability)
        return options

    def breakdown(self, viabilities):
        tightness = (viabilities['tightened']).sum() / len(viabilities.index) * 100
        moneyness = (viabilities['moneyed']).sum() / len(viabilities.index) * 100
        activity = (viabilities['activated']).sum() / len(viabilities.index) * 100
        strings = list()
        try: strings.append(f"Tight<={self.constants['tight']:.2f}: {tightness:.0f}%")
        except KeyError: pass
        try: strings.append(f"Money<={self.constants['money']:.2f}: {moneyness:.0f}%")
        except KeyError: pass
        try: strings.append(f"Active>={self.constants['active']:.0f}: {activity:.0f}%")
        except KeyError: pass
        self.console("Filtered", f"Options[{', '.join(strings)}]")




