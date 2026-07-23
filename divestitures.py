# -*- coding: utf-8 -*-
"""
Created on Mon Jul 6 2026
@name:   Option Divestiture Objects
@author: Jack Kirby Cook

"""

import pandas as pd
from abc import ABC

from options.prospects import Prospect
from finance.enumerations import Spread, Instrument
from finance.logging import Logging
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DivestitureCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class Divestiture(Prospect):
    @property
    def spent(self): return self.securities["spent"].sum()
    @property
    def liquidate(self):
        selling = (self.securities["bid"] * ((self.positions.map(int) + 1) / 2) * self.quantities).sum()
        buying = (self.securities["ask"] * ((self.positions.map(int) - 1) / 2) * self.quantities).sum()
        return selling - buying

    @property
    def gain(self): return max(self.market - self.spent, 0)
    @property
    def loss(self): return max(self.spent - self.market, 0)
    @property
    def profit(self): return self.liquidate - self.spent


class DivestitureCreator(ABC, metaclass=RegistryMeta):
    pass

class FlyAcquisitionCreator(DivestitureCreator, register=Spread.FLY):
    pass

class CalendarAcquisitionCreator(DivestitureCreator, register=Spread.CALENDAR):
    pass


class DivestitureCalculator(Logging):
    def __init__(self, *args, spreads, metrics, **kwargs):
        super().__init__(*args, **kwargs)
        creators = {spread: DivestitureCreator[spread](*args, **kwargs) for spread in spreads}
        self.__creators = creators
        self.__metrics = metrics

    def __call__(self, holdings, /, **kwargs):
        assert isinstance(holdings, pd.DataFrame)
        prospects = self.calculator(holdings, **kwargs)
        prospects = list(prospects)
        self.results(prospects, title="Calculator", instrument=Instrument.SPREAD)
        return prospects

    def calculator(self, holdings, **kwargs):
        assert isinstance(holdings, pd.DataFrame)
        for spread, creator in self.creators.items():
            for prospect in creator(holdings, **kwargs):
                if not self.metrics(prospect): continue
                yield prospect

    @property
    def creators(self): return self.__creators
    @property
    def metrics(self): return self.__metrics



