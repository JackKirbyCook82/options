# -*- coding: utf-8 -*-
"""
Created on Weds Nov 4 2026
@name:   Settlement Objects
@author: Jack Kirby Cook
@file:   options/settlements.py

"""

import numpy as np
from dataclasses import dataclass
from datetime import date as Date
from collections import OrderedDict as ODict

from finance.enumerations import Instrument
from finance.reporting import Results
from support.equations import Equation
from support.custom import NumberRange
from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SettlementCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


@dataclass(frozen=True, slots=True)
class Settlement: expire: Date; strike: NumberRange

@dataclass
class Settlements:
    settlements: list

    def __contains__(self, settlement):
        assert isinstance(settlement, tuple) and len(settlement) == 2
        expire, strike = settlement
        assert isinstance(expire, Date) and isinstance(strike, float)
        settlements = ODict([(settlement.expire, settlement.strike) for settlement in self.settlements])
        return strike in settlements.get(expire, [])


class SettlementCalculator(Results, Logging, Equation, variables=["strike"]):
    forward = lambda spot, expire, today, tau, interest, dividends: spot * np.exp(np.subtract(interest, dividends) * tau(expire))
    strike = lambda forward, zscore, tau, volatility: forward * np.exp(zscore * volatility * np.sqrt(tau))
    tau = lambda expire, today: (expire - today).days / 252

    def __init__(self, *args, today, zscore, **kwargs):
        super().__init__(*args, **kwargs)
        self.__zscore = zscore
        self.__today = today

    def __call__(self, expires, /, **kwargs):
        assert isinstance(expires, list) and all([isinstance(expire, Date) for expire in expires])
        settlements = self.calculate(expires, **kwargs)
        scope = self.scope(settlements, instrument=Instrument.SETTLEMENT)
        results = self.results(scope=scope, size=len(settlements))
        self.console("Calculated", results)
        return settlements

    def calculate(self, expires, /, **kwargs):
        settlements = self.calculator(expires, **kwargs)
        settlements = list(settlements)
        return settlements

    def calculator(self, expires, /, spot, volatility, dividends, interest, **kwargs):
        parameters = dict(spot=spot, volatility=volatility, dividends=dividends, interest=interest)
        for expire in expires:
            minimum = self.evaluate(expire=expire, zscore=-self.zscore, **parameters)["strike"]
            maximum = self.evaluate(expire=expire, zscore=+self.zscore, **parameters)["strike"]
            strikes = NumberRange([minimum, maximum])
            yield expire, strikes

    @property
    def zscore(self): return self.__zscore
    @property
    def today(self): return self.__today



