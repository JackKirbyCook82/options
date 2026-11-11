# -*- coding: utf-8 -*-
"""
Created on Weds Nov 4 2026
@name:   Settlement Objects
@author: Jack Kirby Cook
@file:   options/settlements.py

"""

import numpy as np
import pandas_market_calendars as calendar
from dataclasses import dataclass
from datetime import date as Date
from datetime import timedelta as Timedelta
from collections import OrderedDict as ODict

from support.equations import Equation
from support.custom import NumberRange
from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SettlementCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


nyse = calendar.get_calendar("NYSE")


@dataclass(frozen=True, slots=True)
class Settlement: expire: Date; strike: NumberRange

@dataclass
class Settlements:
    aslist: list

    def __post_init__(self):
        self.asdict = ODict([(settlement.expire, settlement.strike) for settlement in self.aslist])

    def __len__(self): return len(self.aslist)
    def __iter__(self): return iter(self.aslist)

    def __contains__(self, settlement):
        assert isinstance(settlement, tuple) and len(settlement) == 2
        expire, strike = settlement
        strikes = self.asdict.get(expire)
        return strikes is not None and strike in strikes


class SettlementCalculator(Equation, Logging, variables=["strike"]):
    tau = lambda expire, today: len(nyse.valid_days(start_date=today + Timedelta(days=1), end_date=expire)) / 252
    forward = lambda underlying, expire, today, tau, interest, dividends: underlying * np.exp(np.subtract(interest, dividends) * tau)
    strike = lambda forward, zscore, tau, volatility: forward * np.exp(zscore * volatility * np.sqrt(tau))

    def __init__(self, *args, today, zscore, **kwargs):
        super().__init__(*args, **kwargs)
        self.__zscore = zscore
        self.__today = today

    def __call__(self, expires, /, **kwargs):
        if expires is None: return None
        assert isinstance(expires, list) and all([isinstance(expire, Date) for expire in expires])
        if not bool(expires): return None
        settlements = self.calculator(expires, **kwargs)
        settlements = list(settlements)
        return Settlements(settlements)

    def calculator(self, expires, /, underlying, volatility, dividends, interest, **kwargs):
        parameters = dict(underlying=underlying, volatility=volatility, dividends=dividends, interest=interest)
        for expire in expires:
            minimum = self.evaluate(expire=expire, today=self.today, zscore=-self.zscore, **parameters)["strike"]
            maximum = self.evaluate(expire=expire, today=self.today, zscore=+self.zscore, **parameters)["strike"]
            strikes = NumberRange([minimum, maximum])
            settlement = Settlement(expire, strikes)
            yield settlement

    @property
    def zscore(self): return self.__zscore
    @property
    def today(self): return self.__today



