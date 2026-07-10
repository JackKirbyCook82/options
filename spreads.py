# -*- coding: utf-8 -*-
"""
Created on Sat May 16 2026
@name:   Option Spread Objects
@author: Jack Kirby Cook

"""

import pandas as pd
from types import SimpleNamespace

from finance.enumerations import Strategy
from finance.osi import OSI
from support.custom import DateRange

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Spread"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class Spread(object):
    def __init__(self, strategy, securities):
        assert isinstance(securities, pd.DataFrame)
        assert len(securities["ticker"].unique()) == 1
        assert strategy in list(Strategy)
        self.__expires = DateRange.create(securities["expire"].to_list())
        self.__ticker = securities["ticker"].unique()[0]
        self.__securities = securities
        self.__strategy = strategy

    def __iter__(self):
        for osi, position, quantity in zip(self.osi, self.position, self.quantity):
            yield SimpleNamespace(osi=osi, position=position, quantity=quantity)

    @property
    def signature(self): return tuple((str(record.osi), int(record.position), int(record.quantity)) for record in self)

    @property
    def osi(self): return self.securities[["ticker", "expire", "option", "strike"]].apply(OSI, axis=1)

    @property
    def gamma(self): return (self.securities["gamma"] * self.position.map(int) * self.quantity).sum()
    @property
    def theta(self): return (self.securities["theta"] * self.position.map(int) * self.quantity).sum()
    @property
    def vega(self): return (self.securities["vega"] * self.position.map(int) * self.quantity).sum()

    @property
    def value(self): return (self.securities["value"] * self.position.map(int) * self.quantity).sum()
    @property
    def cost(self): return (self.securities["median"] * self.position.map(int) * self.quantity).sum()
    @property
    def gap(self): return (self.securities["gap"] * self.quantity).sum()

    @property
    def tightness(self): return self.securities["tightness"].max()
    @property
    def moneyness(self): return self.securities["moneyness"].max()
    @property
    def activity(self): return self.securities["activity"].min()

    @property
    def position(self): return self.securities["position"]
    @property
    def quantity(self): return self.securities["quantity"]

    @property
    def securities(self): return self.securities
    @property
    def strategy(self): return self.strategy
    @property
    def expires(self): return self.__expires
    @property
    def ticker(self): return self.__ticker







