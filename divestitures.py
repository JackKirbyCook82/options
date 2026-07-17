# -*- coding: utf-8 -*-
"""
Created on Mon Jul 6 2026
@name:   Option Divestiture Objects
@author: Jack Kirby Cook

"""

from options.spreads import Spread

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Divestiture"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class Divestiture(Spread):
    @property
    def value(self): return (self.securities["value"] * self.position.map(int) * self.quantity).sum()
    @property
    def market(self):
        asks = (self.securities["ask"] * ((self.position.map(int) - 1) / 2) * self.quantity).sum()  # SHORT POSITION (-)
        bids = (self.securities["bid"] * ((self.position.map(int) + 1) / 2) * self.quantity).sum()  # LONG POSITION (+)
        return bids - asks


