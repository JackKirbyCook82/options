# -*- coding: utf-8 -*-
"""
Created on Mon Jul 6 2026
@name:   Option Divestiture Objects
@author: Jack Kirby Cook

"""

from options.prospects import Prospect

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Divestiture"]
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



