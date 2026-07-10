# -*- coding: utf-8 -*-
"""
Created on Mon Jul 6 2026
@name:   Option Divestiture Objects
@author: Jack Kirby Cook

"""

import pandas as pd

from finance.logging import Logging
from options.spreads import Spread

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DivestitureCalculator", "Divestiture"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class Divestiture(Spread):
    pass


class DivestitureCalculator(Logging):
    def __call__(self, options, **kwargs):
        assert isinstance(options, pd.DataFrame)



