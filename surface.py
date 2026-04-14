# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 2026
@name:   Surface Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from datetime import date as Date

from support.finance import Concepts, Alerting
from support.equations import Equations

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SurfaceCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class SurfaceCalculator(Equations, Alerting):
    mae = lambda forward, strike, option: np.log10(forward / strike) * option.astype(int)
    tau = lambda expire: (pd.to_datetime(expire) - pd.Timestamp(Date.today())).dt.days
    tiv = lambda implied, tau: tau * np.square(implied)

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if bool(options.empty): return options
        contract = options[["ticker", "expire", "strike", "option"]]
        surface = self.equate(options, *args, **kwargs)
        surface = pd.concat([contract, surface], axis=1)
        self.alert(options, instrument=Concepts.Securities.Instrument.OPTION)
        return surface



