# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 2026
@name:   Surface Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date as Date
from mpl_toolkits.mplot3d import Axes3D

from support.finance import Concepts, Alerting
from support.equations import Equations
from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SurfaceCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class SurfaceCalculator(Equations, Alerting):
    mae = lambda forward, strike, option: np.log10(forward / strike) * option.astype(int)
    dte = lambda expire: (pd.to_datetime(expire) - pd.Timestamp(Date.today())).dt.days
    tiv = lambda implied, tau: tau * np.square(implied)

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if bool(options.empty): return options
        surface = self.execute(options, *args, **kwargs)
        surface = pd.concat([options, surface], axis=1)
        self.alert(options, title="Calculated", instrument=Concepts.Securities.Instrument.OPTION)
        return surface


class SurfacePlotter(Logging):
    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame) and not bool(options.empty)
        surface = options[["mae", "dte", "tiv"]]
        mask = surface["tiv"].notna()
        surface = surface.where(mask)
        surface = surface.dropna(how="all", inplace=False)
        figure = plt.figure(figsize=(10, 10))
        ax = figure.add_subplot(111, projection="3d")
        dte, mae, tiv = (surface["dte"], surface["mae"], surface["tiv"])
        ax.scatter(dte, mae, tiv, s=20)
        ax.plot_trisurf(dte, mae, tiv, alpha=0.5)
        plt.show()




