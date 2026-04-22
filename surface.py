# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 2026
@name:   Surface Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from support.finance import Concepts, Alerting
from support.equations import Equations

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SurfaceCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class SurfaceCalculator(Equations, Alerting):
    mae = lambda forward, strike, option: np.log(forward / strike) * option.astype(int)
    tiv = lambda implied, tau: tau * np.square(implied)

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        surface = self.execute(options, *args, **kwargs)
        surface = pd.concat([options, surface], axis=1)
        self.alert(options, title="Calculated", instrument=Concepts.Securities.Instrument.OPTION)
        return surface


class SurfacePlotter(Alerting):
    def __init__(self, *args, layout=(1, 1), plotsize=8, gridsize=100, **kwargs):
        super().__init__(*args, **kwargs)
        figsize = tuple(dim * plotsize for dim in layout)
        figure = plt.figure(figsize=figsize)
        self.__gridsize = int(gridsize)
        self.__layout = tuple(layout)
        self.__figure = figure

    def __call__(self, options, *args, surfaces, **kwargs):
        options = options[["tau", "mae", "tiv"]].dropna(how="any", inplace=False)
        tau = options["tau"].to_numpy(dtype=float)
        mae = options["mae"].to_numpy(dtype=float)
        tiv = options["tiv"].to_numpy(dtype=float)

        for index, surface in enumerate(surfaces, start=1):
            x = np.linspace(surface.domain.x.min(), surface.domain.x.max(), self.gridsize)
            y = np.linspace(surface.domain.y.min(), surface.domain.y.max(), self.gridsize)
            xx, yy = np.meshgrid(x, y, indexing="ij")
            zz = surface(x, y)
            ax = self.figure.add_subplot(*self.layout, index, projection="3d")
            ax.set_xlabel("t"), ax.set_ylabel("k"), ax.set_zlabel("w")
            ax.plot_surface(xx, yy, zz, alpha=0.75, color="blue")
            ax.scatter(tau, mae, tiv, s=30, color="red")

        self.alert(options, title="Plotted", instrument=Concepts.Securities.Instrument.OPTION)
        plt.show()

    @property
    def gridsize(self): return self.__gridsize
    @property
    def layout(self): return self.__layout
    @property
    def figure(self): return self.__figure



