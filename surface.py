# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 2026
@name:   Surface Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from datetime import date as Date

from numba.core.types import NumPyRandomGeneratorType
from scipy.interpolate import CubicSpline
from collections import namedtuple as ntuple

from support.finance import Concepts, Alerting
from support.equations import Equations
from support.concepts import NumRange
from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SurfaceCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


Sample = ntuple("Sample", "dte mae tiv")
Curve = ntuple("Spline", "dte spline bounds")


class SurfaceCalculator(Equations, Alerting):
    mae = lambda forward, strike, option: np.log10(forward / strike) * option.astype(int)
    dte = lambda expire: (pd.to_datetime(expire) - pd.Timestamp(Date.today())).dt.days
    tiv = lambda implied, tau: tau * np.square(implied)

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        surface = self.execute(options, *args, **kwargs)
        surface = pd.concat([options, surface], axis=1)
        self.alert(options, title="Calculated", instrument=Concepts.Securities.Instrument.OPTION)
        return surface


class SurfaceSpline(Logging):
    def __init__(self, *args, samplesize=5, curvetype="natural", **kwargs):
        super().__init__(*args, **kwargs)
        self.__samplesize = int(samplesize)
        self.__curvetype = str(curvetype)

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        samples = self.samples(options, *args, **kwargs)
        curves = self.curves(samples, *args, **kwargs)


    def curves(self, samples, *args, **kwargs):
        splines = [CubicSpline(mae, tiv, bc_type=self.curvetype) for (dte, mae, tiv) in samples]
        boundarys = [NumRange.create([mae[0], mae[-1]]) for (dte, mae, tiv) in samples]
        curves = [Curve(sample.dte, spline, bounds) for (sample, spline, bounds) in zip(samples, splines, boundarys)]
        return curves

    def samples(self, options, *args, **kwargs):
        samples = options[["dte", "mae", "tiv"]].groupby(["dte", "mae"], as_index=False)["tiv"].mean()
        samples = {dte: dataframe.sort_values("mae") for dte, dataframe in samples.groupby("dte", sort="dte")}
        samples = {dte: (dataframe["mae"].to_numpy(), dataframe["tiv"].to_numpy()) for dte, dataframe in samples.items()}
        samples = {dte: (mae, tiv) for dte, (mae, tiv) in samples.items() if len(mae) >= self.samplesize}
        samples = {dte: (mae, tiv) for dte, (mae, tiv) in samples.items() if not np.any(np.diff(mae) <= 0)}
        samples = {dte: (mae, tiv, np.argsort(mae)) for dte, (mae, tiv) in samples.items()}
        samples = [Sample(dte, mae[order], tiv[order]) for dte, (mae, tiv, order) in samples.items()]
        return samples

    @property
    def samplesize(self): return self.__samplesize
    @property
    def curvetype(self): return self.__curvetype



