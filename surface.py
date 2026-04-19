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
from collections import namedtuple as ntuple
from scipy.interpolate import CubicSpline, RectBivariateSpline

from support.finance import Concepts, Alerting
from support.equations import Equations
from support.concepts import NumRange

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SurfaceCalculator", "SurfaceCreator", "SurfacePlotter"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


Sample = ntuple("Sample", "dte mae tiv")
Curve = ntuple("Spline", "dte spline bounds")
Axes = ntuple("Axes", "t k w")
Domain = ntuple("Domain", "t k")
Range = ntuple("Range", "w")


class SurfaceError(Exception): pass
class SurfaceExtrapolationError(SurfaceError): pass


class Surface(object):
    def __init__(self, taxis, kaxis, waxis, *args, degree, **kwargs):
        assert isinstance(degree, Domain)
        surface = RectBivariateSpline(taxis, kaxis, waxis, kx=degree.t, ky=degree.k, s=0)
        self.__domain = Domain(taxis, kaxis)
        self.__range = Range(waxis)
        self.__surface = surface

    def w(self, t, k):
        t, k = np.asarray(t, dtype=float), np.asarray(k, dtype=float)
        if self.extrapolation(t, k): raise SurfaceExtrapolationError()
        return self.surface.ev(t, k)

    def dwdt(self, t, k):
        t, k = np.asarray(t, dtype=float), np.asarray(k, dtype=float)
        if self.extrapolation(t, k): raise SurfaceExtrapolationError()
        return self.surface.ev(t, k, dx=1, dy=0)

    def dwdk(self, t, k):
        t, k = np.asarray(t, dtype=float), np.asarray(k, dtype=float)
        if self.extrapolation(t, k): raise SurfaceExtrapolationError()
        return self.surface.ev(t, k, dx=0, dy=1)

    def dw2dt2(self, t, k):
        t, k = np.asarray(t, dtype=float), np.asarray(k, dtype=float)
        if self.extrapolation(t, k): raise SurfaceExtrapolationError()
        return self.surface.ev(t, k, dx=2, dy=0)

    def dw2dk2(self, t, k):
        t, k = np.asarray(t, dtype=float), np.asarray(k, dtype=float)
        if self.extrapolation(t, k): raise SurfaceExtrapolationError()
        return self.surface.ev(t, k, dx=0, dy=2)

    def iv(self, t, k):
        t, k = np.asarray(t, dtype=float), np.asarray(k, dtype=float)
        if self.extrapolation(t, k): raise SurfaceExtrapolationError()
        return np.sqrt(np.maximum(self.w(t, k), 0.0) / t)

    def extrapolation(self, taxis, kaxis):
        boundarys = self.boundarys()
        taxis = np.any((taxis < boundarys.t.minimum) | (taxis > boundarys.t.maximum))
        kaxis = np.any((kaxis < boundarys.k.minimum) | (kaxis > boundarys.k.maximum))
        return taxis | kaxis

    def interpolation(self, taxis, kaxis):
        boundarys = self.boundarys()
        taxis = np.all((taxis >= boundarys.t.minimum) | (taxis <= boundarys.t.maximum))
        kaxis = np.all((kaxis >= boundarys.k.minimum) | (kaxis <= boundarys.k.maximum))
        return taxis & kaxis

    def boundarys(self):
        taxis = NumRange.create([self.domain.t[0], self.domain.t[-1]])
        kaxis = NumRange.create([self.domain.k[0], self.domain.k[-1]])
        return Domain(taxis, kaxis)

    def axes(self):
        taxis = self.domain.t.copy()
        kaxis = self.domain.k.copy()
        waxis = self.range.w.copy()
        return Axes(taxis, kaxis, waxis)

    @property
    def surface(self): return self.__surface
    @property
    def domain(self): return self.__domain
    @property
    def range(self): return self.__range


class SurfacePlotter(Alerting):
    def __init__(self, *args, figsize=(10, 10), gridsize=100, **kwargs):
        super().__init__(*args, **kwargs)
        figure = plt.figure(figsize=figsize)
        ax = figure.add_subplot(111, projection="3d")
        ax.set_xlabel("DTE|t")
        ax.set_ylabel("MAE|k")
        ax.set_zlabel("TIV|w")
        self.__gridsize = int(gridsize)
        self.__figure = figure
        self.__ax = ax

    def __call__(self, options, surface, *args, **kwargs):
        self.options(options, *args, **kwargs)
        self.surface(surface, *args, **kwargs)
        plt.show()

    def surface(self, surface, *args, **kwargs):
        axes = surface.axes()
        t = np.linspace(axes.t.min(), axes.t.max(), self.gridsize)
        k = np.linspace(axes.k.min(), axes.k.max(), self.gridsize)
        tt, kk = np.meshgrid(t, k, indexing="ij")
        ww = surface.w(t, k)
        self.ax.plot_surface(tt, kk, ww, alpha=0.5)

    def options(self, options, *args, **kwargs):
        options = options[["mae", "dte", "tiv"]]
        mask = options["tiv"].notna()
        options = options.where(mask)
        options = options.dropna(how="all", inplace=False)
        self.ax.scatter(options["dte"], options["mae"], options["tiv"], s=10)

    @property
    def gridsize(self): return self.__gridsize
    @property
    def figure(self): return self.__figure
    @property
    def ax(self): return self.__ax


class SurfaceCalculator(Equations, Alerting):
    mae = lambda forward, strike, option: np.log10(forward / strike) * option.astype(int)
    dte = lambda expire: (pd.to_datetime(expire) - pd.Timestamp(Date.today())).dt.days
    tiv = lambda implied, tau: tau * np.square(implied)

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        surface = self.execute(options, *args, **kwargs)
        options = pd.concat([options, surface], axis=1)
        self.alert(options, title="Calculated", instrument=Concepts.Securities.Instrument.OPTION)
        return surface


class SurfaceCreator(Alerting):
    def __init__(self, *args, samplesize=5, gridsize=100, curvetype="natural", degree=(3, 3), **kwargs):
        super().__init__(*args, **kwargs)
        degree = (degree, degree) if isinstance(degree, int) else degree
        assert isinstance(degree, tuple) and len(degree) == 2
        self.__samplesize = int(samplesize)
        self.__curvetype = str(curvetype)
        self.__gridsize = int(gridsize)
        self.__degree = Domain(*degree)

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        samples = self.samples(options, *args, **kwargs)
        curves = self.curves(samples, *args, **kwargs)
        taxis = self.taxis(curves, *args, **kwargs)
        kaxis = self.kaxis(curves, *args, **kwargs)
        waxis = np.array([curve.spline(kaxis) for index, curve in enumerate(curves)])
        surface = Surface(taxis, kaxis, waxis, degree=self.degree)
        self.alert(options, title="Created", instrument=Concepts.Securities.Instrument.OPTION)
        return surface

    def samples(self, options, *args, **kwargs):
        samples = options[["dte", "mae", "tiv"]].groupby(["dte", "mae"], as_index=False)["tiv"].mean()
        samples = {dte: dataframe.sort_values("mae") for dte, dataframe in samples.groupby("dte", sort="dte")}
        samples = {dte: (dataframe["mae"].to_numpy(), dataframe["tiv"].to_numpy()) for dte, dataframe in samples.items()}
        samples = {dte: (mae, tiv) for dte, (mae, tiv) in samples.items() if len(mae) >= self.samplesize}
        samples = {dte: (mae, tiv) for dte, (mae, tiv) in samples.items() if not np.any(np.diff(mae) <= 0)}
        samples = {dte: (mae, tiv, np.argsort(mae)) for dte, (mae, tiv) in samples.items()}
        samples = [Sample(dte, mae[order], tiv[order]) for dte, (mae, tiv, order) in samples.items()]
        return samples

    def curves(self, samples, *args, **kwargs):
        splines = [CubicSpline(mae, tiv, bc_type=self.curvetype) for (dte, mae, tiv) in samples]
        boundarys = [NumRange.create([mae[0], mae[-1]]) for (dte, mae, tiv) in samples]
        curves = [Curve(sample.dte, spline, bounds) for (sample, spline, bounds) in zip(samples, splines, boundarys)]
        return curves

    @staticmethod
    def taxis(curves, *args, **kwargs): return np.array([curve.dte for curve in curves])
    def kaxis(self, curves, *args, **kwargs):
        left = max(curve.bounds.minimum for curve in curves)
        right = min(curve.bounds.maximum for curve in curves)
        assert left < right
        return np.linspace(left, right, self.gridsize)

    @property
    def samplesize(self): return self.__samplesize
    @property
    def curvetype(self): return self.__curvetype
    @property
    def gridsize(self): return self.__gridsize
    @property
    def degree(self): return self.__degree



