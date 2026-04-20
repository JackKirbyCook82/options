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


Curve = ntuple("Curve", "sample spline bounds")
Sample = ntuple("Sample", "t k w")
Axes = ntuple("Axes", "t k w")
Domain = ntuple("Domain", "t k")
Range = ntuple("Range", "w")


class SurfaceError(Exception): pass
class SurfaceExtrapolationError(SurfaceError): pass


class SurfaceCalculator(Equations, Alerting):
    mae = lambda forward, strike, option: np.log(forward / strike) * option.astype(int)
    tiv = lambda implied, tau: tau * np.square(implied)

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        surface = self.execute(options, *args, **kwargs)
        surface = pd.concat([options, surface], axis=1)
        self.alert(options, title="Calculated", instrument=Concepts.Securities.Instrument.OPTION)
        return surface


class Curves(object):
    def __iter__(self): return iter(self.curves)
    def __init__(self, options, *args, samplesize=5, curvetype="natural", **kwargs):
        assert isinstance(options, pd.DataFrame)
        samples = self.samples(options, *args, size=samplesize, **kwargs)
        splines = [CubicSpline(mae, tiv, bc_type=curvetype) for (tau, mae, tiv) in samples]
        boundarys = [NumRange.create([mae[0], mae[-1]]) for (tau, mae, tiv) in samples]
        curves = [Curve(sample, spline, bounds) for (sample, spline, bounds) in zip(samples, splines, boundarys)]
        self.__curves = curves

    @staticmethod
    def samples(options, *args, size, **kwargs):
        samples = options[["tau", "mae", "tiv"]].dropna(how="any", inplace=False)
        samples = samples.groupby(["tau", "mae"], as_index=False)["tiv"].mean()
        samples = {tau: dataframe.sort_values("mae") for tau, dataframe in samples.groupby("tau", sort="tau")}
        samples = {tau: (dataframe["mae"].to_numpy(), dataframe["tiv"].to_numpy()) for tau, dataframe in samples.items()}
        samples = {tau: (mae, tiv) for tau, (mae, tiv) in samples.items() if len(mae) >= size}
        samples = {tau: (mae, tiv) for tau, (mae, tiv) in samples.items() if not np.any(np.diff(mae) <= 0)}
        samples = {tau: (mae, tiv, np.argsort(mae)) for tau, (mae, tiv) in samples.items()}
        samples = [Sample(tau, mae[order], tiv[order]) for tau, (mae, tiv, order) in samples.items()]
        return samples

    @property
    def curves(self): return self.__curves


class Surface(object):
    def __init__(self, taxis, kaxis, waxis, *args, degree, smoothing, **kwargs):
        assert isinstance(degree, Domain)
        surface = RectBivariateSpline(taxis, kaxis, waxis, kx=degree.t, ky=degree.k, s=smoothing)
        self.__domain = Domain(taxis, kaxis)
        self.__range = Range(waxis)
        self.__surface = surface

    def __call__(self, t, k):
        t, k = np.asarray(t, dtype=float), np.asarray(k, dtype=float)
        if self.extrapolation(t, k): raise SurfaceExtrapolationError()
        return self.surface(t, k)

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


class SurfaceCreator(Alerting):
    def __init__(self, *args, samplesize=5, curvetype="natural", gridsize=100, degree=(2, 3), smoothing=1e-2, **kwargs):
        super().__init__(*args, **kwargs)
        degree = (degree, degree) if isinstance(degree, int) else degree
        assert isinstance(degree, tuple) and len(degree) == 2
        self.__smoothing = float(smoothing)
        self.__samplesize = int(samplesize)
        self.__curvetype = str(curvetype)
        self.__gridsize = int(gridsize)
        self.__degree = Domain(*degree)

    def __call__(self, options, *args, **kwargs):
        parameters = dict(samplesize=self.samplesize, curvetype=self.curvetype)
        curves = Curves(options, **parameters)
        left = max(curve.bounds.minimum for curve in curves)
        right = min(curve.bounds.maximum for curve in curves)
        assert left < right
        taxis = np.array([curve.sample.t for curve in curves])
        kaxis = np.linspace(left, right, self.gridsize)
        waxis = np.array([curve.spline(kaxis) for index, curve in enumerate(curves)])
        surface = Surface(taxis, kaxis, waxis, degree=self.degree, smoothing=self.smoothing)
        self.alert(options, title="Created", instrument=Concepts.Securities.Instrument.OPTION)
        return surface

    @property
    def samplesize(self): return self.__samplesize
    @property
    def curvetype(self): return self.__curvetype
    @property
    def smoothing(self): return self.__smoothing
    @property
    def gridsize(self): return self.__gridsize
    @property
    def degree(self): return self.__degree


class SurfacePlotter(Alerting):
    def __init__(self, *args, figsize=(10, 10), gridsize=100, **kwargs):
        super().__init__(*args, **kwargs)
        figure = plt.figure(figsize=figsize)
        ax = figure.add_subplot(111, projection="3d")
        ax.set_xlabel("TAU|t")
        ax.set_ylabel("MAE|k")
        ax.set_zlabel("TIV|w")
        self.__gridsize = int(gridsize)
        self.__figure = figure
        self.__ax = ax

    def __call__(self, options, surface, *args, **kwargs):
        self.options(options, *args, **kwargs)
        self.surface(surface, *args, **kwargs)
        self.alert(options, title="Plotted", instrument=Concepts.Securities.Instrument.OPTION)
        plt.show()

    def surface(self, surface, *args, **kwargs):
        axes = surface.axes()
        t = np.linspace(axes.t.min(), axes.t.max(), self.gridsize)
        k = np.linspace(axes.k.min(), axes.k.max(), self.gridsize)
        tt, kk = np.meshgrid(t, k, indexing="ij")
        ww = surface(t, k)
        self.ax.plot_surface(tt, kk, ww, alpha=0.75, color="blue")

    def options(self, options, *args, **kwargs):
        options = options[["tau", "mae", "tiv"]]
        mask = options["tiv"].notna()
        options = options.where(mask)
        options = options.dropna(how="all", inplace=False)
        self.ax.scatter(options["tau"], options["mae"], options["tiv"], s=30, color="red")

    @property
    def gridsize(self): return self.__gridsize
    @property
    def figure(self): return self.__figure
    @property
    def ax(self): return self.__ax



