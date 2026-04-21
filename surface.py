# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 2026
@name:   Surface Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from scipy.interpolate import UnivariateSpline, CubicSpline, PchipInterpolator, Akima1DInterpolator, RectBivariateSpline, SmoothBivariateSpline

from support.finance import Concepts, Alerting
from support.equations import Equations
from support.concepts import NumRange

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SurfaceCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class Curve(ABC):
    def __init__(self, xaxis, yaxis, /, **kwargs):
        assert isinstance(xaxis, (pd.Series, np.ndarray)) and isinstance(yaxis, (pd.Series, np.ndarray))
        assert len(xaxis) == len(yaxis)
        xaxis = np.asarray(xaxis, dtype=np.float32)
        yaxis = np.asarray(yaxis, dtype=np.float32)
        order = np.argsort(xaxis)
        xaxis, yaxis = (xaxis[order], yaxis[order])
        curve = self.create(xaxis, yaxis, **kwargs)
        boundary = NumRange.create([xaxis.min(), yaxis.max()])
        self.__boundary = boundary
        self.__curve = curve

    @staticmethod
    @abstractmethod
    def create(xaxis, yaxis, /, method, weights, smoothing, degree, **kwargs): pass

    @property
    def boundary(self): return self.__boundary
    @property
    def curve(self): return self.__curve


class RegressiveCurve(Curve):
    @staticmethod
    def create(xaxis, yaxis, /, weights, smoothing, degree, **kwargs):
        return UnivariateSpline(xaxis, yaxis, w=weights, s=smoothing, k=degree, ext=2)

class InterpolativeCurve(Curve):
    @staticmethod
    def create(xaxis, yaxis, /, method, **kwargs): return CubicSpline(xaxis, yaxis, bc_type=method)

class ShapeInterpolativeCurve(InterpolativeCurve):
    @staticmethod
    def create(xaxis, yaxis, /, **kwargs): return PchipInterpolator(xaxis, yaxis, extrapolate=False)

class VisualInterpolativeCurve(InterpolativeCurve):
    @staticmethod
    def create(xaxis, yaxis, /, **kwargs): return Akima1DInterpolator(xaxis, yaxis)


class Surface(ABC):
    pass





# RegularGridInterpolator (2D, interpolation)
# RectBivariateSpline (2D, interpolation)
# SmoothBivariateSpline (2D, regression)

# Curve = ntuple("Curve", "sample spline bounds")
# Sample = ntuple("Sample", "t k w")
# Domain = ntuple("Domain", "t k")

class SurfaceError(Exception): pass
class SurfaceExtrapolationError(SurfaceError): pass


class Surface(object):
    def __init__(self, *args, surface, domain, curves=None, **kwargs):
        self.__surface = surface
        self.__curves = curves
        self.__domain = domain

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

    def extrapolation(self, t, k):
        boundarys = self.boundarys()
        t = np.any((t < boundarys.t.minimum) | (t > boundarys.t.maximum))
        k = np.any((k < boundarys.k.minimum) | (k > boundarys.k.maximum))
        return t | k

    def interpolation(self, t, k):
        boundarys = self.boundarys()
        t = np.all((t >= boundarys.t.minimum) & (t <= boundarys.t.maximum))
        k = np.all((k >= boundarys.k.minimum) & (k <= boundarys.k.maximum))
        return t & k

    def boundarys(self):
        t = NumRange.create([self.domain.t[0], self.domain.t[-1]])
        k = NumRange.create([self.domain.k[0], self.domain.k[-1]])
        return Domain(t, k)

    @property
    def surface(self): return self.__surface
    @property
    def curves(self): return self.__curves
    @property
    def domain(self): return self.__domain


class CurveInterpolation(list):
    def __init__(self, options, *args, samplesize=5, curvetype="natural", **kwargs):
        assert isinstance(options, pd.DataFrame)
        samples = options[["tau", "mae", "tiv"]].dropna(how="any", inplace=False)
        samples = samples.groupby(["tau", "mae"], as_index=False)["tiv"].mean()
        samples = self.samples(samples, *args, size=samplesize, **kwargs)
        splines = [CubicSpline(mae, tiv, bc_type=curvetype) for (tau, mae, tiv) in samples]
        boundarys = [NumRange.create([mae[0], mae[-1]]) for (tau, mae, tiv) in samples]
        curves = [Curve(sample, spline, bounds) for (sample, spline, bounds) in zip(samples, splines, boundarys)]
        super().__init__(curves)

    @staticmethod
    def samples(samples, *args, size, **kwargs):
        for tau, sample in samples.groupby("tau", sort="tau"):
            sample = sample.sort_value("mae")
            mae = sample["mae"].to_numpy()
            tiv = sample["tiv"].to_numpy()
            if len(mae) < size: continue
            if np.any(np.diff(mae) > 0): continue
            order = np.argsort(mae)
            sample = Sample(tau, mae[order], tiv[order])
            yield sample


class SurfaceInterpolation(Surface):
    def __init__(self, options, *args, degree=(2, 3), smoothing=1e-4, gridsize=100, **kwargs):
        curves = CurveInterpolation(options, *args, **kwargs)
        left = max(curve.bounds.minimum for curve in curves)
        right = min(curve.bounds.maximum for curve in curves)
        assert left < right
        t = np.array([curve.sample.t for curve in curves])
        k = np.linspace(left, right, gridsize)
        w = np.array([curve.spline(k) for index, curve in enumerate(curves)])
        surface = RectBivariateSpline(t, k, w, kx=degree.t, ky=degree.k, s=smoothing)
        parameters = dict(surface=surface, curves=curves, domain=Domain(t, k))
        super().__init__(*args, **parameters, **kwargs)


class SurfaceRegression(Surface):
    def __init__(self, options, *args, degree=(2, 3), smoothing=1e-4, gridsize=100, **kwargs):
        samples = options[["tau", "mae", "tiv", "quality"]].dropna(how="any", inplace=False)
        tau = samples["tau"].to_numpy(dtype=float)
        mae = samples["mae"].to_numpy(dtype=float)
        tiv = samples["tiv"].to_numpy(dtype=float)
        quality = samples["quality"].to_numpy(dtype=float)
        t = np.linspace(tau.min(), tau.max(), gridsize)
        k = np.linspace(mae.min(), mae.max(), gridsize)
        surface = SmoothBivariateSpline(tau, mae, tiv, w=quality, kx=degree.t, ky=degree.k, s=smoothing)
        parameters = dict(surface=surface, curves=None, domain=Domain(t, k))
        super().__init__(*args, **parameters, **kwargs)


class SurfaceCalculator(Equations, Alerting):
    mae = lambda forward, strike, option: np.log(forward / strike) * option.astype(int)
    tiv = lambda implied, tau: tau * np.square(implied)

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        surface = self.execute(options, *args, **kwargs)
        surface = pd.concat([options, surface], axis=1)
        self.alert(options, title="Calculated", instrument=Concepts.Securities.Instrument.OPTION)
        return surface



