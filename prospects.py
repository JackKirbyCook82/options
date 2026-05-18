# -*- coding: utf-8 -*-
"""
Created on Tues May 12 2026
@name:   Option Prospect Objects
@author: Jack Kirby Cook

"""

import pandas as pd
from abc import ABC, abstractmethod

from support.finance import Concepts, Alerting
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ProspectCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class Scanner(ABC, metaclass=RegistryMeta):
    def __init__(self, *args, metrics, proximity=1, **kwargs):
        self.__proximity = int(proximity)
        self.__metrics = metrics

    def __call__(self, options, *args, **kwargs):
        for spread in self.generator(options):
            prospects = spread(self.metrics, *args, **kwargs)
            if bool(prospects.empty): continue
            yield prospects

    @abstractmethod
    def selector(self, length): pass
    @abstractmethod
    def generator(self, options): pass

    @property
    def proximity(self): return self.__proximity
    @property
    def metrics(self): return self.__metrics


#    def __call__(self, metrics, *args, **kwargs):
#        columns = ["identity"] + list(self.legs.columns)
#        if not self.qualify(metrics): return pd.DataFrame(columns=columns)
#        identity = type(self).counter
#        prospects = self.legs.assign(identity=identity)
#        return prospects

#    def qualify(self, metrics):
#        assert isinstance(metrics, Metrics)
#        if self.profit < metrics.profit: return False
#        ratios = all([self.ratios.gap <= metrics.ratios.gap, self.ratios.theta >= metrics.ratios.theta])
#        zscore = abs(self.zscore) >= abs(metrics.zscore)
#        quality = self.quality >= metrics.quality
#        gamma = True if metrics.gamma is None else abs(self.gamma) <= abs(metrics.gamma)
#        theta = True if metrics.theta is None else self.theta >= metrics.theta
#        vega = True if metrics.vega is None else self.vega > metrics.vega
#        return all([ratios, zscore, quality, gamma, theta, vega])


class ProspectCalculator(Alerting):
    def __init__(self, *args, metrics, **kwargs):
        super().__init__(*args, **kwargs)
        metrics = {Concepts.Strategies.Spread[str(key).upper()]: value for key, value in metrics.items()}
        self.__metrics = metrics

    def __call__(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        prospects = self.scanner(options, *args, **kwargs)
        prospects = list(prospects)
        if bool(prospects): prospects = pd.concat(prospects, axis=0)
        else: prospects = pd.DataFrame(columns=options.columns)
        prospects = prospects.sort_values(by=["identity"], ascending=True, inplace=False)
        prospects = prospects.reset_index(drop=True, inplace=False)
        sizes = dict(previous=len(options), post=len(prospects))
        self.alert(options, title="Calculated", instrument=Concepts.Securities.Instrument.OPTION, **sizes)
        return prospects

    @property
    def metrics(self): return self.__metrics



