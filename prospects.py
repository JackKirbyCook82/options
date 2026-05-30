# -*- coding: utf-8 -*-
"""
Created on Tues May 12 2026
@name:   Option Prospect Objects
@author: Jack Kirby Cook

"""

from finance.variables import Alerting, Enumerations

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ProspectCalculator", "PriorityCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class ProspectCalculator(Alerting):
    def __init__(self, *args, metrics, **kwargs):
        super().__init__(*args, **kwargs)
        metrics = {Enumerations.Spread(key): value for key, value in metrics.items()}
        self.__metrics = metrics

    def __call__(self, spreads, *args, **kwargs):
        assert isinstance(spreads, list)
        generator = self.calculator(spreads, *args, **kwargs)
        prospects = list(generator)
        sizes = dict(previous=len(spreads), post=len(prospects))
        self.alert(prospects, title="Calculator", instrument=Enumerations.Instrument.OPTION, **sizes)
        return prospects

    def calculator(self, spreads, *args, **kwargs):
        assert isinstance(spreads, list)
        for spread in spreads:
            metrics = self.metrics[spread.type]
            qualify = self.prospect(spread, metrics)
            if not qualify: continue
            yield spread

    @staticmethod
    def prospect(spread, metrics):
        profit = spread.profit >= metrics.profit
        zscore = abs(spread.zscore) >= abs(metrics.zscore)
        gap = spread.ratios.gap <= metrics.ratios.gap
        try: gamma = spread.ratios.gamma <= metrics.ratios.gamma
        except TypeError: gamma = True
        try: theta = spread.ratios.theta >= metrics.ratios.theta
        except TypeError: theta = True
        try: vega = spread.ratios.vega >= metrics.ratios.vega
        except TypeError: vega = True
        return all([profit, zscore, gap, gamma, theta, vega])

    @property
    def metrics(self): return self.__metrics


class PriorityCalculator(Alerting):
    def __call__(self, prospects, *args, **kwargs):
        assert isinstance(prospects, list)
        priorities = self.calculate(prospects, *args, **kwargs)
        sizes = dict(previous=len(prospects), post=len(priorities))
        self.alert(priorities, title="Calculator", instrument=Enumerations.Instrument.OPTION, **sizes)
        return priorities

    def calculate(self, prospects, *args, **kwargs):
        assert isinstance(prospects, list)
        metrics = lambda spread: spread.score
        return self.priority(prospects, metrics)

    @staticmethod
    def priority(prospects, metrics):
        priorities = sorted(prospects, key=metrics, reverse=True)
        return priorities



