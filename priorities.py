# -*- coding: utf-8 -*-
"""
Created on Sat May 16 2026
@name:   Option Priority Objects
@author: Jack Kirby Cook

"""

from support.finance import Alerting

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["PriorityCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class PriorityCalculator(Alerting):
    def __call__(self, spread, *args, **kwargs):
        pass

