# -*- coding: utf-8 -*-
"""
Created on Tues May 12 2026
@name:   Option Scanner Objects
@author: Jack Kirby Cook

"""

from types import SimpleNamespace
from abc import ABC, abstractmethod

from support.finance import Concepts, Alerting

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["CalenderScanner", "FlyScanner"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class Scanner(Alerting, ABC):
    def __init__(self, *args, width=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.__width = int(width)

    def indexes(self, dataframe):
        length = len(dataframe)
        for index in range(length - 2 * self.width):
            left = index
            center = index + self.width
            right = index + self.width * 2
            index = SimpleNamespace(left=left, center=center, right=right)
            yield index

    @abstractmethod
    def generator(self, options, *args, **kwargs): pass

    @property
    def width(self): return self.__width


class CalenderScanner(Scanner):
    def __call__(self, options, *args, **kwargs):
        pass

    def generator(self, options, *args, **kwargs):
        pass


class FlyScanner(Scanner):
    def __call__(self, options, *args, **kwargs):
        pass

    def generator(self, options, *args, **kwargs):
        for position, hedge in zip(iter(Concepts.Securities.Position), reversed(Concepts.Securities.Position)):
            for option in iter(Concepts.Securities.Option):
                dataframes = options[options["option"].eq(option)]
                for dte, dataframe in dataframes.groupby("dte"):
                    dataframe = dataframe.sort_values("strike")
                    for index in self.indexes(dataframe):

                        left = dataframe.iloc[index.left]
                        center = dataframe.iloc[index.center]
                        right = dataframe.iloc[index.right]

                        print(dataframe)
                        raise Exception()


