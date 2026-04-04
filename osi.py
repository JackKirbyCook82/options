# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 2026
@name:   Market Objects
@author: Jack Kirby Cook

"""

import regex as re
from enum import Enum
from datetime import date as Date
from datetime import datetime as Datetime
from dataclasses import dataclass, asdict, fields

from support.finance import Concepts, Querys

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["OptionOSI"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class OSIError(Exception): pass
class OSIParseError(OSIError): pass
class OSICreateError(OSIError): pass

@dataclass(frozen=True)
class OptionOSI:
    ticker: str; expire: Date; option: Enum; strike: float

    @classmethod
    def create(cls, contents):
        if isinstance(contents, Querys.Contract): contents = dict(contents.items())
        if isinstance(contents, dict): return cls(**{field.name: contents[field.name] for field in fields(cls)})
        elif isinstance(contents, str): return cls(*cls.parse(contents))
        elif isinstance(contents, (list, tuple)): return cls(*contents)
        else: raise OSICreateError()

    def __str__(self):
        ticker = self.ticker.upper()
        expire = self.expire.strftime("%y%m%d")
        option = str(self.option).upper()[0]
        strike_int = int(round(self.strike * 1000))
        strike = f"{strike_int:08d}"
        return f"{ticker}{expire}{option}{strike}"

    def items(self): return asdict(self).items()
    def values(self): return asdict(self).values()
    def keys(self): return asdict(self).keys()

    @classmethod
    def parse(cls, contents):
        pattern = r"^(?P<ticker>[A-Z]+)(?P<expire>\d{6})(?P<option>[PC])(?P<strike>\d{8})$"
        match = re.search(pattern, contents)
        if not match: raise OSIParseError()
        values = match.groupdict()
        ticker = values["ticker"].upper()
        expire = Datetime.strptime(values["expire"], "%y%m%d").date()
        option = {str(option).upper()[0]: option for option in Concepts.Securities.Option}[values["option"]]
        strike = int(values["strike"]) / 1000.0
        return [ticker, expire, option, strike]



