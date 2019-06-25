# -*- coding: utf-8 -*-

"""This is the bag root package.
"""

import signal

from . import math
from .math import float_to_si_string, si_string_to_float
from . import interface
from . import design
from . import data
from . import tech
from . import layout

from .core import BagProject, create_tech_info

__all__ = ['interface', 'design', 'data', 'math', 'tech', 'layout', 'BagProject',
           'float_to_si_string', 'si_string_to_float', 'create_tech_info']

# make sure that SIGINT will always be catched by python.
signal.signal(signal.SIGINT, signal.default_int_handler)
