# -*- coding: utf-8 -*-

"""This package defines design template classes.
"""

from typing import Iterable

import numpy as np
from . import interpolate

__all__ = ['lcm', 'gcd', 'interpolate', 'float_to_si_string', 'si_string_to_float']


si_mag = [-18, -15, -12, -9, -6, -3, 0, 3, 6, 9, 12]
si_pre = ['a', 'f', 'p', 'n', 'u', 'm', '', 'k', 'M', 'G', 'T']


def float_to_si_string(num, precision=6):
    """Converts the given floating point number to a string using SI prefix.

    Parameters
    ----------
    num : float
        the number to convert.
    precision : int
        number of significant digits, defaults to 6.

    Returns
    -------
    ans : str
        the string representation of the given number using SI suffix.
    """
    if abs(num) < 1e-21:
        return '0'
    exp = np.log10(abs(num))

    pre_idx = len(si_mag) - 1
    for idx in range(len(si_mag)):
        if exp < si_mag[idx]:
            pre_idx = idx - 1
            break

    fmt = '%%.%dg%%s' % precision
    res = 10.0 ** (si_mag[pre_idx])
    return fmt % (num / res, si_pre[pre_idx])


def si_string_to_float(si_str):
    """Converts the given string with SI prefix to float.

    Parameters
    ----------
    si_str : str
        the string to convert

    Returns
    -------
    ans : float
        the floating point value of the given string.
    """
    if si_str[-1] in si_pre:
        idx = si_pre.index(si_str[-1])
        return float(si_str[:-1]) * 10**si_mag[idx]
    else:
        return float(si_str)


def gcd(a, b):
    # type: (int, int) -> int
    """Compute greatest common divisor of two positive integers.

    Parameters
    ----------
    a : int
        the first number.
    b : int
        the second number.

    Returns
    -------
    ans : int
        the greatest common divisor of the two given integers.
    """
    while b:
        a, b = b, a % b
    return a


def lcm(arr, init=1):
    # type: (Iterable[int], int) -> int
    """Compute least common multiple of all numbers in the given list.

    Parameters
    ----------
    arr : Iterable[int]
        a list of integers.
    init : int
        the initial LCM.  Defaults to 1.

    Returns
    -------
    ans : int
        the least common multiple of all the given numbers.
    """
    cur_lcm = init
    for val in arr:
        cur_lcm = cur_lcm * val // gcd(cur_lcm, val)
    return cur_lcm
