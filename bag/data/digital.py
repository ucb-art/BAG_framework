# -*- coding: utf-8 -*-

"""This module defines functions useful for digital verification/postprocessing.
"""

from typing import Optional, List, Tuple

import numpy as np

from .core import Waveform


def de_bruijn(n, symbols=None):
    # type: (int, Optional[List[float]]) -> List[float]
    """Returns a De Bruijn sequence with subsequence of length n.

    a De Bruijn sequence with subsequence of length n is a sequence such that
    all possible subsequences of length n appear exactly once somewhere in the
    sequence.  This method is useful for simulating the worst case eye diagram
    given finite impulse response.

    Parameters
    ----------
    n : int
        length of the subsequence.
    symbols : Optional[List[float]] or None
        the list of symbols.  If None, defaults to [0.0, 1.0].

    Returns
    -------
    seq : List[float]
        the de bruijn sequence.
    """
    symbols = symbols or [0.0, 1.0]
    k = len(symbols)

    a = [0] * (k * n)
    sequence = []

    def db(t, p):
        if t > n:
            if n % p == 0:
                sequence.extend(a[1:p + 1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                db(t + 1, t)

    db(1, 1)
    return [symbols[i] for i in sequence]


def dig_to_pwl(values, tper, trf, td=0):
    # type: (List[float], float, float, float) -> Tuple[List[float], List[float]]
    """Convert a list of digital bits to PWL waveform.

    This function supports negative delay.  However, time/value pairs for negative data
    are truncated.

    Parameters
    ----------
    values : List[float]
        list of values for each bit.
    tper : float
        the period in seconds.
    trf : float
        the rise/fall time in seconds.
    td : float
        the delay

    Returns
    -------
    tvec : List[float]
        the time vector.
    yvec : List[float]
        the value vector.
    """
    y0 = values[0]
    tcur, ycur = td, y0
    tvec, yvec = [], []
    for v in values:
        if v != ycur:
            if tcur >= 0:
                tvec.append(tcur)
                yvec.append(ycur)
            elif tcur < 0 < tcur + trf:
                # make sure time starts at 0
                tvec.append(0)
                yvec.append(ycur - (v - ycur) / trf * tcur)
            ycur = v
            if tcur + trf >= 0:
                tvec.append(tcur + trf)
                yvec.append(ycur)
            elif tcur + trf < 0 < tcur + tper:
                # make sure time starts at 0
                tvec.append(0)
                yvec.append(ycur)
            tcur += tper
        else:
            if tcur <= 0 < tcur + tper:
                # make sure time starts at 0
                tvec.append(0)
                yvec.append(ycur)
            tcur += tper

    if not tvec:
        # only here if input is constant
        tvec = [0, tper]
        yvec = [y0, y0]
    elif tvec[0] > 0:
        # make time start at 0
        tvec.insert(0, 0)
        yvec.insert(0, y0)

    return tvec, yvec


def get_crossing_index(yvec, threshold, n=0, rising=True):
    # type: (np.array, float, int, bool) -> int
    """Returns the first index that the given numpy array crosses the given threshold.

    Parameters
    ----------
    yvec : np.array
        the numpy array.
    threshold : float
        the crossing threshold.
    n : int
        returns the nth edge index, with n=0 being the first index.
    rising : bool
        True to return rising edge index.  False to return falling edge index.

    Returns
    -------
    idx : int
        the crossing edge index.
    """

    bool_vec = yvec >= threshold
    qvec = bool_vec.astype(int)
    dvec = np.diff(qvec)

    dvec = np.maximum(dvec, 0) if rising else np.minimum(dvec, 0)
    idx_list = dvec.nonzero()[0]
    return idx_list[n]


def get_flop_timing(tvec, d, q, clk, ttol, data_thres=0.5,
                    clk_thres=0.5, tstart=0.0, clk_edge='rising', tag=None, invert=False):
    """Calculate flop timing parameters given the associated waveforms.

    This function performs the following steps:

    1. find all valid clock edges.  Compute period of the clock (clock waveform
       must be periodic).
    
    2. For each valid clock edge:

        A. Check if the input changes in the previous cycle.  If so, compute tsetup.
           Otherwise, tsetup = tperiod.
    
        B. Check if input changes in the current cycle.  If so, compute thold.
           Otherwise, thold = tperiod.
  
        C. Check that output transition at most once and that output = input.
           Otherwise, record an error.

        D. record the output data polarity.

    3. For each output data polarity, compute the minimum tsetup and thold and any
       errors.  Return summary as a dictionary.

    
    The output is a dictionary with keys 'setup', 'hold', 'delay', and 'errors'.
    the setup/hold/delay entries contains 2-element tuples describing the worst
    setup/hold/delay time.  The first element is the setup/hold/delay time, and
    the second element is the clock edge time at which it occurs.  The errors field
    stores all clock edge times at which an error occurs.


    Parameters
    ----------
    tvec : np.ndarray
        the time data.
    d : np.ndarray
        the input data.
    q : np.ndarray
        the output data.
    clk : np.ndarray
        the clock data.
    ttol : float
        time resolution.
    data_thres : float
        the data threshold.
    clk_thres : float
        the clock threshold.
    tstart : float
        ignore data points before tstart.
    clk_edge : str
        the clock edge type.  Valid values are "rising", "falling", or "both".
    tag : obj
        an identifier tag to append to results.
    invert : bool
        if True, the flop output is inverted from the data.

    Returns
    -------
    data : dict[str, any]
        A dictionary describing the worst setup/hold/delay and errors, if any.
    """
    d_wv = Waveform(tvec, d, ttol)
    clk_wv = Waveform(tvec, clk, ttol)
    q_wv = Waveform(tvec, q, ttol)
    tend = tvec[-1]

    # get all clock sampling times and clock period
    samp_times = clk_wv.get_all_crossings(clk_thres, start=tstart, edge=clk_edge)
    tper = (samp_times[-1] - samp_times[0]) / (len(samp_times) - 1)
    # ignore last clock cycle if it's not a full cycle.
    if samp_times[-1] + tper > tend:
        samp_times = samp_times[:-1]

    # compute setup/hold/error for each clock period
    data = {'setup': (tper, -1), 'hold': (tper, -1), 'delay': (0.0, -1), 'errors': []}
    for t in samp_times:
        d_prev = d_wv.get_all_crossings(data_thres, start=t - tper, stop=t, edge='both')
        d_cur = d_wv.get_all_crossings(data_thres, start=t, stop=t + tper, edge='both')
        q_cur = q_wv.get_all_crossings(data_thres, start=t, stop=t + tper, edge='both')
        d_val = d_wv(t) > data_thres
        q_val = q_wv(t + tper) > data_thres

        # calculate setup/hold/delay
        tsetup = t - d_prev[-1] if d_prev else tper
        thold = d_cur[0] - t if d_cur else tper
        tdelay = q_cur[0] - t if q_cur else 0.0

        # check if flop has error
        error = (invert != (q_val != d_val)) or (len(q_cur) > 1)

        # record results
        if tsetup < data['setup'][0]:
            data['setup'] = (tsetup, t)
        if thold < data['hold'][0]:
            data['hold'] = (thold, t)
        if tdelay > data['delay'][0]:
            data['delay'] = (tdelay, t)
        if error:
            data['errors'].append(t)

    if tag is not None:
        data['setup'] += (tag, )
        data['hold'] += (tag, )
        data['delay'] += (tag, )
        data['errors'] = [(t, tag) for t in data['errors']]

    return data
