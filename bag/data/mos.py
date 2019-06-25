# -*- coding: utf-8 -*-

"""This module defines classes for computing DC operating point.
"""

from typing import Dict

import numpy as np


def mos_y_to_ss(sim_data, char_freq, fg, ibias, cfit_method='average'):
    # type: (Dict[str, np.ndarray], float, int, np.ndarray, str) -> Dict[str, np.ndarray]
    """Convert transistor Y parameters to small-signal parameters.

    This function computes MOSFET small signal parameters from 3-port
    Y parameter measurements done on gate, drain and source, with body
    bias fixed.  This functions fits the Y parameter to a capcitor-only
    small signal model using least-mean-square error.

    Parameters
    ----------
    sim_data : Dict[str, np.ndarray]
        A dictionary of Y parameters values stored as complex numpy arrays.
    char_freq : float
        the frequency Y parameters are measured at.
    fg : int
        number of transistor fingers used for the Y parameter measurement.
    ibias : np.ndarray
        the DC bias current of the transistor.  Always positive.
    cfit_method : str
        method used to extract capacitance from Y parameters.  Currently
        supports 'average' or 'worst'

    Returns
    -------
    ss_dict : Dict[str, np.ndarray]
        A dictionary of small signal parameter values stored as numpy
        arrays.  These values are normalized to 1-finger transistor.
    """
    w = 2 * np.pi * char_freq

    gm = (sim_data['y21'].real - sim_data['y31'].real) / 2.0  # type: np.ndarray
    gds = (sim_data['y22'].real - sim_data['y32'].real) / 2.0  # type: np.ndarray
    gb = (sim_data['y33'].real - sim_data['y23'].real) / 2.0 - gm - gds  # type: np.ndarray

    cgd12 = -sim_data['y12'].imag / w
    cgd21 = -sim_data['y21'].imag / w
    cgs13 = -sim_data['y13'].imag / w
    cgs31 = -sim_data['y31'].imag / w
    cds23 = -sim_data['y23'].imag / w
    cds32 = -sim_data['y32'].imag / w
    cgg = sim_data['y11'].imag / w
    cdd = sim_data['y22'].imag / w
    css = sim_data['y33'].imag / w

    if cfit_method == 'average':
        cgd = (cgd12 + cgd21) / 2  # type: np.ndarray
        cgs = (cgs13 + cgs31) / 2  # type: np.ndarray
        cds = (cds23 + cds32) / 2  # type: np.ndarray
    elif cfit_method == 'worst':
        cgd = np.maximum(cgd12, cgd21)
        cgs = np.maximum(cgs13, cgs31)
        cds = np.maximum(cds23, cds32)
    else:
        raise ValueError('Unknown cfit_method = %s' % cfit_method)

    cgb = cgg - cgd - cgs  # type: np.ndarray
    cdb = cdd - cds - cgd  # type: np.ndarray
    csb = css - cgs - cds  # type: np.ndarray

    ibias = ibias / fg
    gm = gm / fg
    gds = gds / fg
    gb = gb / fg
    cgd = cgd / fg
    cgs = cgs / fg
    cds = cds / fg
    cgb = cgb / fg
    cdb = cdb / fg
    csb = csb / fg

    return dict(
        ibias=ibias,
        gm=gm,
        gds=gds,
        gb=gb,
        cgd=cgd,
        cgs=cgs,
        cds=cds,
        cgb=cgb,
        cdb=cdb,
        csb=csb,
    )
