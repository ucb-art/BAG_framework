# -*- coding: utf-8 -*-

"""This module defines core data post-processing classes.
"""

import numpy as np
import scipy.interpolate as interp
import scipy.cluster.vq as svq
import scipy.optimize as sciopt


class Waveform(object):
    """A (usually transient) waveform.

    This class provides interpolation and other convenience functions.

    Parameters
    ----------
    xvec : np.multiarray.ndarray
        the X vector.
    yvec : np.multiarray.ndarray
        the Y vector.
    xtol : float
        the X value tolerance.
    order : int
        the interpolation order.  1 for nearest, 2 for linear, 3 for spline.
    ext : int or str
        interpolation extension mode.  See documentation for InterpolatedUnivariateSpline.

    """
    def __init__(self, xvec, yvec, xtol, order=3, ext=3):
        self._xvec = xvec
        self._yvec = yvec
        self._xtol = xtol
        self._order = order
        self._ext = ext
        self._fun = interp.InterpolatedUnivariateSpline(xvec, yvec, k=order, ext=ext)

    @property
    def xvec(self):
        """the X vector"""
        return self._xvec

    @property
    def yvec(self):
        """the Y vector"""
        return self._yvec

    @property
    def order(self):
        """the interpolation order.  1 for nearest, 2 for linear, 3 for spline."""
        return self._order

    @property
    def xtol(self):
        """the X value tolerance."""
        return self._xtol

    @property
    def ext(self):
        """interpolation extension mode.  See documentation for InterpolatedUnivariateSpline."""
        return self._ext

    def __call__(self, *arg, **kwargs):
        """Evaluate the waveform at the given points."""
        return self._fun(*arg, **kwargs)

    def get_xrange(self):
        """Returns the X vector range.

        Returns
        -------
        xmin : float
            minimum X value.
        xmax : float
            maximum X value.
        """
        return self.xvec[0], self.xvec[-1]

    def shift_by(self, xshift):
        """Returns a shifted version of this waveform.

        Parameters
        ----------
        xshift : float
            the amount to shift by.

        Returns
        -------
        wvfm : bag.data.core.Waveform
            a reference to this instance, or a copy if copy is True.
        """
        return Waveform(self.xvec + xshift, self.yvec, self.xtol, order=self.order, ext=self.ext)

    def get_all_crossings(self, threshold, start=None, stop=None, edge='both'):
        """Returns all X values at which this waveform crosses the given threshold.

        Parameters
        ----------
        threshold : float
            the threshold value.
        start : float or None
            if given, search for crossings starting at this X value.
        stop : float or None
            if given, search only for crossings before this X value.
        edge : string
            crossing type.  Valid values are 'rising', 'falling', or 'both'.

        Returns
        -------
        xval_list : list[float]
            all X values at which crossing occurs.
        """
        # determine start and stop indices
        sidx = 0 if start is None else np.searchsorted(self.xvec, [start])[0]
        if stop is None:
            eidx = len(self.xvec)
        else:
            eidx = np.searchsorted(self.xvec, [stop])[0]
            if eidx < len(self.xvec) and abs(self.xvec[eidx] - stop) < self.xtol:
                eidx += 1

        # quantize waveform values, then detect edge.
        bool_vec = self.yvec[sidx:eidx] >= threshold  # type: np.ndarray
        qvec = bool_vec.astype(int)
        dvec = np.diff(qvec)

        # eliminate unwanted edge types.
        if edge == 'rising':
            dvec = np.maximum(dvec, 0)
        elif edge == 'falling':
            dvec = np.minimum(dvec, 0)

        # get crossing indices
        idx_list = dvec.nonzero()[0]

        # convert indices to X value using brentq interpolation.
        def crossing_fun(x):
            return self._fun(x) - threshold

        xval_list = []
        for idx in idx_list:
            t0, t1 = self.xvec[sidx + idx], self.xvec[sidx + idx + 1]
            try:
                tcross = sciopt.brentq(crossing_fun, t0, t1, xtol=self.xtol)
            except ValueError:
                # no solution, this happens only if we have numerical error
                # around the threshold.  In this case just pick the endpoint
                # closest to threshold.
                va = crossing_fun(t0)
                vb = crossing_fun(t1)
                tcross = t0 if abs(va) < abs(vb) else t1

            xval_list.append(tcross)

        return xval_list

    def get_crossing(self, threshold, start=None, stop=None, n=1, edge='both'):
        """Returns the X value at which this waveform crosses the given threshold.

        Parameters
        ----------
        threshold : float
            the threshold value.
        start : float or None
            if given, search for the crossing starting at this X value.'
        stop : float or None
            if given, search only for crossings before this X value.
        n : int
            returns the nth crossing.
        edge : str
            crossing type.  Valid values are 'rising', 'falling', or 'both'.

        Returns
        -------
        xval : float or None
            the X value at which the crossing occurs.  None if no crossings are detected.
        """
        xval_list = self.get_all_crossings(threshold, start=start, stop=stop, edge=edge)
        if len(xval_list) < n:
            return None
        return xval_list[n-1]

    def to_arrays(self, xmin=None, xmax=None):
        """Returns the X and Y arrays representing this waveform.

        Parameters
        ----------
        xmin : float or None
            If given, will start from this value.
        xmax : float or None
            If given, will end at this value.

        Returns
        -------
        xvec : np.multiarray.ndarray
            the X array
        yvec : np.multiarray.ndarray
            the Y array
        """
        sidx = 0 if xmin is None else np.searchsorted(self.xvec, [xmin])[0]
        eidx = len(self.xvec) if xmax is None else np.searchsorted(self.xvec, [xmax])[0]

        if eidx < len(self.xvec) and self.xvec[eidx] == xmax:
            eidx += 1

        xtemp = self.xvec[sidx:eidx]
        if xmin is not None and (len(xtemp) == 0 or xtemp[0] != xmin):
            np.insert(xtemp, 0, [xmin])
        if xmax is not None and (len(xtemp) == 0 or xtemp[-1] != xmax):
            np.append(xtemp, [xmax])
        return xtemp, self(xtemp)

    def get_eye_specs(self, tbit, tsample, thres=0.0, nlev=2):
        """Compute the eye diagram spec of this waveform.

        This algorithm uses the following steps.

        1. set t_off to 0
        2. sample the waveform at tbit interval, starting at t0 + t_off.
        3. sort the sampled values, get gap between adjacent values.
        4. record G, the length of the gap covering thres.
        5. increment t_off by tsample, go to step 2 and repeat until
           t_off >= tbit.
        6. find t_off with maximum G.  This is the eye center.
        7. at the eye center, compute eye height and eye opening using kmeans
           clustering algorithm.
        8. return result.

        Parameters
        ----------
        tbit : float
            eye period.
        tsample : float
            the resolution to sample the eye.  Used to find optimal
            time shift and maximum eye opening.
        thres : float
            the eye vertical threshold.
        nlev : int
            number of expected levels.  2 for NRZ, 4 for PAM4.

        Returns
        -------
        result : dict
            A dictionary from specification to value.
        """

        tstart, tend = self.get_xrange()
        toff_vec = np.arange(0, tbit, tsample)
        best_idx = 0
        best_gap = 0.0
        best_values = None
        mid_lev = nlev // 2
        for idx, t_off in enumerate(toff_vec):
            # noinspection PyTypeChecker
            values = self(np.arange(tstart + t_off, tend, tbit))
            values.sort()

            up_idx = np.searchsorted(values, [thres])[0]
            if up_idx == 0 or up_idx == len(values):
                continue
            cur_gap = values[up_idx] - values[up_idx - 1]
            if cur_gap > best_gap:
                best_idx = idx
                best_gap = cur_gap
                best_values = values

        if best_values is None:
            raise ValueError("waveform never cross threshold=%.4g" % thres)

        vstd = np.std(best_values)
        vtemp = best_values / vstd
        tmp_arr = np.linspace(vtemp[0], vtemp[-1], nlev)  # type: np.ndarray
        clusters = svq.kmeans(vtemp, tmp_arr)[0]
        # clusters = svq.kmeans(vtemp, 4, iter=50)[0]
        clusters *= vstd
        clusters.sort()
        vcenter = (clusters[mid_lev] + clusters[mid_lev - 1]) / 2.0

        # compute eye opening/margin
        openings = []
        tr_widths = []
        last_val = best_values[0]
        bot_val = last_val
        cur_cidx = 0
        for cur_val in best_values:
            cur_cluster = clusters[cur_cidx]
            next_cluster = clusters[cur_cidx + 1]
            if abs(cur_val - cur_cluster) > abs(cur_val - next_cluster):
                openings.append(cur_val - last_val)
                tr_widths.append(last_val - bot_val)
                cur_cidx += 1
                if cur_cidx == len(clusters) - 1:
                    tr_widths.append(best_values[-1] - cur_val)
                    break
                bot_val = cur_val
            last_val = cur_val

        return {'center': (float(toff_vec[best_idx]), vcenter),
                'levels': clusters,
                'heights': clusters[1:] - clusters[:-1],
                'openings': np.array(openings),
                'trace_widths': np.array(tr_widths)
                }

    def _add_xy(self, other):
        if not isinstance(other, Waveform):
            raise ValueError("Trying to add non-Waveform object.")
        xnew = np.concatenate((self.xvec, other.xvec))
        xnew = np.unique(np.around(xnew / self.xtol)) * self.xtol
        # noinspection PyTypeChecker
        y1 = self(xnew)
        y2 = other(xnew)
        return xnew, y1 + y2

    def __add__(self, other):
        if np.isscalar(other):
            return Waveform(np.array(self.xvec), self.yvec + other, self.xtol, order=self.order, ext=self.ext)
        elif isinstance(other, Waveform):
            new_order = max(self.order, other.order)
            xvec, yvec = self._add_xy(other)
            return Waveform(xvec, yvec, self.xtol, order=new_order, ext=self.ext)
        else:
            raise Exception('type %s not supported' % type(other))

    def __neg__(self):
        return Waveform(np.array(self.xvec), -self.yvec, self.xtol, order=self.order, ext=self.ext)

    def __mul__(self, scale):
        if not np.isscalar(scale):
            raise ValueError("Can only multiply by scalar.")
        return Waveform(np.array(self.xvec), scale * self.yvec, self.xtol, order=self.order, ext=self.ext)

    def __rmul__(self, scale):
        return self.__mul__(scale)
