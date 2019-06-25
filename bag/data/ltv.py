# -*- coding: utf-8 -*-

"""This module defines functions and classes for linear time-varying circuits data post-processing.
"""

import numpy as np
import scipy.interpolate as interp
import scipy.sparse as sparse


def _even_quotient(a, b, tol=1e-6):
    """Returns a / b if it is an integer, -1 if it is not.."""
    num = int(round(a / b))
    if abs(a - b * num) < abs(b * tol):
        return num
    return -1


class LTVImpulseFinite(object):
    r"""A class that computes finite impulse response of a linear time-varying circuit.

    This class computes the time-varying impulse response based on PSS/PAC simulation
    data, and provides several useful query methods.  Your simulation should be set up
    as follows:

    #. Setup PSS as usual.  We will denote system period as tper and fc = 1/tper.

    #. In PAC, set the maxmimum sidebands to m.

    #. In PAC, set the input frequency sweep to be absolute, and sweep from 0 to
       n * fstep in steps of fstep, where fstep = fc / k for some integer k.

       k should be chosen so that the output settles back to 0 after time k * tper.  k
       should also be chosen such that fstep is a nice round frequency.  Otherwise,
       numerical errors may introduce strange results.

       n should be chosen so that n * fstep is sufficiently large compared to system
       bandwidth.

    #. In PAC options, set the freqaxis option to be "in".

    #. After simulation, PAC should save the output frequency response as a function of
       output harmonic number and input frequency.  Post-process this into a complex 2D
       matrix hmat with shape (2 * m + 1, n + 1), and pass it to this class's constructor.

    Parameters
    ----------
    hmat : np.ndarray
        the PAC simulation data matrix with shape (2 * m + 1, n + 1).
        hmat[a + m, b] is the complex AC gain from input frequency b * fc / k
        to output frequency a * fc + b * fc / k.
    m : int
        number of output sidebands.
    n : int
        number of input frequencies.
    tper : float
        the system period, in seconds.
    k : int
        the ratio between period of the input impulse train and the system period.
        Must be an integer.
    out0 : :class:`numpy.ndarray`
        steady-state output transient waveform with 0 input over 1 period.  This should
        be a two-column array, where the first column is time vector and second column
        is the output.  Used to compute transient response.

    Notes
    -----
    This class uses the algorithm described in [1]_ to compute impulse response from PSS/PAC
    simulation data.  The impulse response :math:`h(t, \tau)` satisfies the following equation:

    .. math:: y(t) = \int_{-\infty}^{\infty} h(t, \tau) \cdot x(\tau)\ d\tau

    Intuitively, :math:`h(t, \tau)` represents the output at time :math:`t` subject to
    an impulse at time :math:`\tau`.  As described in the paper, If :math:`w_c` is the system
    frequency, and :math:`H_m(jw)` is the frequency response of the system at :math:`mw_c + w`
    due to an input sinusoid with frequency :math:`w`, then the impulse response can be calculated as:

    .. math::

        h(t, \tau) = \frac{1}{kT}\sum_{n=-\infty}^{\infty}\sum_{m=-\infty}^{\infty}
        H_m\left (j\dfrac{nw_c}{k}\right) \exp \left[ jmw_ct + j\dfrac{nw_c}{k} (t - \tau)\right]

    where :math:`0 \le \tau < T` and :math:`\tau \le t \le \tau + kT`.

    References
    ----------
    .. [1] J. Kim, B. S. Leibowitz and M. Jeeradit, "Impulse sensitivity function analysis of
       periodic circuits," 2008 IEEE/ACM International Conference on Computer-Aided Design,
       San Jose, CA, 2008, pp. 386-391.

    .. automethod:: __call__
    """
    def __init__(self, hmat, m, n, tper, k, out0):
        hmat = np.asarray(hmat)
        if hmat.shape != (2 * m + 1, n + 1):
            raise ValueError('hmat shape = %s not compatible with M=%d, N=%d' %
                             (hmat.shape, m, n))

        # use symmetry to fill in negative input frequency data.
        fullh = np.empty((2 * m + 1, 2 * n + 1), dtype=complex)
        fullh[:, n:] = hmat / (k * tper)
        fullh[:, :n] = np.fliplr(np.flipud(fullh[:, n + 1:])).conj()

        self.hmat = fullh
        wc = 2.0 * np.pi / tper
        self.m_col = np.arange(-m, m + 1) * (1.0j * wc)
        self.n_col = np.arange(-n, n + 1) * (1.0j * wc / k)
        self.m_col = self.m_col.reshape((-1, 1))
        self.n_col = self.n_col.reshape((-1, 1))
        self.tper = tper
        self.k = k
        self.outfun = interp.interp1d(out0[:, 0], out0[:, 1], bounds_error=True,
                                      assume_sorted=True)

    @staticmethod
    def _print_debug_msg(result):
        res_imag = np.imag(result).flatten()
        res_real = np.real(result).flatten()
        res_ratio = np.abs(res_imag / (res_real + 1e-18))
        idx = np.argmax(res_ratio)
        print('max imag/real ratio: %.4g, imag = %.4g, real = %.4g' %
              (res_ratio[idx], res_imag[idx], res_real[idx]))

    def __call__(self, t, tau, debug=False):
        """Calculate h(t, tau).

        Compute h(t, tau), which is the output at t subject to an impulse
        at time tau. standard numpy broadcasting rules apply.

        Parameters
        ----------
        t : array-like
            the output time.
        tau : array-like
            the input impulse time.
        debug : bool
            True to print debug messages.

        Returns
        -------
        val : :class:`numpy.ndarray`
            the time-varying impulse response evaluated at the given coordinates.
        """
        # broadcast arguments to same shape
        t, tau = np.broadcast_arrays(t, tau)

        # compute impulse using efficient matrix multiply and numpy broadcasting.
        dt = t - tau
        zero_indices = (dt < 0) | (dt > self.k * self.tper)
        t_row = t.reshape((1, -1))
        dt_row = dt.reshape((1, -1))
        tmp = np.dot(self.hmat, np.exp(np.dot(self.n_col, dt_row))) * np.exp(np.dot(self.m_col, t_row))
        result = np.sum(tmp, axis=0).reshape(dt.shape)

        # zero element such that dt < 0 or dt > k * T.
        result[zero_indices] = 0.0

        if debug:
            self._print_debug_msg(result)

        # discard imaginary part
        return np.real(result)

    def _get_core(self, num_points, debug=False):
        """Returns h(dt, tau) matrix and output waveform over 1 period.  Used by lsim.

        Compute h(dt, tau) for 0 <= tau < T and 0 <= dt < kT, where dt = t - tau.
        """
        dt_vec = np.linspace(0.0, self.k * self.tper, self.k * num_points, endpoint=False)  # type: np.ndarray
        tvec_per = dt_vec[:num_points]
        tau_col = tvec_per.reshape((-1, 1))
        dt_row = dt_vec.reshape((1, -1))
        # use matrix multiply to sum across n
        tmp = np.dot(self.hmat, np.exp(np.dot(self.n_col, dt_row)))
        # use broadcast multiply for exp(-jwm*(t-tau)) term
        tmp = tmp * np.exp(np.dot(self.m_col, dt_row))
        # use matrix multiply to sum across m
        result = np.dot(np.exp(np.dot(tau_col, self.m_col.T)), tmp).T

        if debug:
            self._print_debug_msg(result)

        # discard imaginary part
        result = np.real(result)
        # compute output waveform
        wvfm = self.outfun(tvec_per)
        return result, wvfm

    def visualize(self, fig_idx, num_points, num_period,
                  plot_color=True, plot_3d=False, show=True):
        """Visualize the time-varying impulse response.

        Parameters
        ----------
        fig_idx : int
            starting figure index.
        num_points : int
            number of sample points in a period.
        num_period : int
            number of output period.
        plot_color : bool
            True to create a plot of the time-varying impulse response as 2D color plot.
        plot_3d : bool
            True to create a 3D plot of the impulse response.
        show : bool
            True to show the plots immediately.  Set to False if you want to create some
            other plots.
        """
        if not plot_color and not plot_3d:
            # do nothing.
            return
        tot_points = num_period * num_points
        tau_vec = np.linspace(0, self.tper, num_points, endpoint=False)
        dt_vec = np.linspace(0, num_period * self.tper, tot_points, endpoint=False)
        dt, tau = np.meshgrid(dt_vec, tau_vec, indexing='ij', copy=False)
        t = tau + dt

        result, _ = self._get_core(num_points)
        result = result[:num_period * num_points, :]

        import matplotlib.pyplot as plt
        from matplotlib import cm

        if plot_color:
            # plot 2D color
            fig = plt.figure(fig_idx)
            fig_idx += 1
            ax = fig.gca()
            cp = ax.pcolor(t, tau, result, cmap=cm.cubehelix)
            plt.colorbar(cp)
            ax.set_title('Impulse response contours')
            ax.set_ylabel('impulse time')
            ax.set_xlabel('output time')

        if plot_3d:
            # plot 3D impulse response
            # noinspection PyUnresolvedReferences
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(fig_idx)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(t, tau, result, rstride=1, cstride=1, linewidth=0, cmap=cm.cubehelix)
            ax.set_title('Impulse response')
            ax.set_ylabel('impulse time')
            ax.set_xlabel('output time')

        if show:
            plt.show()

    def lsim(self, u, tstep, tstart=0.0, ac_only=False, periodic=False, debug=False):
        r"""Compute the output waveform given input waveform.

        This method assumes zero initial state.  The output waveform will be the
        same length as the input waveform, so pad zeros if necessary.

        Parameters
        ----------
        u : array-like
            the input waveform.
        tstep : float
            the input/output time step, in seconds.  Must evenly divide system period.
        tstart : float
            the time corresponding to u[0].  Assume u = 0 for all time before tstart.
            Defaults to 0.
        ac_only : bool
            Return output waveform due to AC input only and without steady-state
            transient.
        periodic : bool
            True if the input is periodic.  If so, returns steady state output.
        debug : bool
            True to print debug messages.

        Returns
        -------
        y : :class:`numpy.ndarray`
            the output waveform.

        Notes
        -----
        This method computes the integral:

        .. math:: y(t) = \int_{-\infty}^{\infty} h(t, \tau) \cdot x(\tau)\ d\tau

        using the following algorithm:

        #. set :math:`d\tau = \texttt{tstep}`.
        #. Compute :math:`h(\tau + dt, \tau)` for :math:`0 \le dt < kT` and
           :math:`0 \le \tau < T`, then express as a kN-by-N matrix.  This matrix
           completely describes the time-varying impulse response.
        #. tile the impulse response matrix horizontally until its number of columns
           matches input signal length, then multiply column i by u[i].
        #. Compute y as the sum of all anti-diagonals of the matrix computed in
           previous step, multiplied by :math:`d\tau`.  Truncate if necessary.
        """
        u = np.asarray(u)
        nstep = _even_quotient(self.tper, tstep)
        ndelay = _even_quotient(tstart, tstep)

        # error checking
        if len(u.shape) != 1:
            raise ValueError('u must be a 1D array.')
        if nstep < 0:
            raise ValueError('Time step = %.4g does not evenly divide'
                             'System period = %.4g' % (tstep, self.tper))
        if ndelay < 0:
            raise ValueError('Time step = %.4g does not evenly divide'
                             'Startimg time = %.4g' % (tstep, tstart))
        if periodic and nstep != u.size:
            raise ValueError('Periodic waveform must have same period as system period.')

        # calculate and tile hcore
        ntot = u.size
        hcore, outwv = self._get_core(nstep, debug=debug)
        hcore = np.roll(hcore, -ndelay, axis=1)
        outwv = np.roll(outwv, -ndelay)

        if periodic:
            # input periodic; more efficient math.
            hcore *= u
            hcore = np.tile(hcore, (1, self.k + 1))
            y = np.bincount(np.sum(np.indices(hcore.shape), axis=0).flat, hcore.flat)
            y = y[self.k * nstep:(self.k + 1) * nstep] * tstep
        else:
            ntile = int(np.ceil(ntot * 1.0 / nstep))
            hcore = np.tile(hcore, (1, ntile))
            outwv = np.tile(outwv, (ntile,))
            hcore = hcore[:, :ntot]
            outwv = outwv[:ntot]

            # broadcast multiply
            hcore *= u
            # magic code from stackoverflow
            # returns an array of the sums of all anti-diagonals.
            y = np.bincount(np.sum(np.indices(hcore.shape), axis=0).flat, hcore.flat)[:ntot] * tstep

        if not ac_only:
            # add output steady state transient
            y += outwv
        return y

    def lsim_digital(self, tsym, tstep, data, pulse, tstart=0.0, nchain=1, tdelta=0.0, **kwargs):
        """Compute output waveform given input pulse shape and data.

        This method is similar to :func:`~bag.data.ltv.LTVImpulseFinite.lsim`, but
        assumes the input is superposition of shifted and scaled copies of a given
        pulse waveform.  This assumption speeds up the computation and is useful
        for high speed link design.

        Parameters
        ----------
        tsym : float
            the symbol period, in seconds.  Must evenly divide system period.
        tstep : float
            the output time step, in seconds.  Must evenly divide symbol period.
        data : list[float]
            list of symbol values.
        pulse : np.ndarray
            the pulse waveform as a two-column array.  The first column is time,
            second column is pulse waveform value.  Linear interpolation will be used
            if necessary.  Time must start at 0.0 and be increasing.
        tstart : float
            time of the first data symbol.  Defaults to 0.0
        nchain : int
            number of blocks in a chain.  Defaults to 1.  This argument is useful if
            you have multiple blocks cascaded together in a chain, and you wish to find
            the output waveform at the end of the chain.
        tdelta : float
            time difference between adjacent elements in a chain.  Defaults to 0.  This
            argument is useful for simulating a chain of latches, where blocks operate
            on alternate phases of the clock.
        kwargs : dict[str, any]
            additional keyword arguments for :func:`~bag.data.ltv.LTVImpulseFinite.lsim`.

        Returns
        -------
        output : :class:`numpy.ndarray`
            the output waveform over N symbol period, where N is the given data length.
        """
        # check tsym evenly divides system period
        nsym = _even_quotient(self.tper, tsym)
        if nsym < 0:
            raise ValueError('Symbol period %.4g does not evenly divide '
                             'system period %.4g' % (tsym, self.tper))

        # check tstep evenly divides tsym
        nstep = _even_quotient(tsym, tstep)
        if nstep < 0:
            raise ValueError('Time step %.4g does not evenly divide '
                             'symbol period %.4g' % (tstep, tsym))

        # check tstep evenly divides tstart
        ndelay = _even_quotient(tstart, tstep)
        if ndelay < 0:
            raise ValueError('Time step %.4g does not evenly divide '
                             'starting time %.4g' % (tstep, tstart))

        nper = nstep * nsym

        pulse = np.asarray(pulse)
        tvec = pulse[:, 0]
        pvec = pulse[:, 1]

        # find input length
        # noinspection PyUnresolvedReferences
        nlast = min(np.nonzero(pvec)[0][-1] + 1, tvec.size - 1)
        tlast = tvec[nlast]
        ntot = int(np.ceil(tlast / tstep)) + nchain * self.k * nper + nstep * (nsym - 1)

        # interpolate input
        pfun = interp.interp1d(tvec, pvec, kind='linear', copy=False, bounds_error=False,
                               fill_value=0.0, assume_sorted=True)
        tin = np.linspace(0.0, ntot * tstep, ntot, endpoint=False)
        pin = pfun(tin)

        # super-impose pulse responses
        num_out = len(data) * nstep
        output = np.zeros(num_out)
        for idx in range(nsym):
            # get output pulse response
            pout = pin
            for j in range(nchain):
                pout = self.lsim(pout, tstep, tstart=tstart + j * tdelta, periodic=False,
                                 ac_only=True, **kwargs)

            # construct superposition matrix
            cur_data = data[idx::nsym]
            offsets = np.arange(0, len(cur_data) * nper, nper) * -1
            diags = np.tile(cur_data, (ntot, 1)).T
            dia_mat = sparse.dia_matrix((diags, offsets), shape=(num_out, ntot))

            # superimpose
            output += dia_mat.dot(pout)
            # shift input pulse.
            pin = np.roll(pin, nstep)

        # compute output steady state waveform
        out_pss = self.outfun(np.linspace(0.0, self.tper, nper, endpoint=False))
        out_pss = np.roll(out_pss, -ndelay)
        for j in range(1, nchain):
            out_pss = self.lsim(out_pss, tstep, tstart=tstart + j * tdelta, periodic=True,
                                ac_only=False, **kwargs)

        ntile = int(np.ceil(num_out * 1.0 / nper))
        out_pss = np.tile(out_pss, (ntile,))
        output += out_pss[:num_out]

        return output
