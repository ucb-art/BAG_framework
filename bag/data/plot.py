# -*- coding: utf-8 -*-

"""This module contains utilities to improve waveform plotting in python.
"""

import numpy as np
import scipy.interpolate as interp

from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.text import Annotation
import matplotlib.pyplot as plt

from ..math import float_to_si_string

# Vega category10 palette
color_cycle = ['#1f77b4', '#ff7f0e',
               '#2ca02c', '#d62728',
               '#9467bd', '#8c564b',
               '#e377c2', '#7f7f7f',
               '#bcbd22', '#17becf',
               ]


def figure(fig_id, picker=5.0):
    """Create a WaveformPlotter.

    Parameters
    ----------
    fig_id : int
        the figure ID.
    picker : float
        picker event pixel tolerance.

    Returns
    -------
    plotter : bag.data.plot.WaveformPlotter
        a plotter that helps you make interactive matplotlib figures.
    """
    return WaveformPlotter(fig_id, picker=picker)


def plot_waveforms(xvec, panel_list, fig=1):
    """Plot waveforms in vertical panels with shared X axis.

    Parameters
    ----------
    xvec : :class:`numpy.ndarray`
        the X data.
    panel_list : list[list[(str, :class:`numpy.ndarray`)]]
        list of lists of Y data.  Each sub-list is one panel.  Each element of the sub-list
        is a tuple of signal name and signal data.
    fig : int
        the figure ID.
    """
    nrow = len(panel_list)

    if nrow > 0:
        myfig = plt.figure(fig, FigureClass=MarkerFigure)  # type: MarkerFigure
        ax0 = None
        for idx, panel in enumerate(panel_list):
            if ax0 is None:
                ax = plt.subplot(nrow, 1, idx + 1)
                ax0 = ax
            else:
                ax = plt.subplot(nrow, 1, idx + 1, sharex=ax0)

            for name, sig in panel:
                ax.plot(xvec, sig, label=name, picker=5.0)

            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        myfig.setup_callbacks()
        plt.show(block=False)


def _fpart(x):
    return x - int(x)


def _rfpart(x):
    return 1 - _fpart(x)


def draw_line(x0, y0, x1, y1, xmax, grid):
    """Draws an anti-aliased line in img from p1 to p2 with the given color."""

    if x0 > x1:
        # x1 is wrapped around
        x1 += xmax

    dx, dy = x1 - x0, y1 - y0
    steep = dx < abs(dy)
    if steep:
        x0, y0, x1, y1, dx, dy = y0, x0, y1, x1, dy, dx

    gradient = dy * 1.0 / dx
    # handle first endpoint
    xpxl1 = int(x0 + 0.5)
    yend = y0 + gradient * (xpxl1 - x0)
    xgap = _rfpart(x0 + 0.5)
    ypxl1 = int(yend)
    if steep:
        grid[ypxl1 % xmax, xpxl1] += _rfpart(yend) * xgap
        grid[(ypxl1 + 1) % xmax, xpxl1] += _fpart(yend) * xgap
    else:
        grid[xpxl1 % xmax, ypxl1] += _rfpart(yend) * xgap
        grid[xpxl1 % xmax, ypxl1 + 1] += _fpart(yend) * xgap

    intery = yend + gradient  # first y-intersection for the main loop

    # do not color second endpoint to avoid double coloring.
    xpxl2 = int(x1 + 0.5)
    # main loop
    if steep:
        for x in range(xpxl1 + 1, xpxl2):
            xval = int(intery)
            grid[xval % xmax, x] += _rfpart(intery)
            grid[(xval + 1) % xmax, x] += _fpart(intery)
            intery += gradient
    else:
        for x in range(xpxl1 + 1,  xpxl2):
            xval = x % xmax
            grid[xval, int(intery)] += _rfpart(intery)
            grid[xval, int(intery) + 1] += _fpart(intery)
            intery += gradient


def plot_eye_heatmap(fig, tvec, yvec, tper, tstart=None, tend=None, toff=None,
                     tstep=None, vstep=None,
                     cmap=None, vmargin=0.05, interpolation='gaussian',
                     repeat=False):
    """Plot eye diagram heat map.

    Parameters
    ----------
    fig : int
        the figure ID.
    tvec : np.ndarray
        the time data.
    yvec : np.ndarray
        waveform data.
    tper : float
        the eye period.
    tstart : float
        starting time.  Defaults to first point.
    tend : float
        ending time.  Defaults to last point.
    toff : float
        eye offset.  Defaults to 0.
    tstep : float or None
        horizontal bin size.  Defaults to using 200 bins.
    vstep : float or None
        vertical bin size.  Defaults to using 200 bins.
    cmap :
        the colormap used for coloring the heat map.  If None, defaults to cubehelix_r
    vmargin : float
        vertical margin in percentage of maximum/minimum waveform values.  Defaults
        to 5 percent.  This is used so that there some room between top/bottom of
        eye and the plot.
    interpolation : str
        interpolation method.  Defaults to 'gaussian'.  Use 'none' for no interpolation.
    repeat : bool
        True to repeat the eye diagram once to the right.  This is useful if you
        want to look at edge transistions.
    """
    if not toff:
        toff = 0.0
    if tstart is None:
        tstart = tvec[0]
    if tend is None:
        tend = tvec[-1]

    if tstep is None:
        num_h = 200
    else:
        num_h = int(np.ceil(tper / tstep))

    arr_idx = (tstart <= tvec) & (tvec < tend)
    tplot = np.mod((tvec[arr_idx] - toff), tper) / tper * num_h  # type: np.ndarray
    yplot = yvec[arr_idx]

    # get vertical range
    ymin, ymax = np.amin(yplot), np.amax(yplot)
    yrang = (ymax - ymin) * (1 + vmargin)
    ymid = (ymin + ymax) / 2.0
    ymin = ymid - yrang / 2.0
    ymax = ymin + yrang

    if vstep is None:
        num_v = 200
    else:
        num_v = int(np.ceil(yrang / vstep))

    # rescale Y axis
    yplot = (yplot - ymin) / yrang * num_v

    grid = np.zeros((num_h, num_v), dtype=float)
    for idx in range(yplot.size - 1):
        draw_line(tplot[idx], yplot[idx], tplot[idx + 1], yplot[idx + 1], num_h, grid)

    if cmap is None:
        from matplotlib import cm
        # noinspection PyUnresolvedReferences
        cmap = cm.cubehelix_r

    plt.figure(fig)
    grid = grid.T[::-1, :]
    if repeat:
        grid = np.tile(grid, (1, 2))
        tper *= 2.0
    plt.imshow(grid, extent=[0, tper, ymin, ymax], cmap=cmap,
               interpolation=interpolation, aspect='auto')
    cb = plt.colorbar()
    cb.set_label('counts')
    return grid


def plot_eye(fig, tvec, yvec_list, tper, tstart=None, tend=None,
             toff_list=None, name_list=None, alpha=1.0):
    """Plot eye diagram.

    Parameters
    ----------
    fig : int
        the figure ID.
    tvec : np.ndarray
        the time data.
    yvec_list : list[np.ndarray]
        list of waveforms to plot in eye diagram.
    tper : float
        the period.
    tstart : float
        starting time.  Defaults to first point.
    tend : float
        ending time.  Defaults to last point.
    toff_list : list[float]
        offset to apply to each waveform.  Defaults to zeros.
    name_list : list[str] or None
        the name of each waveform.  Defaults to numbers.
    alpha : float
        the transparency of each trace.  Can be used to mimic heatmap.
    """
    if not yvec_list:
        return

    if not name_list:
        name_list = [str(num) for num in range(len(yvec_list))]
    if not toff_list:
        toff_list = [0.0] * len(yvec_list)
    if tstart is None:
        tstart = tvec[0]
    if tend is None:
        tend = tvec[-1]

    # get new tstep that evenly divides tper and new x vector
    tstep_given = (tvec[-1] - tvec[0]) / (tvec.size - 1)
    num_samp = int(round(tper / tstep_given))
    t_plot = np.linspace(0.0, tper, num_samp, endpoint=False)

    # find tstart and tend in number of tper.
    nstart = int(np.floor(tstart / tper))
    nend = int(np.ceil(tend / tper))
    ncycle = nend - nstart
    teye = np.linspace(nstart * tper, nend * tper, num_samp * ncycle, endpoint=False)  # type: np.ndarray
    teye = teye.reshape((ncycle, num_samp))

    myfig = plt.figure(fig, FigureClass=MarkerFigure)  # type: MarkerFigure
    ax = plt.subplot()
    legend_lines = []
    for idx, yvec in enumerate(yvec_list):
        color = color_cycle[idx % len(color_cycle)]
        toff = toff_list[idx]
        # get eye traces
        yfun = interp.interp1d(tvec - toff, yvec, kind='linear', copy=False, bounds_error=False,
                               fill_value=np.nan, assume_sorted=True)
        plot_list = []
        for cycle_idx in range(ncycle):
            plot_list.append(t_plot)
            plot_list.append(yfun(teye[cycle_idx, :]))

        lines = ax.plot(*plot_list, alpha=alpha, color=color, picker=4.0, linewidth=2)
        legend_lines.append(lines[0])

    # Put a legend to the right of the current axis
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    ax.legend(legend_lines, name_list, loc='center left', bbox_to_anchor=(1, 0.5))

    myfig.setup_callbacks()
    plt.show(block=False)


def _find_closest_point(x, y, xvec, yvec, xnorm, ynorm):
    """Find point on PWL waveform described by xvec, yvec closest to (x, y)"""
    xnvec = xvec / xnorm
    ynvec = yvec / ynorm
    xn = x / xnorm
    yn = y / ynorm

    dx = np.diff(xnvec)
    dy = np.diff(ynvec)
    px = (xn - xnvec[:-1])
    py = (yn - ynvec[:-1])

    that = (px * dx + py * dy) / (dx ** 2 + dy ** 2)
    t = np.minimum(np.maximum(that, 0), 1)

    minx = xnvec[:-1] + t * dx
    miny = ynvec[:-1] + t * dy

    dist = (minx - xn) ** 2 + (miny - yn) ** 2
    idx = np.argmin(dist)
    return minx[idx] * xnorm, miny[idx] * ynorm


class WaveformPlotter(object):
    """A custom matplotlib interactive plotting class.

    This class adds many useful features, such as ability to add/remove markers,
    ability to toggle waveforms on and off, and so on.

    Parameters
    ----------
    fig_idx : int
        the figure index.
    picker : float
        picker event pixel tolerance.
    normal_width : float
        normal linewidth.
    select_width : float
        selected linewidth.
    """

    def __init__(self, fig_idx, picker=5.0, normal_width=1.5, select_width=3.0):
        self.figure = plt.figure(fig_idx, FigureClass=MarkerFigure)  # type: MarkerFigure
        self.picker = picker
        self.norm_lw = normal_width
        self.top_lw = select_width
        self.ax = self.figure.gca()
        self.ax.set_prop_cycle('color', color_cycle)
        self.leline_lookup = {}
        self.letext_lookup = {}
        self.last_top = None
        self.legend = None
        self.resized_legend = False

    def plot(self, *args, **kwargs):
        if self.figure is None:
            raise ValueError('figure closed already')

        if 'picker' not in kwargs:
            kwargs['picker'] = self.picker
        kwargs['linewidth'] = self.norm_lw
        if 'lw' in kwargs:
            del kwargs['lw']
        return self.ax.plot(*args, **kwargs)

    def setup(self):
        if self.figure is None:
            raise ValueError('figure closed already')

        self.figure.tight_layout()
        # Put a legend to the right of the current axis
        ax_lines, ax_labels = self.ax.get_legend_handles_labels()
        self.legend = self.ax.legend(ax_lines, ax_labels, loc='center left',
                                     bbox_to_anchor=(1, 0.5), fancybox=True)
        le_lines = self.legend.get_lines()
        le_texts = self.legend.get_texts()

        for leline, letext, axline in zip(le_lines, le_texts, ax_lines):
            self.leline_lookup[leline] = (letext, axline)
            self.letext_lookup[letext] = (leline, axline)
            leline.set_picker(self.picker)
            letext.set_picker(self.picker)
            letext.set_alpha(0.5)

        le_texts[-1].set_alpha(1.0)
        ax_lines[-1].set_zorder(2)
        ax_lines[-1].set_linewidth(self.top_lw)
        self.last_top = (le_texts[-1], ax_lines[-1])

        self.figure.register_pick_event(self.leline_lookup, self.legend_line_picked)
        self.figure.register_pick_event(self.letext_lookup, self.legend_text_picked)
        self.figure.setup_callbacks()
        self.figure.canvas.mpl_connect('draw_event', self.fix_legend_location)
        self.figure.canvas.mpl_connect('close_event', self.figure_closed)
        self.figure.canvas.mpl_connect('resize_event', self.figure_resized)

    # noinspection PyUnusedLocal
    def figure_closed(self, event):
        self.figure.close_figure()
        self.figure = None
        self.ax = None
        self.leline_lookup = None
        self.letext_lookup = None
        self.last_top = None
        self.legend = None

    # noinspection PyUnusedLocal
    def figure_resized(self, event):
        self.resized_legend = False
        self.fix_legend_location(None)

    # noinspection PyUnusedLocal
    def fix_legend_location(self, event):
        if not self.resized_legend:
            self.figure.tight_layout()
            inv_tran = self.figure.transFigure.inverted()
            leg_box = inv_tran.transform(self.legend.get_window_extent())
            leg_width = leg_box[1][0] - leg_box[0][0]
            box = self.ax.get_position()
            # print box.x0, box.y0, box.width, box.height, leg_width, leg_frame.get_height()
            self.ax.set_position([box.x0, box.y0, box.width - leg_width, box.height])
            self.resized_legend = True
            self.figure.canvas.draw()

    def legend_line_picked(self, artist):
        letext, axline = self.leline_lookup[artist]
        visible = not axline.get_visible()
        if visible:
            artist.set_alpha(1.0)
        else:
            artist.set_alpha(0.2)
        if visible and (self.last_top[1] is not axline):
            # set to be top line
            self.legend_text_picked(letext, draw=False)
        self.figure.set_line_visibility(axline, visible)

    def legend_text_picked(self, artist, draw=True):
        leline, axline = self.letext_lookup[artist]
        self.last_top[0].set_alpha(0.5)
        self.last_top[1].set_zorder(1)
        self.last_top[1].set_linewidth(self.norm_lw)
        axline.set_zorder(2)
        artist.set_alpha(1.0)
        axline.set_linewidth(self.top_lw)
        self.last_top = (artist, axline)

        # if draw is False, this method is not called from
        # legend_line_picked(), so we'll never have recursion issues.
        if draw:
            if not axline.get_visible():
                # set line to be visible if not
                # draw() will be called in legend_line_picked
                self.legend_line_picked(leline)
            else:
                self.figure.canvas.draw()


# noinspection PyAbstractClass
class MarkerFigure(Figure):
    def __init__(self, **kwargs):
        Figure.__init__(self, **kwargs)
        self.markers = []
        self.epsilon = 10.0
        self.drag_idx = -1
        self.timer = None
        self.marker_line_info = None
        self.pick_sets = []
        self.pick_funs = []

    def set_line_visibility(self, axline, visible):
        axline.set_visible(visible)
        if not visible:
            # delete all markers on this line
            del_idx_list = [idx for idx, item in enumerate(self.markers) if item[2] is axline]
            for targ_idx in reversed(del_idx_list):
                an, pt, _, _ = self.markers[targ_idx]
                del self.markers[targ_idx]
                # print targ_idx, an
                an.set_visible(False)
                pt.set_visible(False)

        self.canvas.draw()

    def register_pick_event(self, artist_set, fun):
        self.pick_sets.append(artist_set)
        self.pick_funs.append(fun)

    def on_button_release(self, event):
        """Disable data cursor dragging. """
        if event.button == 1:
            self.drag_idx = -1

    def on_motion(self, event):
        """Move data cursor around. """
        ax = event.inaxes
        if self.drag_idx >= 0 and ax is not None and event.button == 1:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            anno, pt, line, bg = self.markers[self.drag_idx]
            x, y = _find_closest_point(event.xdata, event.ydata,
                                       line.get_xdata(), line.get_ydata(),
                                       xmax - xmin, ymax - ymin)
            pt.set_data([x], [y])
            xstr, ystr = float_to_si_string(x, 4), float_to_si_string(y, 4)
            anno.set_text('x: %s\ny: %s' % (xstr, ystr))
            anno.xy = (x, y)
            self.canvas.restore_region(bg)
            anno.set_visible(True)
            pt.set_visible(True)
            ax.draw_artist(anno)
            ax.draw_artist(pt)
            self.canvas.blit(ax.bbox)

    def _get_idx_under_point(self, event):
        """Find selected data cursor."""
        mx = event.x
        my = event.y
        mind = None
        minidx = None
        # find closest marker point
        for idx, (an, pt, _, _) in enumerate(self.markers):
            xv, yv = pt.get_xdata()[0], pt.get_ydata()[0]
            xp, yp = event.inaxes.transData.transform([xv, yv])
            # print xv, yv, xp, yp, mx, my
            d = ((mx - xp) ** 2 + (my - yp) ** 2) ** 0.5
            if mind is None or d < mind:
                mind = d
                minidx = idx

        if mind is not None and mind < self.epsilon:
            return minidx
        return -1

    def on_pick(self, event):
        artist = event.artist
        if not artist.get_visible():
            return
        for idx, artist_set in enumerate(self.pick_sets):
            if artist in artist_set:
                self.pick_funs[idx](artist)
                return

        if isinstance(artist, Line2D):
            mevent = event.mouseevent
            # figure out if we picked marker or line
            self.drag_idx = self._get_idx_under_point(mevent)

            if self.drag_idx >= 0:
                # picked marker.
                ax = mevent.inaxes
                an, pt, _, _ = self.markers[self.drag_idx]
                an.set_visible(False)
                pt.set_visible(False)
                self.canvas.draw()
                self.markers[self.drag_idx][-1] = self.canvas.copy_from_bbox(ax.bbox)
                an.set_visible(True)
                pt.set_visible(True)
                ax.draw_artist(an)
                ax.draw_artist(pt)
                self.canvas.blit(ax.bbox)

            else:
                # save data to plot marker later
                mxval = mevent.xdata
                button = mevent.button
                if mxval is not None and button == 1 and not self.marker_line_info:
                    self.marker_line_info = (artist, mxval, mevent.ydata,
                                             button, mevent.inaxes)
        elif isinstance(artist, Annotation):
            # delete marker.
            mevent = event.mouseevent
            if mevent.button == 3:
                targ_idx = None
                for idx, (an, pt, _, _) in enumerate(self.markers):
                    if an is artist:
                        targ_idx = idx
                        break
                if targ_idx is not None:
                    an, pt, _, _ = self.markers[targ_idx]
                    del self.markers[targ_idx]
                    an.set_visible(False)
                    pt.set_visible(False)
                    self.canvas.draw()

    def _create_marker(self):
        if self.marker_line_info:
            artist, mxval, myval, button, ax = self.marker_line_info
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            mxval, myval = _find_closest_point(mxval, myval,
                                               artist.get_xdata(), artist.get_ydata(),
                                               xmax - xmin, ymax - ymin)
            pt = ax.plot(mxval, myval, 'ko', picker=5.0)[0]
            xstr, ystr = float_to_si_string(mxval, 4), float_to_si_string(myval, 4)
            msg = 'x: %s\ny: %s' % (xstr, ystr)
            anno = ax.annotate(msg, xy=(mxval, myval), bbox=dict(boxstyle='round', fc='yellow', alpha=0.3),
                               arrowprops=dict(arrowstyle="->"))
            anno.draggable()
            anno.set_picker(True)

            self.markers.append([anno, pt, artist, None])
            ax.draw_artist(anno)
            ax.draw_artist(pt)
            self.canvas.blit(ax.bbox)
            self.marker_line_info = None

    def close_figure(self):
        self.timer.stop()

    def setup_callbacks(self):
        self.canvas.mpl_connect('pick_event', self.on_pick)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_button_release)
        # use timer to make sure we won't create multiple markers at once when
        # clicked on overlapping lines.
        self.timer = self.canvas.new_timer(interval=100)
        self.timer.add_callback(self._create_marker)
        self.timer.start()
