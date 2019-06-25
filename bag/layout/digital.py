# -*- coding: utf-8 -*-

"""This module defines layout template classes for digital standard cells.
"""

from typing import Dict, Any, Set, Tuple, List, Optional

import abc

import yaml

from bag.util.interval import IntervalSet
from .util import BBox
from .template import TemplateDB, TemplateBase
from .objects import Instance
from .routing import TrackID, WireArray


class StdCellBase(TemplateBase, metaclass=abc.ABCMeta):
    """The base class of all micro templates.

    Parameters
    ----------
    temp_db : TemplateDB
            the template database.
    lib_name : str
        the layout library name.
    params : Dict[str, Any]
        the parameter values.
    used_names : Set[str]
        a set of already used cell names.
    **kwargs
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        with open(params['config_file'], 'r') as f:
            self._config = yaml.load(f)
        self._tech_params = self._config['tech_params']
        self._cells = self._config['cells']
        self._spaces = self._config['spaces']
        self._bound_params = self._config['boundaries']
        TemplateBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)
        self._std_size = None  # type: Optional[Tuple[int, int]]
        self._std_size_bare = None  # type: Optional[Tuple[int, int]]
        self._draw_boundaries = False  # type: bool
        self._used_blocks = []  # type: List[IntervalSet]

    @property
    def min_space_width(self):
        # type: () -> int
        """Returns the minimum space block width in number of standard cell columns."""
        return self._spaces[-1]['num_col']

    @property
    def std_col_width(self):
        # type: () -> float
        """Returns the standard cell column width."""
        return self._tech_params['col_pitch']

    @property
    def std_col_width_unit(self):
        # type: () -> float
        """Returns the standard cell column width in resolution units."""
        res = self.grid.resolution
        return int(round(self._tech_params['col_pitch'] / res))

    @property
    def std_row_height(self):
        # type: () -> float
        """Returns the standard cell row height."""
        return self._tech_params['height']

    @property
    def std_row_height_unit(self):
        # type: () -> float
        """Returns the standard cell row height in resolution units."""
        res = self.grid.resolution
        return int(round(self._tech_params['height'] / res))

    @property
    def std_size(self):
        # type: () -> Optional[Tuple[int, int]]
        """Returns the number of columns/rows that this standard cell occupies."""
        return self._std_size

    @property
    def std_routing_layers(self):
        # type: () -> List[int]
        """Returns the routing layers used by this standard cell."""
        return self._tech_params['layers']

    def get_num_columns(self, layer_id, num_tr):
        # type: (int, int) -> int
        """Returns the number of standard cell columns needed to contain the given amount of tracks.

        Parameters
        ----------
        layer_id : int
            the track layer ID.
        num_tr : int
            number of tracks.

        Returns
        -------
        num_col : int
            number of standard cell columns that span the given number of tracks.
        """
        col_width_unit = int(round(self._tech_params['col_pitch'] / self.grid.resolution))
        tr_pitch = int(self.grid.get_track_pitch(layer_id, unit_mode=True))  # type: int
        return -(-(tr_pitch * num_tr) // col_width_unit)  # ceiling division

    def set_draw_boundaries(self, draw_boundaries):
        # type: (bool) -> None
        """Sets whether this standard cell have boundaries drawn around it.

        To draw boundaries around a standard cell, first call this method
        with draw_boundaries=True, then call set_std_size() method when
        all blocks have been placed.  Finally, call draw_boundaries()
        to draw the bounded cells.

        Parameters
        ----------
        draw_boundaries : bool
            True to draw boundaries around this standard cell.
        """
        self._draw_boundaries = draw_boundaries

    def get_space_blocks(self):
        # type: () -> List[Dict[str, Any]]
        """Returns the space blocks parameters.  Used internally."""
        return self._spaces

    def get_cell_params(self, cell_name):
        # type: (str) -> Dict[str, Any]
        """Returns parameters for the given standard cell.  Used internally.

        Parameters
        ----------
        cell_name : str
            the standard cell name.
        """
        for key, val in self._cells.items():
            if key == cell_name:
                return val
        raise ValueError('Cannot find standard cell with name %s' % cell_name)

    def set_std_size(self, std_size, top_layer=-1):
        # type: (Tuple[int, int], int) -> None
        """Sets the size of this standard cell.

        This method computes self.size, self.array_box, and self.std_size.
        If you will draw boundaries around this standard cell,
        self.set_draw_boundaries(True) should be called first.

        Parameters
        ----------
        std_size : Tuple[int, int]
            the standard cell size as (number of std. columns, number of std. rows) Tuple.
        top_layer : int
            the top level routing layer.  If negative, default to standard cell top routing layer.
        """
        num_col, num_row = std_size
        self._std_size_bare = std_size
        if self._draw_boundaries:
            dx = self._bound_params['lr_width'] * self.std_col_width
            dy = self._bound_params['tb_height'] * self.std_row_height
            self._std_size = (int(std_size[0] + 2 * self._bound_params['lr_width']),
                              int(std_size[1] + 2 * self._bound_params['tb_height']))
        else:
            self._std_size = std_size
            dx, dy = 0, 0
        self.array_box = BBox(0.0, 0.0, num_col * self.std_col_width + 2 * dx,
                              num_row * self.std_row_height + 2 * dy, self.grid.resolution)
        if top_layer < 0:
            top_layer = self.std_routing_layers[-1]

        if self.grid.size_defined(top_layer):
            self.set_size_from_array_box(top_layer)
        else:
            self.prim_top_layer = top_layer
            self.prim_bound_box = self.array_box

    def update_routing_grid(self):
        # type: () -> None
        """Register standard cell routing layers in the RoutingGrid.

        This method must be called first in draw_layout().
        """
        layers = self._tech_params['layers']
        widths = self._tech_params['widths']
        spaces = self._tech_params['spaces']
        directions = self._tech_params['directions']

        self.grid = self.grid.copy()
        for lay_id, w, sp, tdir in zip(layers, widths, spaces, directions):
            self.grid.add_new_layer(lay_id, sp, w, tdir, override=True)
        self.grid.update_block_pitch()

    def get_num_tracks(self, layer_id):
        # type: (int) -> int
        """Get number of tracks in this standard cell.

        Parameters
        ----------
        layer_id : int
            the layer ID.

        Returns
        -------
        num_tracks : int
            number of tracks on the given layer in this standard cell.
        """
        std_size = self.std_size
        if std_size is None:
            raise ValueError("std_size is unset. Try calling set_std_size()?")
        ncol, nrow = std_size

        tdir = self.grid.get_direction(layer_id)
        pitch = int(self.grid.get_track_pitch(layer_id, unit_mode=True))
        if tdir == 'x':
            tot_dim = nrow * int(round(self.std_row_height / self.grid.resolution))
        else:
            tot_dim = ncol * int(round(self.std_col_width / self.grid.resolution))
        return tot_dim // pitch

    def add_std_instance(self, master, inst_name=None, loc=(0, 0), nx=1, ny=1,
                         spx=0, spy=0, flip_lr=False):
        # type: (StdCellBase, Optional[str], Tuple[int, int], int, int, int, int, bool) -> Instance
        """Add a new standard cell instance.

        Parameters
        ----------
        master : StdCellBase
            the standard cell template master to add.
        inst_name : Optional[str]
            the instance name.
        loc : Tuple[int, int]
            lower-left corner of the instance in number of standard cell columns/rows.
        nx : int
            horizontal array count.
        ny : int
            vertical array count.
        spx : int
            horizontal pitch in number of standard cell columns.
        spy : int
            vertical pitch in number of standard cell rows.  Must be even.
        flip_lr : bool
            True to flip the standard cell over Y axis.

        Returns
        -------
        inst : Instance
            the standard cell instance.
        """
        if spy % 2 != 0:
            raise ValueError('row pitch must be even')

        # update self._used_blocks
        master_std_size = master.std_size
        if master_std_size is None:
            raise ValueError("master.std_size is unset. Try calling master.set_std_size()?")
        inst_ncol, inst_nrow = master_std_size
        cur_nrow = loc[1] + inst_nrow + (ny - 1) * spy
        while len(self._used_blocks) < cur_nrow:
            self._used_blocks.append(IntervalSet())
        for col_off in range(nx):
            xoff = col_off * spx + loc[0]
            for row_off in range(ny):
                yoff = row_off * spy + loc[1]
                for std_row_idx in range(yoff, yoff + inst_nrow):
                    success = self._used_blocks[std_row_idx].add((xoff, xoff + inst_ncol))
                    if not success:
                        raise ValueError('Cannot add instance at std loc (%d, %d)' % (xoff, yoff))

        col_pitch = self.std_col_width
        row_pitch = self.std_row_height
        if loc[1] % 2 == 0:
            orient = 'R0'
            dy = loc[1] * row_pitch
        else:
            orient = 'MX'
            dy = (loc[1] + 1) * row_pitch

        dx = loc[0] * col_pitch
        if flip_lr:
            dx += inst_ncol * col_pitch
            if orient == 'R0':
                orient = 'MY'
            else:
                orient = 'R180'

        spx_new = spx * col_pitch
        spy_new = spy * row_pitch
        if self._draw_boundaries:
            dx += self._bound_params['lr_width'] * self.std_col_width
            dy += self._bound_params['tb_height'] * self.std_row_height

        return self.add_instance(master, inst_name=inst_name, loc=(dx, dy),
                                 orient=orient, nx=nx, ny=ny, spx=spx_new, spy=spy_new)

    def draw_boundaries(self):
        # type: () -> None
        """Draw the boundary cells around this standard cell."""
        lib_name = self._bound_params['lib_name']
        suffix = self._bound_params.get('suffix', '')
        std_size_bare = self._std_size_bare
        if std_size_bare is None:
            raise ValueError("std_size_bare is unset. Try calling set_std_size()?")
        num_col, num_row = std_size_bare
        num_row_even = (num_row + 1) // 2
        num_row_odd = num_row - num_row_even
        wcol, hrow = self.std_col_width, self.std_row_height
        dx = self._bound_params['lr_width'] * wcol
        dy = self._bound_params['tb_height'] * hrow

        # add bottom-left
        self.add_instance_primitive(lib_name, 'boundary_bottomleft' + suffix, (0, 0))

        # add left
        self.add_instance_primitive(lib_name, 'boundary_left' + suffix, (0, dy), ny=num_row_even,
                                    spy=hrow * 2)
        if num_row_odd > 0:
            self.add_instance_primitive(lib_name, 'boundary_left' + suffix, (0, dy + 2 * hrow),
                                        orient='MX', ny=num_row_odd, spy=hrow * 2)

        # add top-left
        if num_row % 2 == 1:
            yc = dy + num_row * hrow
            self.add_instance_primitive(lib_name, 'boundary_topleft' + suffix, (0, yc))
        else:
            yc = 2 * dy + num_row * hrow
            self.add_instance_primitive(lib_name, 'boundary_bottomleft' + suffix, (0, yc),
                                        orient='MX')

        # add bottom
        self.add_instance_primitive(lib_name, 'boundary_bottom' + suffix, (dx, 0), nx=num_col,
                                    spx=wcol)

        # add top
        if num_row % 2 == 1:
            self.add_instance_primitive(lib_name, 'boundary_top' + suffix, (dx, yc), nx=num_col,
                                        spx=wcol)
        else:
            self.add_instance_primitive(lib_name, 'boundary_bottom' + suffix, (dx, yc), orient='MX',
                                        nx=num_col, spx=wcol)

        # add bottom right
        xc = dx + num_col * wcol
        self.add_instance_primitive(lib_name, 'boundary_bottomright' + suffix, (xc, 0))

        # add right
        self.add_instance_primitive(lib_name, 'boundary_right' + suffix, (xc, dy), ny=num_row_even,
                                    spy=hrow * 2)
        if num_row_odd > 0:
            self.add_instance_primitive(lib_name, 'boundary_right' + suffix, (xc, dy + 2 * hrow),
                                        orient='MX', ny=num_row_odd, spy=hrow * 2)

        # add top right
        if num_row % 2 == 1:
            self.add_instance_primitive(lib_name, 'boundary_topright' + suffix, (xc, yc))
        else:
            self.add_instance_primitive(lib_name, 'boundary_bottomright' + suffix, (xc, yc),
                                        orient='MX')

    def fill_space(self):
        # type: () -> None
        """Fill all unused blocks with spaces."""
        std_size_bare = self._std_size_bare
        if std_size_bare is None:
            raise ValueError("std_size_bare is unset. Try calling set_std_size()?")
        tot_intv = (0, std_size_bare[0])
        for row_idx, intv_set in enumerate(self._used_blocks):
            for intv in intv_set.get_complement(tot_intv).intervals():
                loc = (intv[0], row_idx)
                num_spaces = intv[1] - intv[0]
                self.add_std_space(loc, num_spaces, update_used_blks=False)

    def add_std_space(self, loc, num_col, update_used_blks=True):
        # type: (Tuple[int, int], int, bool) -> None
        """Add standard cell spaces at the given location.

        Parameters
        ----------
        loc : Tuple[int, int]
            the lower-left corner of the space block.
        num_col : int
            the space block width in number of columns.
        update_used_blks : bool
            True to register space blocks.  This flag is for internal use only.
        """
        if update_used_blks:
            # update self._used_blocks
            while len(self._used_blocks) < loc[1] + 1:
                self._used_blocks.append(IntervalSet())
            success = self._used_blocks[loc[1]].add((loc[0], loc[0] + num_col))
            if not success:
                raise ValueError('Cannot add space at std loc (%d, %d)' % (loc[0], loc[1]))

        col_pitch = self.std_col_width
        xcur = loc[0] * col_pitch
        if loc[1] % 2 == 0:
            orient = 'R0'
            ycur = loc[1] * self.std_row_height
        else:
            orient = 'MX'
            ycur = (loc[1] + 1) * self.std_row_height

        if self._draw_boundaries:
            dx = self._bound_params['lr_width'] * self.std_col_width
            dy = self._bound_params['tb_height'] * self.std_row_height
        else:
            dx = dy = 0

        for blk_params in self.get_space_blocks():
            lib_name = blk_params['lib_name']
            cell_name = blk_params['cell_name']
            blk_col = blk_params['num_col']
            num_blk, num_col = divmod(num_col, blk_col)
            blk_width = blk_col * col_pitch
            if num_blk > 0:
                self.add_instance_primitive(lib_name, cell_name, (xcur + dx, ycur + dy),
                                            orient=orient, nx=num_blk, spx=blk_width)
                xcur += num_blk * blk_width

        if num_col > 0:
            raise ValueError('has %d columns remaining' % num_col)


class StdCellTemplate(StdCellBase):
    """A template wrapper around a standard cell block.

    Parameters
    ----------
    temp_db : TemplateDB
        the template database.
    lib_name : str
        the layout library name.
    params : Dict[str, Any]
        the parameter values.
    used_names : Set[str]
        a set of already used cell names.
    **kwargs :
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        StdCellBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)
        self._sch_params = None

    @property
    def sch_params(self):
        return self._sch_params

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        """Returns a dictionary containing parameter descriptions.

        Override this method to return a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : Dict[str, str]
            dictionary from parameter name to description.
        """
        return dict(
            cell_name='standard cell cell name.',
            config_file='standard cell configuration file name.',
        )

    def get_layout_basename(self):
        return 'stdcell_%s' % self.params['cell_name']

    def compute_unique_key(self):
        cell_params = self.get_cell_params(self.params['cell_name'])
        return 'stdcell_%s_%s' % (cell_params['lib_name'], cell_params['cell_name'])

    def get_sch_master_info(self):
        # type: () -> Tuple[str, str]
        """Returns the schematic master library/cell name tuple."""
        cell_params = self.get_cell_params(self.params['cell_name'])
        return cell_params['lib_name'], cell_params['cell_name']

    def draw_layout(self):
        # type: () -> None

        cell_params = self.get_cell_params(self.params['cell_name'])
        lib_name = cell_params['lib_name']
        cell_name = cell_params['cell_name']
        size = cell_params['size']
        ports = cell_params['ports']

        # update routing grid
        self.update_routing_grid()
        # add instance
        self.add_instance_primitive(lib_name, cell_name, (0, 0))
        # compute size
        self.set_std_size(size)

        # add pins
        res = self.grid.resolution
        for port_name, pin_list in ports.items():
            for pin in pin_list:
                port_lay_id = pin['layer']
                bbox = pin['bbox']
                layer_dir = self.grid.get_direction(port_lay_id)
                if layer_dir == 'x':
                    intv = bbox[1], bbox[3]
                    lower, upper = bbox[0], bbox[2]
                else:
                    intv = bbox[0], bbox[2]
                    lower, upper = bbox[1], bbox[3]
                tr_idx, tr_w = self.grid.interval_to_track(port_lay_id, intv)
                warr = WireArray(TrackID(port_lay_id, tr_idx, width=tr_w), lower, upper,
                                 res=res, unit_mode=False)
                self.add_pin(port_name, warr, show=False)

        # set properties
        self._sch_params = cell_params.get('sch_params', None)
