# -*- coding: utf-8 -*-

"""This module defines various layout objects one can add and manipulate in a template.
"""
from typing import TYPE_CHECKING, Union, List, Tuple, Optional, Dict, Any, Iterator, Iterable, \
    Generator

import abc
import numpy as np
from copy import deepcopy

from .util import transform_table, BBox, BBoxArray, transform_point, get_inverse_transform
from .routing.base import Port, WireArray

import bag.io

if TYPE_CHECKING:
    from .template import TemplateBase
    from .routing.grid import RoutingGrid

ldim = Union[float, int]
loc_type = Tuple[ldim, ldim]


class Figure(object, metaclass=abc.ABCMeta):
    """Base class of all layout objects.

    Parameters
    ----------
    resolution : float
        layout unit resolution.
    """

    def __init__(self, resolution):
        # type: (float) -> None
        self._res = resolution
        self._destroyed = False

    @abc.abstractmethod
    def transform(self, loc=(0, 0), orient='R0', unit_mode=False, copy=False):
        # type: (Tuple[ldim, ldim], str, bool, bool) -> Figure
        """Transform this figure."""
        pass

    @abc.abstractmethod
    def move_by(self, dx=0, dy=0, unit_mode=False):
        # type: (ldim, ldim, bool) -> None
        """Move this path by the given amount.

        Parameters
        ----------
        dx : float
            the X shift.
        dy : float
            the Y shift.
        unit_mode : bool
            True if shifts are given in resolution units.
        """
        pass

    @property
    def resolution(self):
        # type: () -> float
        """Retuns the layout unit resolution."""
        return self._res

    @property
    def destroyed(self):
        # type: () -> bool
        """Returns True if this instance is destroyed"""
        return self._destroyed

    @property
    def valid(self):
        # type: () -> bool
        """Returns True if this figure is valid."""
        return not self._destroyed

    def check_destroyed(self):
        # type: () -> None
        """Raises an exception if this object is already destroyed."""
        if self._destroyed:
            raise Exception('This %s is already destroyed.' % self.__class__.__name__)

    def destroy(self):
        # type: () -> None
        """Destroy this instance."""
        self._destroyed = True


# noinspection PyAbstractClass
class Arrayable(Figure, metaclass=abc.ABCMeta):
    """A layout object with arraying support.

    Also handles destroy support.

    Parameters
    ----------
    res : float
        layout unit resolution.
    nx : int
        number of columns.
    ny : int
        number of rows.
    spx : Union[float or int]
        column pitch.
    spy : Union[float or int]
        row pitch.
    unit_mode : bool
        True if spx/spy are specified in resolution units.
    """

    def __init__(self, res, nx=1, ny=1, spx=0, spy=0, unit_mode=False):
        # type: (float, int, int, ldim, ldim, bool) -> None
        Figure.__init__(self, res)
        self._nx = nx
        self._ny = ny
        if unit_mode:
            self._spx_unit = spx
            self._spy_unit = spy
        else:
            self._spx_unit = int(round(spx / res))
            self._spy_unit = int(round(spy / res))

    @property
    def nx(self):
        # type: () -> int
        """Number of columns."""
        return self._nx

    @nx.setter
    def nx(self, val):
        # type: (int) -> None
        """Sets the number of columns."""
        self.check_destroyed()
        if val <= 0:
            raise ValueError('Cannot have non-positive number of columns.')
        self._nx = val

    @property
    def ny(self):
        # type: () -> int
        """Number of rows."""
        return self._ny

    @ny.setter
    def ny(self, val):
        # type: (int) -> None
        """Sets the number of rows."""
        self.check_destroyed()
        if val <= 0:
            raise ValueError('Cannot have non-positive number of rows.')
        self._ny = val

    @property
    def spx(self):
        # type: () -> float
        """The column pitch."""
        return self._spx_unit * self.resolution

    @spx.setter
    def spx(self, val):
        # type: (float) -> None
        """Sets the new column pitch."""
        self.check_destroyed()
        if val < 0:
            raise ValueError('Currently does not support negative pitches.')
        self._spx_unit = int(round(val / self.resolution))

    @property
    def spx_unit(self):
        # type: () -> int
        """The column pitch in resolution units."""
        return self._spx_unit

    @spx_unit.setter
    def spx_unit(self, val):
        # type: (int) -> None
        """Sets the new column pitch in resolution units."""
        self.check_destroyed()
        if val < 0:
            raise ValueError('Currently does not support negative pitches.')
        self._spx_unit = val

    @property
    def spy(self):
        # type: () -> float
        """The row pitch."""
        return self._spy_unit * self.resolution

    @spy.setter
    def spy(self, val):
        # type: (float) -> None
        """Sets the new row pitch."""
        self.check_destroyed()
        if val < 0:
            raise ValueError('Currently does not support negative pitches.')
        self._spy_unit = int(round(val / self.resolution))

    @property
    def spy_unit(self):
        # type: () -> int
        """The row pitch in resolution units."""
        return self._spy_unit

    @spy_unit.setter
    def spy_unit(self, val):
        # type: (int) -> None
        """Sets the new row pitch in resolution units."""
        self.check_destroyed()
        if val < 0:
            raise ValueError('Currently does not support negative pitches.')
        self._spy_unit = val

    @Figure.valid.getter
    def valid(self):
        # type: () -> bool
        """Returns True if this instance is valid, i.e. not destroyed and nx, ny >= 1."""
        return not self.destroyed and self.nx >= 1 and self.ny >= 1

    def get_item_location(self, row=0, col=0, unit_mode=False):
        # type: (int, int, bool) -> Tuple[ldim, ldim]
        """Returns the location of the given item in the array.

        Parameters
        ----------
        row : int
            the item row index.  0 is the bottom-most row.
        col : int
            the item column index.  0 is the left-most column.
        unit_mode : bool
            True to return coordinates in resolution units

        Returns
        -------
        xo : Union[float, int]
            the item X coordinate.
        yo : Union[float, int]
            the item Y coordinate.
        """
        if row < 0 or row >= self.ny or col < 0 or col >= self.nx:
            raise ValueError('Invalid row/col index: row=%d, col=%d' % (row, col))

        xo = col * self._spx_unit
        yo = row * self._spy_unit
        if unit_mode:
            return xo, yo
        return xo * self.resolution, yo * self.resolution


class InstanceInfo(dict):
    """A dictionary that represents a layout instance.
    """

    param_list = ['lib', 'cell', 'view', 'name', 'loc', 'orient', 'num_rows',
                  'num_cols', 'sp_rows', 'sp_cols', 'master_key']

    def __init__(self, res, change_orient=True, **kwargs):
        kv_iter = ((key, kwargs.get(key, None)) for key in self.param_list)
        dict.__init__(self, kv_iter)
        self._resolution = res
        if 'params' in kwargs:
            self.params = kwargs['params']

        # skill/OA array before rotation, while we're doing the opposite.
        # this is supposed to fix it.
        if change_orient:
            orient = self['orient']
            if orient == 'R180':
                self['sp_rows'] *= -1
                self['sp_cols'] *= -1
            elif orient == 'MX':
                self['sp_rows'] *= -1
            elif orient == 'MY':
                self['sp_cols'] *= -1
            elif orient == 'R90':
                self['sp_rows'], self['sp_cols'] = self['sp_cols'], -self['sp_rows']
                self['num_rows'], self['num_cols'] = self['num_cols'], self['num_rows']
            elif orient == 'MXR90':
                self['sp_rows'], self['sp_cols'] = self['sp_cols'], self['sp_rows']
                self['num_rows'], self['num_cols'] = self['num_cols'], self['num_rows']
            elif orient == 'MYR90':
                self['sp_rows'], self['sp_cols'] = -self['sp_cols'], -self['sp_rows']
                self['num_rows'], self['num_cols'] = self['num_cols'], self['num_rows']
            elif orient == 'R270':
                self['sp_rows'], self['sp_cols'] = -self['sp_cols'], self['sp_rows']
                self['num_rows'], self['num_cols'] = self['num_cols'], self['num_rows']
            elif orient != 'R0':
                raise ValueError('Unknown orientation: %s' % orient)

    @property
    def lib(self):
        # type: () -> str
        return self['lib']

    @property
    def cell(self):
        # type: () -> str
        return self['cell']

    @property
    def view(self):
        # type: () -> str
        return self['view']

    @property
    def name(self):
        # type: () -> str
        return self['name']

    @name.setter
    def name(self, new_name):
        # type: (str) -> None
        self['name'] = new_name

    @property
    def loc(self):
        # type: () -> Tuple[float, float]
        loc_list = self['loc']
        return loc_list[0], loc_list[1]

    @property
    def orient(self):
        # type: () -> str
        return self['orient']

    @property
    def num_rows(self):
        # type: () -> int
        return self['num_rows']

    @property
    def num_cols(self):
        # type: () -> int
        return self['num_cols']

    @property
    def sp_rows(self):
        # type: () -> float
        return self['sp_rows']

    @property
    def sp_cols(self):
        # type: () -> float
        return self['sp_cols']

    @property
    def params(self):
        # type: () -> Optional[Dict[str, Any]]
        return self.get('params', None)

    @params.setter
    def params(self, new_params):
        # type: (Optional[Dict[str, Any]]) -> None
        self['params'] = new_params

    @property
    def master_key(self):
        return self.get('master_key', None)

    @master_key.setter
    def master_key(self, value):
        self['master_key'] = value

    @property
    def angle_reflect(self):
        # type: () -> Tuple[int, bool]
        orient = self['orient']
        if orient == 'R0':
            return 0, False
        elif orient == 'R180':
            return 180, False
        elif orient == 'MX':
            return 0, True
        elif orient == 'MY':
            return 180, True
        elif orient == 'R90':
            return 90, False
        elif orient == 'MXR90':
            return 90, True
        elif orient == 'MYR90':
            return 270, True
        elif orient == 'R270':
            return 270, False
        else:
            raise ValueError('Unknown orientation: %s' % orient)

    def copy(self):
        """Override copy method of dictionary to return an InstanceInfo instead."""
        return InstanceInfo(self._resolution, change_orient=False, **self)

    def move_by(self, dx=0, dy=0):
        # type: (float, float) -> None
        """Move this instance by the given amount.

        Parameters
        ----------
        dx : float
            the X shift.
        dy : float
            the Y shift.
        """
        res = self._resolution
        loc = self.loc
        self['loc'] = [round((loc[0] + dx) / res) * res,
                       round((loc[1] + dy) / res) * res]


class Instance(Arrayable):
    """A layout instance, with optional arraying parameters.

    Parameters
    ----------
    parent_grid : RoutingGrid
        the parent RoutingGrid object.
    lib_name : str
        the layout library name.
    master : TemplateBase
        the master template of this instance.
    loc : Tuple[Union[float, int], Union[float, int]]
        the origin of this instance.
    orient : str
        the orientation of this instance.
    name : Optional[str]
        name of this instance.
    nx : int
        number of columns.
    ny : int
        number of rows.
    spx : Union[float, int]
        column pitch.
    spy : Union[float, int]
        row pitch.
    unit_mode : bool
        True if layout dimensions are specified in resolution units.
    """

    def __init__(self,
                 parent_grid,  # type: RoutingGrid
                 lib_name,  # type: str
                 master,  # type: TemplateBase
                 loc,  # type: Tuple[ldim, ldim]
                 orient,  # type: str
                 name=None,  # type: Optional[str]
                 nx=1,  # type: int
                 ny=1,  # type: int
                 spx=0,  # type: ldim
                 spy=0,  # type: ldim
                 unit_mode=False,  # type: bool
                 ):
        # type: (...) -> None
        res = parent_grid.resolution
        Arrayable.__init__(self, res, nx=nx, ny=ny, spx=spx, spy=spy, unit_mode=unit_mode)
        self._parent_grid = parent_grid
        self._lib_name = lib_name
        self._inst_name = name
        self._master = master
        if unit_mode:
            self._loc_unit = loc[0], loc[1]
        else:
            self._loc_unit = int(round(loc[0] / res)), int(round(loc[1] / res))
        self._orient = orient

    def new_master_with(self, **kwargs):
        # type: (**Any) -> None
        """Change the master template of this instance.

        This method will get the old master template layout parameters, update
        the parameter values with the given dictionary, then create a new master
        template with those parameters and associate it with this instance.

        Parameters
        ----------
        **kwargs
            a dictionary of new parameter values.
        """
        self._master = self._master.new_template_with(**kwargs)

    def blockage_iter(self, layer_id, test_box, spx=0, spy=0):
        # type: (int, BBox, int, int) -> Generator[BBox, None, None]
        # transform the given BBox to master coordinate
        if self.destroyed:
            return

        base_box = self._master.get_track_bbox(layer_id)
        if not base_box.is_physical():
            return
        base_box = self.translate_master_box(base_box)
        test = test_box.expand(dx=spx, dy=spy, unit_mode=True)

        inst_spx = max(self.spx_unit, 1)
        inst_spy = max(self.spy_unit, 1)
        xl = base_box.left_unit
        yb = base_box.bottom_unit
        xr = base_box.right_unit
        yt = base_box.top_unit
        nx0 = max(0, -(-(test.left_unit - xr) // inst_spx))
        nx1 = min(self.nx - 1, (test.right_unit - xl) // inst_spx)
        ny0 = max(0, -(-(test.bottom_unit - yt) // inst_spy))
        ny1 = min(self.ny - 1, (test.top_unit - yb) // inst_spy)
        orient = self._orient
        x0, y0 = self._loc_unit
        if (orient == 'R90' or orient == 'R270' or
                orient == 'MXR90' or orient == 'MYR90'):
            spx, spy = spy, spx
        for row in range(ny0, ny1 + 1):
            for col in range(nx0, nx1 + 1):
                dx, dy = self.get_item_location(row=row, col=col, unit_mode=True)
                loc = dx + x0, dy + y0
                inv_loc, inv_orient = get_inverse_transform(loc, orient)
                cur_box = test_box.transform(inv_loc, inv_orient, unit_mode=True)
                for box in self._master.blockage_iter(layer_id, cur_box, spx=spx, spy=spy):
                    yield box.transform(loc, orient, unit_mode=True)

    def all_rect_iter(self):
        # type: () -> Generator[Tuple[BBox, int, int], None, None]
        if self.destroyed:
            return

        orient = self._orient
        x0, y0 = self._loc_unit
        flip = (orient == 'R90' or orient == 'R270' or orient == 'MXR90' or orient == 'MYR90')
        for layer_id, box, sdx, sdy in self._master.all_rect_iter():
            if flip:
                sdx, sdy = sdy, sdx
            for row in range(self.ny):
                for col in range(self.nx):
                    dx, dy = self.get_item_location(row=row, col=col, unit_mode=True)
                    loc = dx + x0, dy + y0
                    yield layer_id, box.transform(loc, orient, unit_mode=True), sdx, sdy

    def intersection_rect_iter(self, layer_id, test_box):
        # type: (int, BBox) -> Generator[BBox, None, None]
        if self.destroyed:
            return

        base_box = self._master.get_track_bbox(layer_id)
        if not base_box.is_physical():
            return
        base_box = self.translate_master_box(base_box)

        inst_spx = max(self.spx_unit, 1)
        inst_spy = max(self.spy_unit, 1)
        xl = base_box.left_unit
        yb = base_box.bottom_unit
        xr = base_box.right_unit
        yt = base_box.top_unit
        nx0 = max(0, -(-(test_box.left_unit - xr) // inst_spx))
        nx1 = min(self.nx - 1, (test_box.right_unit - xl) // inst_spx)
        ny0 = max(0, -(-(test_box.bottom_unit - yt) // inst_spy))
        ny1 = min(self.ny - 1, (test_box.top_unit - yb) // inst_spy)
        orient = self._orient
        x0, y0 = self._loc_unit
        for row in range(ny0, ny1 + 1):
            for col in range(nx0, nx1 + 1):
                dx, dy = self.get_item_location(row=row, col=col, unit_mode=True)
                loc = dx + x0, dy + y0
                inv_loc, inv_orient = get_inverse_transform(loc, orient)
                cur_box = test_box.transform(inv_loc, inv_orient, unit_mode=True)
                for box in self._master.intersection_rect_iter(layer_id, cur_box):
                    yield box.transform(loc, orient, unit_mode=True)

    def get_rect_bbox(self, layer):
        """Returns the overall bounding box of all rectangles on the given layer.

        Note: currently this does not check primitive instances or vias.
        """
        bbox = self._master.get_rect_bbox(layer)
        if not bbox.is_valid():
            return bbox
        box_arr = BBoxArray(self.translate_master_box(bbox), nx=self.nx, ny=self.ny,
                            spx=self.spx_unit, spy=self.spy_unit, unit_mode=True)
        return box_arr.get_overall_bbox()

    def track_bbox_iter(self):
        for layer_id, bbox in self._master.track_bbox_iter():
            box_arr = BBoxArray(self.translate_master_box(bbox), nx=self.nx, ny=self.ny,
                                spx=self.spx_unit, spy=self.spy_unit, unit_mode=True)
            yield layer_id, box_arr.get_overall_bbox()

    @property
    def master(self):
        # type: () -> TemplateBase
        """The master template of this instance."""
        return self._master

    @property
    def location(self):
        # type: () -> Tuple[float, float]
        """The instance location."""
        return self._loc_unit[0] * self.resolution, self._loc_unit[1] * self.resolution

    @location.setter
    def location(self, new_loc):
        # type: (Tuple[float, float]) -> None
        """Sets the instance location."""
        self.check_destroyed()
        self._loc_unit = (int(round(new_loc[0] / self.resolution)),
                          int(round(new_loc[1] / self.resolution)))

    @property
    def location_unit(self):
        # type: () -> Tuple[int, int]
        """The instance location."""
        return self._loc_unit

    @location_unit.setter
    def location_unit(self, new_loc):
        # type: (Tuple[int, int]) -> None
        """Sets the instance location."""
        self.check_destroyed()
        self._loc_unit = (new_loc[0], new_loc[1])

    @property
    def orientation(self):
        # type: () -> str
        """The instance orientation"""
        return self._orient

    @orientation.setter
    def orientation(self, val):
        # type: (str) -> None
        """Sets the instance orientation."""
        self.check_destroyed()
        if val not in transform_table:
            raise ValueError('Unsupported orientation: %s' % val)
        self._orient = val

    @property
    def content(self):
        # type: () -> InstanceInfo
        """A dictionary representation of this instance."""
        return InstanceInfo(self.resolution,
                            lib=self._lib_name,
                            cell=self.master.cell_name,
                            view='layout',
                            name=self._inst_name,
                            loc=list(self.location),
                            orient=self.orientation,
                            num_rows=self.ny,
                            num_cols=self.nx,
                            sp_rows=self.spy,
                            sp_cols=self.spx,
                            master_key=self.master.key
                            )

    @property
    def bound_box(self):
        # type: () -> BBox
        """Returns the overall bounding box of this instance."""
        box_arr = BBoxArray(self._master.bound_box, nx=self.nx, ny=self.ny,
                            spx=self._spx_unit, spy=self._spy_unit, unit_mode=True)
        return box_arr.get_overall_bbox().transform(self.location_unit, self.orientation,
                                                    unit_mode=True)

    @property
    def array_box(self):
        # type: () -> BBox
        """Returns the array box of this instance."""
        master_box = getattr(self._master, 'array_box', None)  # type: BBox
        if master_box is None:
            raise ValueError('Master template array box is not defined.')

        box_arr = BBoxArray(master_box, nx=self.nx, ny=self.ny,
                            spx=self._spx_unit, spy=self._spy_unit, unit_mode=True)
        return box_arr.get_overall_bbox().transform(self.location_unit, self.orientation,
                                                    unit_mode=True)

    @property
    def fill_box(self):
        # type: () -> BBox
        """Returns the array box of this instance."""
        master_box = getattr(self._master, 'fill_box', None)  # type: BBox
        if master_box is None:
            raise ValueError('Master template fill box is not defined.')

        box_arr = BBoxArray(master_box, nx=self.nx, ny=self.ny,
                            spx=self._spx_unit, spy=self._spy_unit, unit_mode=True)
        return box_arr.get_overall_bbox().transform(self.location_unit, self.orientation,
                                                    unit_mode=True)

    def get_bound_box_of(self, row=0, col=0):
        """Returns the bounding box of an instance in this mosaic."""
        dx, dy = self.get_item_location(row=row, col=col, unit_mode=True)
        xshift, yshift = self._loc_unit
        xshift += dx
        yshift += dy
        return self._master.bound_box.transform((xshift, yshift), self.orientation, unit_mode=True)

    def move_by(self, dx=0, dy=0, unit_mode=False):
        # type: (Union[float, int], Union[float, int], bool) -> None
        """Move this instance by the given amount.

        Parameters
        ----------
        dx : Union[float, int]
            the X shift.
        dy : Union[float, int]
            the Y shift.
        unit_mode : bool
            True if shifts are given in resolution units
        """
        if not unit_mode:
            dx = int(round(dx / self.resolution))
            dy = int(round(dy / self.resolution))
        self._loc_unit = self._loc_unit[0] + dx, self._loc_unit[1] + dy

    def translate_master_box(self, box):
        # type: (BBox) -> BBox
        """Transform the bounding box in master template.

        Parameters
        ----------
        box : BBox
            the BBox in master template coordinate.

        Returns
        -------
        new_box : BBox
            the cooresponding BBox in instance coordinate.
        """
        return box.transform(self.location_unit, self.orientation, unit_mode=True)

    def translate_master_location(self,
                                  mloc,  # type: Tuple[Union[float, int], Union[float, int]]
                                  unit_mode=False,  # type: bool
                                  ):
        # type: (...) -> Tuple[Union[float, int], Union[float, int]]
        """Returns the actual location of the given point in master template.

        Parameters
        ----------
        mloc : Tuple[Union[float, int], Union[float, int]]
            the location in master coordinate.
        unit_mode : bool
            True if location is given in resolution units.

        Returns
        -------
        xi : Union[float, int]
            the actual X coordinate.  Integer if unit_mode is True.
        yi : Union[float, int]
            the actual Y coordinate.  Integer if unit_mode is True.
        """
        res = self.resolution
        if unit_mode:
            mx, my = mloc[0], mloc[1]
        else:
            mx, my = int(round(mloc[0] / res)), int(round(mloc[1] / res))
        p = transform_point(mx, my, self.location_unit, self.orientation)
        if unit_mode:
            return p[0], p[1]
        return p[0] * res, p[1] * res

    def translate_master_track(self, layer_id, track_idx):
        # type: (int, Union[float, int]) -> Union[float, int]
        """Returns the actual track index of the given track in master template.

        Parameters
        ----------
        layer_id : int
            the layer ID.
        track_idx : Union[float, int]
            the track index.

        Returns
        -------
        new_idx : Union[float, int]
            the new track index.
        """
        dx, dy = self.location_unit
        return self._parent_grid.transform_track(layer_id, track_idx, dx=dx, dy=dy,
                                                 orient=self.orientation, unit_mode=True)

    def get_port(self, name='', row=0, col=0):
        # type: (Optional[str], int, int) -> Port
        """Returns the port object of the given instance in the array.

        Parameters
        ----------
        name : Optional[str]
            the port terminal name.  If None or empty, check if this
            instance has only one port, then return it.
        row : int
            the instance row index.  Index 0 is the bottom-most row.
        col : int
            the instance column index.  Index 0 is the left-most column.

        Returns
        -------
        port : Port
            the port object.
        """
        dx, dy = self.get_item_location(row=row, col=col, unit_mode=True)
        xshift, yshift = self._loc_unit
        loc = (xshift + dx, yshift + dy)
        return self._master.get_port(name).transform(self._parent_grid, loc=loc,
                                                     orient=self.orientation, unit_mode=True)

    def get_pin(self, name='', row=0, col=0, layer=-1):
        # type: (Optional[str], int, int, int) -> Union[WireArray, BBox]
        """Returns the first pin with the given name.

        This is an efficient method if you know this instance has exactly one pin.

        Parameters
        ----------
        name : Optional[str]
            the port terminal name.  If None or empty, check if this
            instance has only one port, then return it.
        row : int
            the instance row index.  Index 0 is the bottom-most row.
        col : int
            the instance column index.  Index 0 is the left-most column.
        layer : int
            the pin layer.  If negative, check to see if the given port has only one layer.
            If so then use that layer.

        Returns
        -------
        pin : Union[WireArray, BBox]
            the first pin associated with the port of given name.
        """
        port = self.get_port(name, row, col)
        return port.get_pins(layer)[0]

    def get_all_port_pins(self, name='', layer=-1):
        # type: (Optional[str], int) -> List[WireArray]
        """Returns a list of all pins of all ports with the given name in this instance array.

        This method gathers ports from all instances in this array with the given name,
        then find all pins of those ports on the given layer, then return as list of WireArrays.

        Parameters
        ----------
        name : Optional[str]
            the port terminal name.  If None or empty, check if this
            instance has only one port, then return it.
        layer : int
            the pin layer.  If negative, check to see if the given port has only one layer.
            If so then use that layer.

        Returns
        -------
        pin_list : List[WireArray]
            the list of pins as WireArrays.
        """
        results = []
        for col in range(self.nx):
            for row in range(self.ny):
                port = self.get_port(name, row, col)
                results.extend(port.get_pins(layer))
        return results

    def port_pins_iter(self, name='', layer=-1):
        # type: (Optional[str], int) -> Iterator[WireArray]
        """Iterate through all pins of all ports with the given name in this instance array.

        Parameters
        ----------
        name : Optional[str]
            the port terminal name.  If None or empty, check if this
            instance has only one port, then return it.
        layer : int
            the pin layer.  If negative, check to see if the given port has only one layer.
            If so then use that layer.

        Yields
        ------
        pin : WireArray
            the pin as WireArray.
        """
        for col in range(self.nx):
            for row in range(self.ny):
                try:
                    port = self.get_port(name, row, col)
                except KeyError:
                    return
                for warr in port.get_pins(layer):
                    yield warr

    def port_names_iter(self):
        # type: () -> Iterable[str]
        """Iterates over port names in this instance.

        Yields
        ------
        port_name : str
            name of a port in this instance.
        """
        return self._master.port_names_iter()

    def has_port(self, port_name):
        # type: (str) -> bool
        """Returns True if this instance has the given port."""
        return self._master.has_port(port_name)

    def has_prim_port(self, port_name):
        # type: (str) -> bool
        """Returns True if this instance has the given primitive port."""
        return self._master.has_prim_port(port_name)

    def transform(self, loc=(0, 0), orient='R0', unit_mode=False, copy=False):
        # type: (Tuple[ldim, ldim], str, bool, bool) -> Optional[Figure]
        """Transform this figure."""
        if not unit_mode:
            res = self.resolution
            loc = int(round(loc[0] / res)), int(round(loc[1] / res))

        if not copy:
            ans = self
        else:
            ans = deepcopy(self)
        ans._loc_unit = loc
        ans._orient = orient
        return ans


class Rect(Arrayable):
    """A layout rectangle, with optional arraying parameters.

    Parameters
    ----------
    layer : string or (string, string)
        the layer name, or a tuple of layer name and purpose name.
        If pupose name not given, defaults to 'drawing'.
    bbox : bag.layout.util.BBox or bag.layout.util.BBoxArray
        the base bounding box.  If this is a BBoxArray, the BBoxArray's
        arraying parameters are used.
    nx : int
        number of columns.
    ny : int
        number of rows.
    spx : float
        column pitch.
    spy : float
        row pitch.
    unit_mode : bool
        True if layout dimensions are specified in resolution units.
    """

    def __init__(self, layer, bbox, nx=1, ny=1, spx=0, spy=0, unit_mode=False):
        # python 2/3 compatibility: convert raw bytes to string.
        layer = bag.io.fix_string(layer)
        if isinstance(layer, str):
            layer = (layer, 'drawing')
        self._layer = layer[0], layer[1]
        if isinstance(bbox, BBoxArray):
            self._bbox = bbox.base
            Arrayable.__init__(self, self._bbox.resolution, nx=bbox.nx, ny=bbox.ny,
                               spx=bbox.spx_unit, spy=bbox.spy_unit, unit_mode=True)
        else:
            self._bbox = bbox
            Arrayable.__init__(self, self._bbox.resolution, nx=nx, ny=ny, spx=spx, spy=spy,
                               unit_mode=unit_mode)

    @property
    def bbox_array(self):
        """The BBoxArray representing this (Arrayed) rectangle.

        Returns
        -------
        barr : :class:`bag.layout.util.BBoxArray`
            the BBoxArray representing this (Arrayed) rectangle.
        """
        return BBoxArray(self._bbox, nx=self.nx, ny=self.ny,
                         spx=self.spx_unit, spy=self.spy_unit, unit_mode=True)

    @property
    def layer(self):
        """The rectangle (layer, purpose) pair."""
        return self._layer

    @layer.setter
    def layer(self, val):
        """Sets the rectangle layer."""
        self.check_destroyed()
        # python 2/3 compatibility: convert raw bytes to string.
        val = bag.io.fix_string(val)
        if isinstance(val, str):
            val = (val, 'drawing')
        self._layer = val[0], val[1]
        print("WARNING: USING THIS BREAKS POWER FILL ALGORITHM.")

    @property
    def bbox(self):
        """The rectangle bounding box."""
        return self._bbox

    @bbox.setter
    def bbox(self, val):
        """Sets the rectangle bounding box."""
        self.check_destroyed()
        if not val.is_physical():
            raise ValueError('Bounding box %s is not physical' % val)
        print("WARNING: USING THIS BREAKS POWER FILL ALGORITHM.")
        self._bbox = val

    @property
    def content(self):
        """A dictionary representation of this rectangle."""
        content = dict(layer=list(self.layer),
                       bbox=[[self.bbox.left, self.bbox.bottom], [self.bbox.right, self.bbox.top]],
                       )
        if self.nx > 1 or self.ny > 1:
            content['arr_nx'] = self.nx
            content['arr_ny'] = self.ny
            content['arr_spx'] = self.spx
            content['arr_spy'] = self.spy

        return content

    def move_by(self, dx=0, dy=0, unit_mode=False):
        """Move the base rectangle by the given amount.

        Parameters
        ----------
        dx : float
            the X shift.
        dy : float
            the Y shift.
        unit_mode : bool
        True if layout dimensions are specified in resolution units.
        """
        print("WARNING: USING THIS BREAKS POWER FILL ALGORITHM.")
        self._bbox = self._bbox.move_by(dx=dx, dy=dy, unit_mode=unit_mode)

    def extend(self, x=None, y=None):
        """extend the base rectangle horizontally or vertically so it overlaps the given X/Y coordinate.

        Parameters
        ----------
        x : float or None
            if not None, make sure the base rectangle overlaps this X coordinate.
        y : float or None
            if not None, make sure the base rectangle overlaps this Y coordinate.
        """
        print("WARNING: USING THIS BREAKS POWER FILL ALGORITHM.")
        self._bbox = self._bbox.extend(x=x, y=y)

    def transform(self, loc=(0, 0), orient='R0', unit_mode=False, copy=False):
        # type: (Tuple[ldim, ldim], str, bool, bool) -> Optional[Figure]
        """Transform this figure."""
        new_box = self._bbox.transform(loc=loc, orient=orient, unit_mode=unit_mode)
        if not copy:
            print("WARNING: USING THIS BREAKS POWER FILL ALGORITHM.")
            ans = self
        else:
            ans = deepcopy(self)

        ans._bbox = new_box
        return ans

    def destroy(self):
        # type: () -> None
        """Destroy this instance."""
        print("WARNING: USING THIS BREAKS POWER FILL ALGORITHM.")
        Arrayable.destroy(self)


class Path(Figure):
    """A layout path.  Only 45/90 degree turns are allowed.

    Parameters
    ----------
    resolution : float
        the layout grid resolution.
    layer : string or (string, string)
        the layer name, or a tuple of layer name and purpose name.
        If purpose name not given, defaults to 'drawing'.
    width : float
        width of this path, in layout units.
    points : List[Tuple[float, float]]
        list of path points.
    end_style : str
        the path ends style.  Currently support 'truncate', 'extend', and 'round'.
    join_style : str
        the ends style at intermediate points of the path.  Currently support 'extend' and 'round'.
    unit_mode : bool
        True if width and points are given as resolution units instead of layout units.
    """

    def __init__(self,
                 resolution,  # type: float
                 layer,  # type: Union[str, Tuple[str, str]]
                 width,  # type: Union[int, float]
                 points,  # type: List[Tuple[Union[int, float], Union[int, float]]]
                 end_style='truncate',  # type: str
                 join_style='extend',  # type: str
                 unit_mode=False,  # type: bool
                 ):
        # type: (...) -> None
        layer = bag.io.fix_string(layer)
        Figure.__init__(self, resolution)
        if isinstance(layer, str):
            layer = (layer, 'drawing')

        self._layer = layer
        self._end_style = end_style
        self._join_style = join_style
        self._destroyed = False
        self._width = 0
        self._points = None
        if not unit_mode:
            self._width = int(round(width / resolution))
            pt_list = self.compress_points(((int(round(x / resolution)), int(round(y / resolution)))
                                            for x, y in points))
        else:
            self._width = width
            pt_list = self.compress_points(points)

        self._points = np.array(pt_list, dtype=int)

    @classmethod
    def compress_points(cls, pts_unit):
        # remove collinear/duplicate points, and make sure all segments are 45 degrees.
        cur_len = 0
        pt_list = []
        for x, y in pts_unit:
            if cur_len == 0:
                pt_list.append((x, y))
                cur_len += 1
            else:
                lastx, lasty = pt_list[-1]
                # make sure we don't have duplicate points
                if x != lastx or y != lasty:
                    dx, dy = x - lastx, y - lasty
                    if dx != 0 and dy != 0 and abs(dx) != abs(dy):
                        # we don't have 45 degree wires
                        raise ValueError('Cannot have line segment (%d, %d)->(%d, %d) in path'
                                         % (lastx, lasty, x, y))
                    if cur_len >= 2:
                        # check for collinearity
                        dx0, dy0 = lastx - pt_list[-2][0], lasty - pt_list[-2][1]
                        if (dx == 0 and dx0 == 0) or (dx != 0 and dx0 != 0 and
                                                      dy / dx == dy0 / dx0):
                            # collinear, remove middle point
                            del pt_list[-1]
                            cur_len -= 1
                    pt_list.append((x, y))
                    cur_len += 1

        return pt_list

    @property
    def layer(self):
        # type: () -> Tuple[str, str]
        """The rectangle (layer, purpose) pair."""
        return self._layer

    @Figure.valid.getter
    def valid(self):
        # type: () -> bool
        """Returns True if this instance is valid."""
        return not self.destroyed and len(self._points) >= 2 and self._width > 0

    @property
    def width(self):
        return self._width * self._res

    @property
    def points(self):
        return [(self._points[idx][0] * self._res, self._points[idx][1] * self._res)
                for idx in range(self._points.shape[0])]

    @property
    def points_unit(self):
        return [(self._points[idx][0], self._points[idx][1])
                for idx in range(self._points.shape[0])]

    @property
    def content(self):
        # type: () -> Dict[str, Any]
        """A dictionary representation of this path."""
        content = dict(layer=list(self.layer),
                       width=self._width * self._res,
                       points=self.points,
                       end_style=self._end_style,
                       join_style=self._join_style,
                       )
        return content

    def move_by(self, dx=0, dy=0, unit_mode=False):
        # type: (ldim, ldim, bool) -> None
        """Move this path by the given amount.

        Parameters
        ----------
        dx : float
            the X shift.
        dy : float
            the Y shift.
        unit_mode : bool
            True if shifts are given in resolution units.
        """
        if not unit_mode:
            dx = int(round(dx / self._res))
            dy = int(round(dy / self._res))
        self._points += np.array([dx, dy])

    def transform(self, loc=(0, 0), orient='R0', unit_mode=False, copy=False):
        # type: (Tuple[ldim, ldim], str, bool, bool) -> Figure
        """Transform this figure."""
        res = self.resolution
        if unit_mode:
            dx, dy = loc
        else:
            dx = int(round(loc[0] / res))
            dy = int(round(loc[1] / res))
        dvec = np.array([dx, dy])
        mat = transform_table[orient]
        new_points = np.dot(mat, self._points.T).T + dvec

        if not copy:
            ans = self
        else:
            ans = deepcopy(self)

        ans._points = new_points
        return ans


class PathCollection(Figure):
    """A layout figure that consists of one or more paths.

    This class make it easy to draw bus/trasmission line objects.

    Parameters
    ----------
    resolution : float
        layout unit resolution.
    paths : List[Path]
        paths in this collection.
    """

    def __init__(self, resolution, paths):
        Figure.__init__(self, resolution)
        self._paths = paths

    def move_by(self, dx=0, dy=0, unit_mode=False):
        # type: (ldim, ldim, bool) -> None
        """Move this path by the given amount.

        Parameters
        ----------
        dx : float
            the X shift.
        dy : float
            the Y shift.
        unit_mode : bool
            True if shifts are given in resolution units.
        """
        for path in self._paths:
            path.move_by(dx=dx, dy=dy, unit_mode=unit_mode)

    def transform(self, loc=(0, 0), orient='R0', unit_mode=False, copy=True):
        # type: (Tuple[ldim, ldim], str, bool, bool) -> PathCollection
        """Transform this figure."""
        if copy:
            ans = deepcopy(self)
        else:
            ans = self

        for p in ans._paths:
            p.transform(loc=loc, orient=orient, unit_mode=unit_mode, copy=False)
        return ans


class TLineBus(PathCollection):
    """A transmission line bus drawn using Path.

    assumes only 45 degree turns are used, and begin and end line segments are straight.

    Parameters
    ----------
    resolution : float
        layout unit resolution.
    layer : Union[str, Tuple[str, str]]
        the bus layer.
    points : List[Tuple[Union[float, int], Union[float, int]]]
        list of center points of the bus.
    widths : List[Union[float, int]]
        list of wire widths.  0 index is left/bottom most wire.
    spaces : List[Union[float, int]]
        list of wire spacings.
    end_style : str
        the path ends style.  Currently support 'truncate', 'extend', and 'round'.
    unit_mode : bool
        True if width and points are given as resolution units instead of layout units.
    """

    def __init__(self, resolution, layer, points, widths, spaces, end_style='truncate',
                 unit_mode=False):
        npoints = len(points)
        if npoints < 2:
            raise ValueError('Must have >= 2 points.')

        if not unit_mode:
            points = ((int(round(px / resolution)), int(round(py / resolution)))
                      for px, py in points)
            widths = [int(round(v / resolution / 2.0)) * 2 for v in widths]
            spaces = [int(round(v / resolution / 2.0)) * 2 for v in spaces]

        points = Path.compress_points(points)

        self._points = np.array(points, dtype=int)
        self._layer = layer
        self._widths = widths
        self._spaces = spaces
        self._end_style = end_style

        tot_width = sum(self._widths) + sum(self._spaces)
        delta_list = [(-tot_width + self._widths[0]) // 2]
        for w0, w1, sp in zip(self._widths, self._widths[1:], self._spaces):
            delta_list.append(delta_list[-1] + sp + ((w0 + w1) // 2))

        print(tot_width)
        print(self._widths)
        print(self._spaces)
        print(delta_list)

        paths = self.create_paths(delta_list, resolution)
        PathCollection.__init__(self, resolution, paths)

    def paths_iter(self):
        return iter(self._paths)

    def create_paths(self, delta_list, res):
        npoints = len(self._points)
        npaths = len(self._widths)
        path_points = [[] for _ in range(npaths)]

        print(self._points)
        # add first point
        p0 = self._points[0, :]
        s0 = self._points[1, :] - p0
        s0 //= np.amax(np.absolute(s0))
        s0_norm = np.linalg.norm(s0)
        d0 = np.array([-s0[1], s0[0]])
        for path, delta in zip(path_points, delta_list):
            tmp = p0 + d0 * int(round(delta / s0_norm))
            path.append((tmp[0], tmp[1]))

        # add intermediate points
        for last_idx in range(2, npoints):
            p1 = self._points[last_idx - 1, :]
            p0 = self._points[last_idx - 2, :]
            s0 = p1 - p0
            s1 = self._points[last_idx, :] - p1
            s0 //= np.amax(np.absolute(s0))
            s1 //= np.amax(np.absolute(s1))
            s0_norm = np.linalg.norm(s0)
            s1_norm = np.linalg.norm(s1)
            dir0 = np.array([-s0[1], s0[0]])
            dir1 = np.array([-s1[1], s1[0]])
            for path, delta in zip(path_points, delta_list):
                d0 = p0 + dir0 * int(round(delta / s0_norm))
                d1 = p1 + dir1 * int(round(delta / s1_norm))
                a = np.array([[-s1[1], s1[0]],
                              [s0[1], s0[0]]], dtype=int) // (s0[1] * s1[0] - s0[0] * s1[1])
                sol = np.dot(a, d1 - d0)
                tmp = sol[0] * s0 + d0
                path.append((tmp[0], tmp[1]))

        # add last points
        p1 = self._points[-1, :]
        s0 = p1 - self._points[-2, :]
        s0 //= np.amax(np.absolute(s0))
        s0_norm = np.linalg.norm(s0)
        d0 = np.array([-s0[1], s0[0]])
        for path, delta in zip(path_points, delta_list):
            tmp = p1 + d0 * int(round(delta / s0_norm))
            path.append((tmp[0], tmp[1]))

        print(path_points)

        paths = [Path(res, self._layer, w, pp, end_style=self._end_style,
                      join_style='round', unit_mode=True)
                 for w, pp in zip(self._widths, path_points)]
        return paths


class Polygon(Figure):
    """A layout polygon object.

    Parameters
    ----------
    resolution : float
        the layout grid resolution.
    layer : Union[str, Tuple[str, str]]
        the layer name, or a tuple of layer name and purpose name.
        If purpose name not given, defaults to 'drawing'.
    points : List[Tuple[Union[float, int], Union[float, int]]]
        the points defining the polygon.
    unit_mode : bool
        True if the points are given in resolution units.
    """

    def __init__(self,
                 resolution,  # type: float
                 layer,  # type: Union[str, Tuple[str, str]]
                 points,  # type: List[Tuple[Union[float, int], Union[float, int]]]
                 unit_mode=False,  # type: bool
                 ):
        # type: (...) -> None
        Figure.__init__(self, resolution)
        layer = bag.io.fix_string(layer)
        if isinstance(layer, str):
            layer = (layer, 'drawing')
        self._layer = layer

        if not unit_mode:
            self._points = np.array(points) / resolution
            self._points = self._points.astype(int)
        else:
            self._points = np.array(points, dtype=int)

    @property
    def layer(self):
        # type: () -> str
        """The blockage layer."""
        return self._layer

    @property
    def points(self):
        return [(self._points[idx][0] * self._res, self._points[idx][1] * self._res)
                for idx in range(self._points.shape[0])]

    @property
    def points_unit(self):
        return [(self._points[idx][0], self._points[idx][1])
                for idx in range(self._points.shape[0])]

    @property
    def content(self):
        # type: () -> Dict[str, Any]
        """A dictionary representation of this blockage."""
        content = dict(layer=self.layer,
                       points=self.points,
                       )
        return content

    def move_by(self, dx=0, dy=0, unit_mode=False):
        # type: (ldim, ldim, bool) -> None
        if not unit_mode:
            dx = int(round(dx / self._res))
            dy = int(round(dy / self._res))
        self._points += np.array([dx, dy])

    def transform(self, loc=(0, 0), orient='R0', unit_mode=False, copy=False):
        # type: (Tuple[ldim, ldim], str, bool, bool) -> Figure
        """Transform this figure."""
        res = self.resolution
        if unit_mode:
            dx, dy = loc
        else:
            dx = int(round(loc[0] / res))
            dy = int(round(loc[1] / res))
        dvec = np.array([dx, dy])
        mat = transform_table[orient]
        new_points = np.dot(mat, self._points.T).T + dvec

        if not copy:
            ans = self
        else:
            ans = deepcopy(self)

        ans._points = new_points
        return ans


class Blockage(Polygon):
    """A blockage object.

    Subclass Polygon for code reuse.

    Parameters
    ----------
    resolution : float
        the layout grid resolution.
    block_type : str
        the blockage type.  Currently supports 'routing' and 'placement'.
    block_layer : str
        the blockage layer.  This value is ignored if blockage type is 'placement'.
    points : List[Tuple[Union[float, int], Union[float, int]]]
        the points defining the blockage.
    unit_mode : bool
        True if the points are given in resolution units.
    """

    def __init__(self, resolution, block_type, block_layer, points, unit_mode=False):
        # type: (float, str, str, List[Tuple[Union[float, int], Union[float, int]]], bool) -> None
        Polygon.__init__(self, resolution, block_layer, points, unit_mode=unit_mode)
        self._type = block_type
        self._block_layer = block_layer

    @property
    def layer(self):
        """The blockage layer."""
        return self._block_layer

    @property
    def type(self):
        # type: () -> str
        """The blockage type."""
        return self._type

    @property
    def content(self):
        # type: () -> Dict[str, Any]
        """A dictionary representation of this blockage."""
        content = dict(layer=self.layer,
                       btype=self.type,
                       points=self.points,
                       )
        return content


class Boundary(Polygon):
    """A boundary object.

    Subclass Polygon for code reuse.

    Parameters
    ----------
    resolution : float
        the layout grid resolution.
    boundary_type : str
        the boundary type.  Currently supports 'PR', 'snap', and 'area'.
    points : List[Tuple[Union[float, int], Union[float, int]]]
        the points defining the blockage.
    unit_mode : bool
        True if the points are given in resolution units.
    """

    def __init__(self, resolution, boundary_type, points, unit_mode=False):
        # type: (float, str, List[Tuple[Union[float, int], Union[float, int]]], bool) -> None
        Polygon.__init__(self, resolution, ('', ''), points, unit_mode=unit_mode)
        self._type = boundary_type

    @property
    def type(self):
        # type: () -> str
        """The blockage type."""
        return self._type

    @property
    def content(self):
        # type: () -> Dict[str, Any]
        """A dictionary representation of this blockage."""
        content = dict(btype=self.type,
                       points=self.points,
                       )
        return content


class ViaInfo(dict):
    """A dictionary that represents a layout via.
    """

    param_list = ['id', 'loc', 'orient', 'num_rows', 'num_cols', 'sp_rows', 'sp_cols',
                  'enc1', 'enc2']

    def __init__(self, res, **kwargs):
        kv_iter = ((key, kwargs[key]) for key in self.param_list)
        dict.__init__(self, kv_iter)
        for opt_par in ['cut_width', 'cut_height', 'arr_nx', 'arr_ny', 'arr_spx', 'arr_spy']:
            if opt_par in kwargs:
                self[opt_par] = kwargs[opt_par]

        self._resolution = res

    @property
    def id(self):
        # type: () -> str
        return self['id']

    @property
    def loc(self):
        # type: () -> Tuple[float, float]
        loc_list = self['loc']
        return loc_list[0], loc_list[1]

    @property
    def orient(self):
        # type: () -> str
        return self['orient']

    @property
    def num_rows(self):
        # type: () -> int
        return self['num_rows']

    @property
    def num_cols(self):
        # type: () -> int
        return self['num_cols']

    @property
    def sp_rows(self):
        # type: () -> float
        return self['sp_rows']

    @property
    def sp_cols(self):
        # type: () -> float
        return self['sp_cols']

    @property
    def enc1(self):
        # type: () -> Tuple[float, float, float, float]
        enc_list = self['enc1']
        return enc_list[0], enc_list[1], enc_list[2], enc_list[3]

    @property
    def enc2(self):
        # type: () -> Tuple[float, float, float, float]
        enc_list = self['enc2']
        return enc_list[0], enc_list[1], enc_list[2], enc_list[3]

    @property
    def cut_width(self):
        # type: () -> float
        return self.get('cut_width', -1)

    @property
    def cut_height(self):
        # type: () -> float
        return self.get('cut_height', -1)

    @property
    def arr_nx(self):
        # type: () -> int
        return self.get('arr_nx', 1)

    @property
    def arr_ny(self):
        # type: () -> int
        return self.get('arr_ny', 1)

    @property
    def arr_spx(self):
        # type: () -> float
        return self.get('arr_spx', 0)

    @property
    def arr_spy(self):
        # type: () -> float
        return self.get('arr_spy', 0)

    def move_by(self, dx=0, dy=0):
        # type: (float, float) -> None
        """Move this instance by the given amount.

        Parameters
        ----------
        dx : float
            the X shift.
        dy : float
            the Y shift.
        """
        res = self._resolution
        loc = self.loc
        self['loc'] = [round((loc[0] + dx) / res) * res,
                       round((loc[1] + dy) / res) * res]


class Via(Arrayable):
    """A layout via, with optional arraying parameters.

    Parameters
    ----------
    tech : bag.layout.core.TechInfo
        the technology class used to calculate via information.
    bbox : bag.layout.util.BBox or bag.layout.util.BBoxArray
        the via bounding box, not including extensions.
        If this is a BBoxArray, the BBoxArray's arraying parameters are used.
    bot_layer : str or (str, str)
        the bottom layer name, or a tuple of layer name and purpose name.
        If purpose name not given, defaults to 'drawing'.
    top_layer : str or (str, str)
        the top layer name, or a tuple of layer name and purpose name.
        If purpose name not given, defaults to 'drawing'.
    bot_dir : str
        the bottom layer extension direction.  Either 'x' or 'y'.
    nx : int
        arraying parameter.  Number of columns.
    ny : int
        arraying parameter.  Mumber of rows.
    spx : float
        arraying parameter.  Column pitch.
    spy : float
        arraying parameter.  Row pitch.
    extend : bool
        True if via extension can be drawn outside of bounding box.
    top_dir : Optional[str]
        top layer extension direction.  Can force to extend in same direction as bottom.
    unit_mode : bool
        True if array pitches are given in resolution units.
    """

    def __init__(self, tech, bbox, bot_layer, top_layer, bot_dir,
                 nx=1, ny=1, spx=0, spy=0, extend=True, top_dir=None, unit_mode=False):
        if isinstance(bbox, BBoxArray):
            self._bbox = bbox.base
            Arrayable.__init__(self, tech.resolution, nx=bbox.nx, ny=bbox.ny,
                               spx=bbox.spx_unit, spy=bbox.spy_unit, unit_mode=True)

        else:
            self._bbox = bbox
            Arrayable.__init__(self, tech.resolution, nx=nx, ny=ny, spx=spx, spy=spy,
                               unit_mode=unit_mode)

        # python 2/3 compatibility: convert raw bytes to string.
        bot_layer = bag.io.fix_string(bot_layer)
        top_layer = bag.io.fix_string(top_layer)

        if isinstance(bot_layer, str):
            bot_layer = (bot_layer, 'drawing')
        if isinstance(top_layer, str):
            top_layer = (top_layer, 'drawing')

        self._tech = tech
        self._bot_layer = bot_layer[0], bot_layer[1]
        self._top_layer = top_layer[0], top_layer[1]
        self._bot_dir = bot_dir
        self._top_dir = top_dir
        self._extend = extend
        self._info = self._tech.get_via_info(self._bbox, bot_layer, top_layer, bot_dir,
                                             top_dir=top_dir, extend=extend)
        if self._info is None:
            raise ValueError('Cannot make via with bounding box %s' % self._bbox)

    def _update(self):
        """Update via parameters."""
        self._info = self._tech.get_via_info(self.bbox, self.bot_layer, self.top_layer,
                                             self.bottom_direction, top_dir=self.top_direction,
                                             extend=self.extend)

    @property
    def top_box(self):
        # type: () -> BBox
        """the top via layer bounding box."""
        return self._info['top_box']

    @property
    def bottom_box(self):
        # type: () -> BBox
        """the bottom via layer bounding box."""
        return self._info['bot_box']

    @property
    def bot_layer(self):
        """The bottom via (layer, purpose) pair."""
        return self._bot_layer

    @property
    def top_layer(self):
        """The top via layer."""
        return self._top_layer

    @property
    def bottom_direction(self):
        """the bottom via extension direction."""
        return self._bot_dir

    @bottom_direction.setter
    def bottom_direction(self, new_bot_dir):
        """Sets the bottom via extension direction."""
        self.check_destroyed()
        self._bot_dir = new_bot_dir
        self._update()

    @property
    def top_direction(self):
        """the bottom via extension direction."""
        if not self._top_dir:
            return 'x' if self._bot_dir == 'y' else 'y'
        return self._top_dir

    @top_direction.setter
    def top_direction(self, new_top_dir):
        """Sets the bottom via extension direction."""
        self.check_destroyed()
        self._top_dir = new_top_dir
        self._update()

    @property
    def extend(self):
        """True if via extension can grow beyond bounding box."""
        return self._extend

    @extend.setter
    def extend(self, new_val):
        self._extend = new_val

    @property
    def bbox(self):
        """The via bounding box not including extensions."""
        return self._bbox

    @property
    def bbox_array(self):
        """The via bounding box array, not including extensions.

        Returns
        -------
        barr : :class:`bag.layout.util.BBoxArray`
            the BBoxArray representing this (Arrayed) rectangle.
        """
        return BBoxArray(self._bbox, nx=self.nx, ny=self.ny, spx=self.spx_unit,
                         spy=self.spy_unit, unit_mode=True)

    @bbox.setter
    def bbox(self, new_bbox):
        """Sets the via bounding box.  Will redraw the via."""
        self.check_destroyed()
        if not new_bbox.is_physical():
            raise ValueError('Bounding box %s is not physical' % new_bbox)
        self._bbox = new_bbox
        self._update()

    @property
    def content(self):
        """A dictionary representation of this via."""
        via_params = self._info['params']
        content = ViaInfo(self._tech.resolution, **via_params)

        if self.nx > 1 or self.ny > 1:
            content['arr_nx'] = self.nx
            content['arr_ny'] = self.ny
            content['arr_spx'] = self.spx
            content['arr_spy'] = self.spy

        return content

    def move_by(self, dx=0, dy=0, unit_mode=False):
        # type: (ldim, ldim, bool) -> None
        """Move this path by the given amount.

        Parameters
        ----------
        dx : float
            the X shift.
        dy : float
            the Y shift.
        unit_mode : bool
            True if shifts are given in resolution units.
        """
        self._bbox = self._bbox.move_by(dx=dx, dy=dy, unit_mode=unit_mode)
        self._info['top_box'] = self._info['top_box'].move_by(dx=dx, dy=dy, unit_mode=unit_mode)
        self._info['bot_box'] = self._info['bot_box'].move_by(dx=dx, dy=dy, unit_mode=unit_mode)
        self._info['params']['loc'] = [self._bbox.xc, self._bbox.yc]

    def transform(self, loc=(0, 0), orient='R0', unit_mode=False, copy=False):
        # type: (Tuple[ldim, ldim], str, bool, bool) -> Figure
        """Transform this figure."""
        new_box = self._bbox.transform(loc=loc, orient=orient, unit_mode=unit_mode)
        if copy:
            return Via(self._tech, new_box, self._bot_layer, self._top_layer, self._bot_dir,
                       nx=self.nx, ny=self.ny, spx=self.spx_unit, spy=self.spy_unit,
                       unit_mode=True)
        else:
            self._bbox = new_box
            self._info['top_box'] = self._info['top_box'].transform(loc=loc, orient=orient,
                                                                    unit_mode=unit_mode)
            self._info['bot_box'] = self._info['bot_box'].transform(loc=loc, orient=orient,
                                                                    unit_mode=unit_mode)
            self._info['params']['loc'] = [self._bbox.xc, self._bbox.yc]


class PinInfo(dict):
    """A dictionary that represents a layout pin.
    """

    param_list = ['net_name', 'pin_name', 'label', 'layer', 'bbox', 'make_rect']

    def __init__(self, res, **kwargs):
        kv_iter = ((key, kwargs[key]) for key in self.param_list)
        dict.__init__(self, kv_iter)

        self._resolution = res

    @property
    def net_name(self):
        # type: () -> str
        return self['net_name']

    @property
    def pin_name(self):
        # type: () -> str
        return self['pin_name']

    @property
    def label(self):
        # type: () -> str
        return self['label']

    @property
    def layer(self):
        # type: () -> Tuple[str, str]
        lay_list = self['layer']
        return lay_list[0], lay_list[1]

    @property
    def bbox(self):
        # type: () -> BBox
        bbox_list = self['bbox']
        return BBox(bbox_list[0][0], bbox_list[0][1], bbox_list[1][0], bbox_list[1][1],
                    self._resolution)

    @property
    def make_rect(self):
        # type: () -> bool
        return self['make_rect']

    def move_by(self, dx=0, dy=0):
        # type: (float, float) -> None
        """Move this instance by the given amount.

        Parameters
        ----------
        dx : float
            the X shift.
        dy : float
            the Y shift.
        """
        new_box = self.bbox.move_by(dx=dx, dy=dy)
        self['bbox'] = [[new_box.left, new_box.bottom], [new_box.right, new_box.top]]
