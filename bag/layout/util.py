# -*- coding: utf-8 -*-

"""This module contains utility classes used for layout
"""

from typing import Iterator, Union, Tuple, List, Any

import pprint

import numpy as np

__all__ = ['BBox', 'BBoxArray', 'Pin', 'transform_table', 'transform_point',
           'get_inverse_transform', 'tuple2_to_int', 'tuple2_to_float_int']

transform_table = {'R0': np.array([[1, 0], [0, 1]], dtype=int),
                   'MX': np.array([[1, 0], [0, -1]], dtype=int),
                   'MY': np.array([[-1, 0], [0, 1]], dtype=int),
                   'R180': np.array([[-1, 0], [0, -1]], dtype=int),
                   'R90': np.array([[0, -1], [1, 0]], dtype=int),
                   'MXR90': np.array([[0, 1], [1, 0]], dtype=int),
                   'MYR90': np.array([[0, -1], [-1, 0]], dtype=int),
                   'R270': np.array([[0, 1], [-1, 0]], dtype=int),
                   }


def tuple2_to_int(input_tuple: Tuple[Any, Any]) -> Tuple[int, int]:
    """
    Cast a tuple of 2 elements to a tuple of 2 ints.
    :param input_tuple: Tuple of two elements
    :return: Tuple of two ints
    """
    return int(input_tuple[0]), int(input_tuple[1])


def tuple2_to_float_int(input_tuple: Tuple[Any, Any]) -> Tuple[float, int]:
    """
    Cast a tuple of 2 elements to a tuple of 2 ints.
    :param input_tuple: Tuple of two elements
    :return: Tuple of two ints
    """
    return float(input_tuple[0]), int(input_tuple[1])


def transform_point(x, y, loc, orient):
    """Transform the (x, y) point using the given location and orientation."""
    shift = np.asarray(loc)
    if orient not in transform_table:
        raise ValueError('Unsupported orientation: %s' % orient)

    mat = transform_table[orient]
    ans = np.dot(mat, np.array([x, y])) + shift
    return ans.item(0), ans.item(1)


def get_inverse_transform(loc, orient):
    """Returns the inverse transform"""
    if orient == 'R90':
        orient_inv = 'R270'
    elif orient == 'R270':
        orient_inv = 'R90'
    else:
        orient_inv = orient

    inv_mat = transform_table[orient_inv]
    new_shift = np.dot(inv_mat, -np.asarray(loc))
    return (new_shift.item(0), new_shift.item(1)), orient_inv


def transform_loc_orient(loc, orient, trans_loc, trans_orient):
    """Transforms loc orient with trans_loc and trans_orient"""
    mat1 = transform_table[orient]
    mat2 = transform_table[trans_orient]
    new_mat = np.dot(mat2, mat1)
    new_loc = np.array(trans_loc) + np.dot(mat2, np.array(loc))

    for key, val in transform_table.items():
        if np.allclose(new_mat, val):
            return (new_loc.item(0), new_loc.item(1)), key


class PortSpec(object):
    """Specification of a port.

    Parameters
    ----------
    ntr : int
        number of tracks the port should occupy
    idc : float
        DC current the port should support, in Amperes.
    """

    def __init__(self, ntr, idc):
        self._ntr = ntr
        self._idc = idc

    @property
    def ntr(self):
        """minimum number of tracks the port should occupy"""
        return self._ntr

    @property
    def idc(self):
        """minimum DC current the port should support, in Amperes"""
        return self._idc

    def __str__(self):
        return repr(self)

    def __repr__(self):
        fmt_str = '%s(%d, %.4g)'
        return fmt_str % (self.__class__.__name__, self._ntr, self._idc)


class BBox(object):
    """An immutable bounding box.

    Parameters
    ----------
    left : float or int
        left coordinate.
    bottom : float or int
        bottom coordinate.
    right : float or int
        right coordinate.
    top : float or int
        top coordinate.
    resolution : float
        the coordinate resolution
    unit_mode : bool
        True if the given coordinates are in layout units already.

    """

    def __init__(self, left, bottom, right, top, resolution, unit_mode=False):
        if not unit_mode:
            self._left_unit = int(round(left / resolution))
            self._bot_unit = int(round(bottom / resolution))
            self._right_unit = int(round(right / resolution))
            self._top_unit = int(round(top / resolution))
        else:
            self._left_unit = int(round(left))
            self._bot_unit = int(round(bottom))
            self._right_unit = int(round(right))
            self._top_unit = int(round(top))
            # self._left_unit = left
            # self._bot_unit = bottom
            # self._right_unit = right
            # self._top_unit = top
        self._res = resolution

    @classmethod
    def get_invalid_bbox(cls):
        # type: () -> BBox
        """Returns a default invalid bounding box.

        Returns
        -------
        box : bag.layout.util.BBox
            an invalid bounding box.
        """
        return cls(0, 0, -1, -1, 0.1, unit_mode=True)

    @property
    def left(self):
        """left coordinate."""
        return self._left_unit * self._res

    @property
    def left_unit(self):
        """left coordinate."""
        return self._left_unit

    @property
    def right(self):
        """right coordinate."""
        return self._right_unit * self._res

    @property
    def right_unit(self):
        """right coordinate."""
        return self._right_unit

    @property
    def bottom(self):
        """bottom coordinate."""
        return self._bot_unit * self._res

    @property
    def bottom_unit(self):
        """bottom coordinate."""
        return self._bot_unit

    @property
    def top(self):
        """top coordinate."""
        return self._top_unit * self._res

    @property
    def top_unit(self):
        """top coordinate."""
        return self._top_unit

    @property
    def resolution(self):
        """coordinate resolution."""
        return self._res

    @property
    def width(self):
        """width of this bounding box."""
        return self.width_unit * self._res

    @property
    def width_unit(self):
        """width of this bounding box in resolution units."""
        return self._right_unit - self._left_unit

    @property
    def height(self):
        """height of this bounding box."""
        return self.height_unit * self._res

    @property
    def height_unit(self):
        """height of this bounding box in resolution units."""
        return self._top_unit - self._bot_unit

    @property
    def xc(self):
        """The center X coordinate, rounded to nearest grid point."""
        return ((self._left_unit + self._right_unit) // 2) * self._res

    @property
    def xc_unit(self):
        """The center X coordinate in resolution units."""
        return (self._left_unit + self._right_unit) // 2

    @property
    def yc(self):
        """The center Y coordinate, rounded to nearest grid point."""
        return ((self._bot_unit + self._top_unit) // 2) * self._res

    @property
    def yc_unit(self):
        """The center Y coordinate in resolution units."""
        return (self._bot_unit + self._top_unit) // 2

    def get_points(self, unit_mode=False):
        # type: (bool) -> List[Tuple[Union[float, int], Union[float, int]]]
        """Returns this bounding box as a list of points.

        Parameters
        ----------
        unit_mode : bool
            True to return points in resolution units.

        Returns
        -------
        points : List[Tuple[Union[float, int], Union[float, int]]]
            this bounding box as a list of points.
        """
        if unit_mode:
            return [(self._left_unit, self._bot_unit),
                    (self._left_unit, self._top_unit),
                    (self._right_unit, self._top_unit),
                    (self._right_unit, self._bot_unit)]
        else:
            return [(self.left, self.bottom),
                    (self.left, self.top),
                    (self.right, self.top),
                    (self.right, self.bottom)]

    def as_bbox_array(self):
        """Cast this BBox as a BBoxArray."""
        return BBoxArray(self)

    def as_bbox_collection(self):
        """Cast this BBox as a BBoxCollection."""
        return BBoxCollection([BBoxArray(self)])

    def merge(self, bbox):
        # type: (BBox) -> BBox
        """Returns a new bounding box that's the union of this bounding box and the given one.

        Parameters
        ----------
        bbox : bag.layout.util.BBox
            the bounding box to merge with.

        Returns
        -------
        total : bag.layout.util.BBox
            the merged bounding box.
        """
        if not self.is_valid():
            return bbox
        elif not bbox.is_valid():
            return self

        return BBox(min(self._left_unit, bbox._left_unit),
                    min(self._bot_unit, bbox._bot_unit),
                    max(self._right_unit, bbox._right_unit),
                    max(self._top_unit, bbox._top_unit),
                    self._res, unit_mode=True)

    def intersect(self, bbox):
        # type: (BBox) -> BBox
        """Returns a new bounding box that's the intersection of this bounding box and the given one.

        Parameters
        ----------
        bbox : bag.layout.util.BBox
            the bounding box to intersect with.

        Returns
        -------
        intersect : bag.layout.util.BBox
            the intersection bounding box.
        """
        return BBox(max(self._left_unit, bbox._left_unit),
                    max(self._bot_unit, bbox._bot_unit),
                    min(self._right_unit, bbox._right_unit),
                    min(self._top_unit, bbox._top_unit),
                    self._res, unit_mode=True)

    def overlaps(self, bbox):
        # type: (BBox) -> bool
        """Returns True if this BBox overlaps the given BBox."""

        return ((max(self._left_unit, bbox._left_unit) <
                 min(self._right_unit, bbox._right_unit)) and
                (max(self._bot_unit, bbox._bot_unit) <
                 min(self._top_unit, bbox._top_unit)))

    def extend(self, x=None, y=None, unit_mode=False):
        # type: (Union[float, int], Union[float, int], bool) -> BBox
        """Returns an extended BBox that covers the given point.

        Parameters
        ----------
        x : float or None
            if given, the X coordinate to extend to.
        y : float or None
            if given, the Y coordinate to extend to
        unit_mode : bool
            True if x and y are given in resolution units.

        Returns
        -------
        ext_box : BBox
            the extended bounding box.
        """
        if x is None:
            x = self._left_unit
        elif not unit_mode:
            x = int(round(x / self._res))
        if y is None:
            y = self._bot_unit
        elif not unit_mode:
            y = int(round(y / self._res))

        return BBox(min(self._left_unit, x),
                    min(self._bot_unit, y),
                    max(self._right_unit, x),
                    max(self._top_unit, y), self._res, unit_mode=True)

    def expand(self, dx=0, dy=0, unit_mode=False):
        # type: (Union[float, int], Union[float, int], bool) -> BBox
        """Returns a BBox expanded by the given amount.

        Parameters
        ----------
        dx : Union[float, int]
            if given, expand left and right edge by this amount.
        dy : Union[float, int]
            if given, expand top and bottom edge by this amount.
        unit_mode : bool
            True if x and y are given in resolution units.

        Returns
        -------
        ext_box : BBox
            the extended bounding box.
        """
        if not unit_mode:
            dx = int(round(dx / self._res))
            dy = int(round(dy / self._res))

        return BBox(self._left_unit - dx, self._bot_unit - dy, self._right_unit + dx,
                    self._top_unit + dy, self._res, unit_mode=True)

    def transform(self, loc=(0, 0), orient='R0', unit_mode=False):
        # type: (Tuple[Union[float, int], Union[float, int]], str, bool) -> BBox
        """Returns a new BBox under the given transformation.

        rotates first before shift.

        Parameters
        ----------
        loc : Tuple[Union[float, int], Union[float, int]]
            location of the anchor.
        orient : str
            the orientation of the bounding box.
        unit_mode : bool
            True if location is given in resolution units

        Returns
        -------
        box : BBox
            the new bounding box.
        """
        if not self.is_valid():
            return BBox.get_invalid_bbox()

        if not unit_mode:
            loc = int(round(loc[0] / self._res)), int(round(loc[1] / self._res))

        p1 = transform_point(self._left_unit, self._bot_unit, loc, orient)
        p2 = transform_point(self._right_unit, self._top_unit, loc, orient)
        return BBox(min(p1[0], p2[0]), min(p1[1], p2[1]),
                    max(p1[0], p2[0]), max(p1[1], p2[1]),
                    self._res, unit_mode=True)

    def move_by(self, dx=0, dy=0, unit_mode=False):
        # type: (Union[float, int], Union[float, int], bool) -> BBox
        """Returns a new BBox shifted by the given amount.

        Parameters
        ----------
        dx : float
            shift in X direction.
        dy : float
            shift in Y direction.
        unit_mode : bool
            True if shifts are given in resolution units

        Returns
        -------
        box : bag.layout.util.BBox
            the new bounding box.
        """
        if not unit_mode:
            dx = int(round(dx / self._res))
            dy = int(round(dy / self._res))
        return BBox(self._left_unit + dx, self._bot_unit + dy,
                    self._right_unit + dx, self._top_unit + dy,
                    self._res, unit_mode=True)

    def flip_xy(self):
        # type: () -> BBox
        """Returns a new BBox with X and Y coordinate swapped."""
        return BBox(self._bot_unit, self._left_unit, self._top_unit, self._right_unit,
                    self._res, unit_mode=True)

    def with_interval(self, direction, lower, upper, unit_mode=False):
        if not unit_mode:
            lower = int(round(lower / self._res))
            upper = int(round(upper / self._res))
        if direction == 'x':
            return BBox(lower, self._bot_unit, upper, self._top_unit, self._res, unit_mode=True)
        else:
            return BBox(self._left_unit, lower, self._right_unit, upper, self._res, unit_mode=True)

    def get_interval(self, direction, unit_mode=False):
        # type: (str, bool) -> Tuple[Union[float, int], Union[float, int]]
        """Returns the interval of this bounding box along the given direction.

        Parameters
        ----------
        direction : str
            direction along which to campute the bounding box interval.  Either 'x' or 'y'.
        unit_mode : bool
            True to return dimensions in resolution units.

        Returns
        -------
        lower : float
            the lower coordinate along the given direction.
        upper : float
            the upper coordinate along the given direction.
        """
        if direction == 'x':
            ans = self._left_unit, self._right_unit
        else:
            ans = self._bot_unit, self._top_unit

        if unit_mode:
            return ans
        return ans[0] * self._res, ans[1] * self._res

    def get_bounds(self, unit_mode=False):
        # type: (bool) -> Tuple[Union[float, int], ...]
        """Returns the bounds of this bounding box.

        Parameters
        ----------
        unit_mode : bool
            True to return bounds in resolution units.

        Returns
        -------
        bounds : Tuple[Union[float, int], ...]
            a tuple of (left, bottom, right, top) coordinates.
        """
        if unit_mode:
            return self._left_unit, self._bot_unit, self._right_unit, self._top_unit
        else:
            return self.left, self.bottom, self.right, self.top

    def is_physical(self):
        """Returns True if this bounding box has positive area.

        Returns
        -------
        is_physical : bool
            True if this bounding box has positive area.
        """
        return self._right_unit - self._left_unit > 0 and self._top_unit - self._bot_unit > 0

    def is_valid(self):
        """Returns True if this bounding box is valid, i.e. nonnegative area.

        Returns
        -------
        is_valid : bool
            True if this bounding box has nonnegative area.
        """
        return self._right_unit >= self._left_unit and self._top_unit >= self._bot_unit

    def get_immutable_key(self):
        """Returns an immutable key object that can be used to uniquely identify this BBox."""
        return (self.__class__.__name__, self._left_unit, self._bot_unit,
                self._right_unit, self._top_unit, self._res)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        precision = max(1, -1 * int(np.floor(np.log10(self._res))))
        fmt_str = '%s(%.{0}f, %.{0}f, %.{0}f, %.{0}f)'.format(precision)
        return fmt_str % (self.__class__.__name__, self.left, self.bottom, self.right, self.top)

    def __hash__(self):
        return hash(self.get_immutable_key())

    def __eq__(self, other):
        return self.get_immutable_key() == other.get_immutable_key()


class BBoxArray(object):
    """An array of bounding boxes.

    Useful for representing bus of wires.

    Parameters
    ----------
    bbox : BBox
        the lower-left bounding box.
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

    def __init__(self, bbox, nx=1, ny=1, spx=0, spy=0, unit_mode=False):
        # type: (BBox, int, int, Union[float, int], Union[float, int], bool) -> None
        if not isinstance(bbox, BBox):
            raise ValueError('%s is not a BBox object' % bbox)
        if nx <= 0 or ny <= 0:
            raise ValueError('Cannot have 0 bounding boxes.')
        if spx < 0 or spy < 0:
            raise ValueError('Currently does not support negative pitches.')

        self._bbox = bbox
        self._nx = nx
        self._ny = ny
        if unit_mode:
            self._spx_unit = int(spx)  # type: int
            self._spy_unit = int(spy)  # type: int
        else:
            self._spx_unit = int(round(spx / bbox.resolution))
            self._spy_unit = int(round(spy / bbox.resolution))

    def __iter__(self):
        # type: () -> Iterator[BBox]
        """Iterates over all bounding boxes in this BBoxArray.

        traverses from left to right, then from bottom to top.
        """
        for idx in range(self._nx * self._ny):
            yield self.get_bbox(idx)

    @property
    def base(self):
        # type: () -> BBox
        """the lower-left bounding box"""
        return self._bbox

    @property
    def nx(self):
        # type: () -> int
        """number of columns"""
        return self._nx

    @property
    def ny(self):
        # type: () -> int
        """number of columns"""
        return self._ny

    @property
    def spx(self):
        # type: () -> float
        """column pitch"""
        return self._spx_unit * self._bbox.resolution

    @property
    def spx_unit(self):
        # type: () -> int
        """column pitch in resolution units."""
        return self._spx_unit

    @property
    def spy(self):
        # type: () -> float
        """row pitch"""
        return self._spy_unit * self._bbox.resolution

    @property
    def spy_unit(self):
        # type: () -> int
        """row pitch in resolution units."""
        return self._spy_unit

    @property
    def left(self):
        # type: () -> float
        """left-most edge coordinate."""
        return self._bbox.left

    @property
    def left_unit(self):
        # type: () -> int
        """left-most edge coordinate."""
        return self._bbox.left_unit

    @property
    def right(self):
        # type: () -> float
        """right-most edge coordinate."""
        return self.right_unit * self._bbox.resolution

    @property
    def right_unit(self):
        # type: () -> int
        """right-most edge coordinate."""
        return self._bbox.right_unit + self._spx_unit * (self._nx - 1)

    @property
    def bottom(self):
        # type: () -> float
        """bottom-most edge coordinate."""
        return self._bbox.bottom

    @property
    def bottom_unit(self):
        # type: () -> int
        """bottom-most edge coordinate."""
        return self._bbox.bottom_unit

    @property
    def top(self):
        # type: () -> float
        """top-most edge coordinate."""
        return self.top_unit * self._bbox.resolution

    @property
    def top_unit(self):
        # type: () -> int
        """top-most edge coordinate."""
        return self._bbox.top_unit + self._spy_unit * (self._ny - 1)

    @property
    def xc(self):
        return self.xc_unit * self._bbox.resolution

    @property
    def xc_unit(self):
        # type: () -> int
        return (self.left_unit + self.right_unit) // 2

    @property
    def yc(self):
        return self.yc_unit * self._bbox.resolution

    @property
    def yc_unit(self):
        # type: () -> int
        return (self.bottom_unit + self.top_unit) // 2

    def as_bbox_collection(self):
        # type: () -> 'BBoxCollection'
        """Cast this BBoxArray as a BBoxCollection."""
        return BBoxCollection([self])

    def get_bbox(self, idx):
        # type: (int) -> BBox
        """Returns the bounding box with the given index.

        index increases from left to right, then from bottom to top.  lower-left box is index 0.

        Returns
        -------
        bbox : bag.layout.util.BBox
            the bounding box with the given index.
        """
        row_idx, col_idx = divmod(idx, self._nx)
        return self._bbox.transform(loc=(col_idx * self._spx_unit,
                                         row_idx * self._spy_unit), unit_mode=True)

    def get_overall_bbox(self):
        """Returns the overall bounding box of this BBoxArray.

        Returns
        -------
        overall_bbox : bag.layout.util.BBox
            the overall bounding box of this BBoxArray.
        """
        return BBox(self.left_unit, self.bottom_unit, self.right_unit, self.top_unit,
                    self._bbox.resolution, unit_mode=True)

    def move_by(self, dx=0, dy=0, unit_mode=False):
        # type: (Union[float, int], Union[float, int], bool) -> BBoxArray
        """Returns a new BBox shifted by the given amount.

        Parameters
        ----------
        dx : float
            shift in X direction.
        dy : float
            shift in Y direction.
        unit_mode : bool
            True if shifts are given in resolution units

        Returns
        -------
        box_arr : BBoxArray
            the new BBoxArray.
        """
        return self.transform((dx, dy), unit_mode=unit_mode)

    def transform(self, loc=(0, 0), orient='R0', unit_mode=False):
        # type: (Tuple[Union[float, int], Union[float, int]], str, bool) -> BBoxArray
        """Returns a new BBoxArray under the given transformation.

        rotates first before shift.

        Parameters
        ----------
        loc : Tuple[Union[float, int], Union[float, int]]
            location of the anchor.
        orient : str
            the orientation of the bounding box.
        unit_mode : bool
            True if location is given in resolution units

        Returns
        -------
        box_arr : BBoxArray
            the new BBoxArray.
        """
        if unit_mode:
            dx, dy = loc[0], loc[1]
        else:
            res = self._bbox.resolution
            dx = int(round(loc[0] / res))
            dy = int(round(loc[1] / res))

        if orient == 'R0':
            left = self.left_unit + dx
            bottom = self.bottom_unit + dy
        elif orient == 'MX':
            left = self.left_unit + dx
            bottom = -self.top_unit + dy
        elif orient == 'MY':
            left = -self.right_unit + dx
            bottom = self.bottom_unit + dy
        elif orient == 'R180':
            left = -self.right_unit + dx
            bottom = -self.top_unit + dy
        else:
            raise ValueError('Invalid orientation: ' + orient)

        # no 90 degree-ish rotation; width and height will not interchange
        new_base = BBox(left, bottom, left + self._bbox.width_unit,
                        bottom + self._bbox.height_unit, self._bbox.resolution,
                        unit_mode=True)
        return BBoxArray(new_base, nx=self._nx, ny=self._ny,
                         spx=self._spx_unit, spy=self._spy_unit, unit_mode=True)

    def arrayed_copies(self, nx=1, ny=1, spx=0, spy=0, unit_mode=False):
        # type: (int, int, Union[float, int], Union[float, int], bool) -> 'BBoxCollection'
        """Returns a BBoxCollection containing arrayed copies of this BBoxArray

        Parameters
        ----------
        nx : int
            number of copies in horizontal direction.
        ny : int
            number of copies in vertical direction.
        spx : Union[float, int]
            pitch in horizontal direction.
        spy : Union[float, int]
            pitch in vertical direction.
        unit_mode : bool
            True if location is given in resolution units

        Returns
        -------
        bcol : :class:`bag.layout.util.BBoxCollection`
            a BBoxCollection of the arrayed copies.
        """
        if not unit_mode:
            res = self._bbox.resolution
            spx = int(round(spy / res))
            spy = int(round(spy / res))

        x_info = self._array_helper(nx, spx, self.nx, self._spx_unit)
        y_info = self._array_helper(ny, spy, self.ny, self._spy_unit)

        base = self.base
        barr_list = [BBoxArray(base.move_by(dx, dy, unit_mode=True), nx=new_nx, ny=new_ny,
                               spx=new_spx, spy=new_spy)
                     for new_nx, new_spx, dx in zip(*x_info)
                     for new_ny, new_spy, dy in zip(*y_info)]
        return BBoxCollection(barr_list)

    @staticmethod
    def _array_helper(n1, sp1, n2, sp2):
        if n1 == 1:
            return [n2], [sp2], [0]
        elif n2 == 1:
            return [n1], [sp1], [0]
        elif sp1 == sp2 * n2:
            return [n1 * n2], [sp2], [0]
        elif sp2 == sp1 * n1:
            return [n1 * n2], [sp1], [0]
        else:
            # no way to express as single array
            if n1 < n2 or (n1 == n2 and sp2 < sp1):
                return [n2] * n1, [sp2] * n1, list(range(0, sp1 * n1, sp1))
            else:
                return [n1] * n2, [sp1] * n2, list(range(0, sp2 * n2, sp2))

    def __str__(self):
        return repr(self)

    def __repr__(self):
        precision = max(1, -1 * int(np.floor(np.log10(self._bbox.resolution))))
        fmt_str = '%s(%s, %d, %d, %.{0}f, %.{0}f)'.format(precision)
        return fmt_str % (self.__class__.__name__, self._bbox, self._nx,
                          self._ny, self.spx, self.spy)


class BBoxCollection(object):
    """A collection of bounding boxes.

    To support efficient computation, this class stores bounding boxes as a list of
    BBoxArray objects.

    Parameters
    ----------
    box_arr_list : list[bag.layout.util.BBoxArray]
        list of BBoxArrays in this collections.
    """

    def __init__(self, box_arr_list):
        self._box_arr_list = box_arr_list

    def __iter__(self):
        """Iterates over all BBoxArray in this collection."""
        return self._box_arr_list.__iter__()

    def __reversed__(self):
        return self._box_arr_list.__reversed__()

    def __len__(self):
        return len(self._box_arr_list)

    def as_bbox_array(self):
        """Attempt to cast this BBoxCollection into a BBoxArray.

        Returns
        -------
        bbox_arr : bag.layout.util.BBoxArray
            the BBoxArray object that's equivalent to this BBoxCollection.

        Raises
        ------
        Exception :
            if this BBoxCollection cannot be cast into a BBoxArray.
        """
        if len(self._box_arr_list) != 1:
            raise Exception('Unable to cast this BBoxCollection into a BBoxArray.')

        return self._box_arr_list[0]

    def as_bbox(self):
        """Attempt to cast this BBoxCollection into a BBox.

        Returns
        -------
        bbox : bag.layout.util.BBox
            the BBox object that's equivalent to this BBoxCollection.

        Raises
        ------
        Exception :
            if this BBoxCollection cannot be cast into a BBox.
        """
        if len(self._box_arr_list) != 1:
            raise Exception('Unable to cast this BBoxCollection into a BBoxArray.')
        box_arr = self._box_arr_list[0]
        if box_arr.nx != 1 or box_arr.ny != 1:
            raise Exception('Unable to cast this BBoxCollection into a BBoxArray.')
        return box_arr.base

    def get_bounding_box(self):
        """Returns the bounding box that encloses all boxes in this collection.

        Returns
        -------
        bbox : bag.layout.util.BBox
            the bounding box of this BBoxCollection.
        """
        box = BBox.get_invalid_bbox()
        for box_arr in self._box_arr_list:
            all_box = BBox(box_arr.left, box_arr.bottom, box_arr.right, box_arr.top,
                           box_arr.base.resolution)
            box = box.merge(all_box)

        return box

    def transform(self, loc=(0, 0), orient='R0'):
        """Returns a new BBoxCollection under the given transformation.

        rotates first before shift.

        Parameters
        ----------
        loc : (float, float)
            location of the anchor.
        orient : str
            the orientation of the bounding box.

        Returns
        -------
        box_collection : bag.layout.util.BBoxCollection
            the new BBoxCollection.
        """
        new_list = [box_arr.transform(loc=loc, orient=orient) for box_arr in self._box_arr_list]
        return BBoxCollection(new_list)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return pprint.pformat(self._box_arr_list)


class Pin(object):
    """A layout pin.

    Multiple pins can share the same terminal name.

    Parameters
    ----------
    pin_name : str
        the pin label.
    term_name : str
        the terminal name.
    layer : str
        the pin layer name.
    bbox : bag.layout.util.BBox
        the pin bounding box.
    """

    def __init__(self, pin_name, term_name, layer, bbox):
        if not bbox.is_physical():
            raise Exception('Non-physical pin bounding box: %s' % bbox)

        self._pin_name = pin_name
        self._term_name = term_name
        self._layer = layer
        self._bbox = bbox

    @property
    def pin_name(self):
        """the pin label."""
        return self._pin_name

    @property
    def term_name(self):
        """the terminal name."""
        return self._term_name

    @property
    def layer(self):
        """the pin layer name"""
        return self._layer

    @property
    def bbox(self):
        """the pin bounding box."""
        return self._bbox

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return '%s(%s, %s, %s, %s)' % (self.__class__.__name__, self._pin_name,
                                       self._term_name, self._layer, self._bbox)
