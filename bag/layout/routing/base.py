# -*- coding: utf-8 -*-

"""This module provides basic routing classes.
"""

from typing import Tuple, Union, Generator, Dict, List, Sequence

import numbers

from ...util.search import BinaryIterator
from ..util import BBox, BBoxArray
from .grid import RoutingGrid


class TrackID(object):
    """A class that represents locations of track(s) on the routing grid.

    Parameters
    ----------
    layer_id : int
        the layer ID.
    track_idx : Union[float, int]
        the smallest middle track index in the array.  Multiples of 0.5
    width : int
        width of one track in number of tracks.
    num : int
        number of tracks in this array.
    pitch : Union[float, int]
        pitch between adjacent tracks, in number of track pitches.
    """

    def __init__(self, layer_id, track_idx, width=1, num=1, pitch=0.0):
        # type: (int, Union[float, int], int, int, Union[float, int]) -> None
        if num < 1:
            raise ValueError('TrackID must have 1 or more tracks.')

        self._layer_id = layer_id
        self._hidx = int(round(2 * track_idx)) + 1
        self._w = width
        self._n = num
        self._hpitch = 0 if num == 1 else int(pitch * 2)

    def __repr__(self):
        arg_list = ['layer=%d' % self._layer_id]
        if self._hidx % 2 == 1:
            arg_list.append('track=%d' % ((self._hidx - 1) // 2))
        else:
            arg_list.append('track=%.1f' % ((self._hidx - 1) / 2))
        if self._w != 1:
            arg_list.append('width=%d' % self._w)
        if self._n != 1:
            arg_list.append('num=%d' % self._n)
            if self._hpitch % 2 == 0:
                arg_list.append('pitch=%d' % (self._hpitch // 2))
            else:
                arg_list.append('pitch=%.1f' % (self._hpitch / 2))

        return '%s(%s)' % (self.__class__.__name__, ', '.join(arg_list))

    def __str__(self):
        return repr(self)

    @property
    def layer_id(self):
        # type: () -> int
        return self._layer_id

    @property
    def width(self):
        # type: () -> int
        return self._w

    @property
    def base_index(self):
        # type: () -> Union[float, int]
        if self._hidx % 2 == 1:
            return (self._hidx - 1) // 2
        return (self._hidx - 1) / 2

    @property
    def index_htr(self):
        # type: () -> int
        return self._hidx

    @property
    def num(self):
        # type: () -> int
        return self._n

    @property
    def pitch(self):
        # type: () -> Union[float, int]
        if self._hpitch % 2 == 0:
            return self._hpitch // 2
        return self._hpitch / 2

    @property
    def pitch_htr(self):
        # type: () -> int
        return self._hpitch

    def get_immutable_key(self):
        return self.__class__.__name__, self._layer_id, self._hidx, self._w, self._n, self._hpitch

    def get_bounds(self, grid, unit_mode=False):
        # type: (RoutingGrid, bool) -> Tuple[Union[float, int], Union[float, int]]
        """Calculate the track bounds coordinate.

        Parameters
        ----------
        grid : RoutingGrid
            the RoutingGrid object.
        unit_mode : bool
            True to return coordinates in resolution units.

        Returns
        -------
        lower : Union[float, int]
            the lower bound coordinate perpendicular to track direction.
        upper : Union[float, int]
            the upper bound coordinate perpendicular to track direction.
        """
        lower, upper = grid.get_wire_bounds(self.layer_id, self.base_index,
                                            width=self.width, unit_mode=True)
        pitch_dim = (self._hpitch * grid.get_track_pitch(self._layer_id, unit_mode=True)) // 2
        upper += (self.num - 1) * pitch_dim
        if unit_mode:
            return lower, upper
        else:
            res = grid.resolution
            return lower * res, upper * res

    def __iter__(self):
        # type: () -> Generator[Union[float, int]]
        """Iterate over all middle track indices in this TrackID."""
        for idx in range(self._n):
            num = self._hidx + idx * self._hpitch
            if num % 2 == 1:
                yield (num - 1) // 2
            else:
                yield (num - 1) / 2

    def sub_tracks_iter(self, grid):
        # type: (RoutingGrid) -> Generator[TrackID]
        """Iterate through sub-TrackIDs where every track in sub-TrackID has the same layer name.

        This method is used to deal with double patterning layer.  If this TrackID is not
        on a double patterning layer, it simply yields itself.

        Parameters
        ----------
        grid : RoutingGrid
            the RoutingGrid object.

        Yields
        ------
        sub_id : TrackID
            a TrackID where all tracks has the same layer name.
        """
        layer_id = self._layer_id
        layer_names = grid.tech_info.get_layer_name(layer_id)
        if isinstance(layer_names, tuple):
            den = 2 * len(layer_names)
            if self._hpitch % den == 0:
                # layer name will never change
                yield self
            else:
                # TODO: have more robust solution than just yielding tracks one by one?
                for tr_idx in self:
                    yield TrackID(layer_id, tr_idx, width=self.width)
        else:
            yield self

    def transform(self, grid, loc=(0, 0), orient="R0", unit_mode=False):
        # type: (RoutingGrid, Tuple[Union[float, int], Union[float, int]], str, bool) -> TrackID
        """returns a transformation of this TrackID."""
        layer_id = self._layer_id
        is_x = grid.get_direction(layer_id) == 'x'
        if orient == 'R0':
            base_hidx = self._hidx
        elif orient == 'MX':
            if is_x:
                base_hidx = -self._hidx - (self._n - 1) * self._hpitch
            else:
                base_hidx = self._hidx
        elif orient == 'MY':
            if is_x:
                base_hidx = self._hidx
            else:
                base_hidx = -self._hidx - (self._n - 1) * self._hpitch
        elif orient == 'R180':
            base_hidx = -self._hidx - (self._n - 1) * self._hpitch
        else:
            raise ValueError('Unsupported orientation: %s' % orient)

        delta = loc[1] if is_x else loc[0]
        delta = grid.coord_to_track(layer_id, delta, unit_mode=unit_mode) + 0.5
        return TrackID(layer_id, (base_hidx - 1) / 2 + delta, width=self._w,
                       num=self._n, pitch=self.pitch)


class WireArray(object):
    """An array of wires on the routing grid.

    Parameters
    ----------
    track_id : :class:`bag.layout.routing.TrackID`
        TrackArray representing the track locations of this wire array.
    lower : Union[float, int]
        the lower coordinate along the track direction.
    upper : Union[float, int]
        the upper coordinate along the track direction.
    res : Optional[float]
        the resolution unit.
    unit_mode : bool
        True if lower/upper are specified in resolution units.
    """

    def __init__(self, track_id, lower, upper, res=None, unit_mode=False):
        # type: (TrackID, Union[float, int], Union[float, int], Optional[float], bool) -> None
        if res is None:
            raise ValueError('Please specify the layout distance resolution.')

        self._track_id = track_id
        self._res = res
        if unit_mode:
            self._lower_unit = int(lower)  # type: int
            self._upper_unit = int(upper)  # type: int
        else:
            self._lower_unit = int(round(lower / res))
            self._upper_unit = int(round(upper / res))

    def __repr__(self):
        return '%s(%s, %.d, %.d, %.4g)' % (self.__class__.__name__, self._track_id,
                                           self._lower_unit, self._upper_unit, self._res)

    def __str__(self):
        return repr(self)

    @property
    def resolution(self):
        return self._res

    @property
    def lower(self):
        return self._lower_unit * self._res

    @property
    def upper(self):
        return self._upper_unit * self._res

    @property
    def middle(self):
        return (self._lower_unit + self._upper_unit) // 2 * self._res

    @property
    def lower_unit(self):
        return self._lower_unit

    @property
    def upper_unit(self):
        return self._upper_unit

    @property
    def middle_unit(self):
        return (self._lower_unit + self._upper_unit) // 2

    @property
    def track_id(self):
        # type: () -> TrackID
        """Returns the TrackID of this WireArray."""
        return self._track_id

    @property
    def layer_id(self):
        # type: () -> int
        """Returns the layer ID of this WireArray."""
        return self.track_id.layer_id

    @property
    def width(self):
        return self.track_id.width

    @classmethod
    def list_to_warr(cls, warr_list):
        # type: (List[WireArray]) -> WireArray
        """Convert a list of WireArrays to a single WireArray.

        this method assumes all WireArrays have the same layer, width, and lower/upper coordinates.
        Overlapping WireArrays will be compacted.
        """
        if len(warr_list) == 1:
            return warr_list[0]

        tid0 = warr_list[0].track_id
        layer = tid0.layer_id
        width = tid0.width
        res = warr_list[0].resolution
        lower, upper = warr_list[0].lower_unit, warr_list[0].upper_unit
        tid_list = sorted(set((int(idx * 2) for warr in warr_list for idx in warr.track_id)))
        base_idx2 = tid_list[0]
        base_idx = base_idx2 // 2 if base_idx2 % 2 == 0 else base_idx2 / 2
        if len(tid_list) < 2:
            return WireArray(TrackID(layer, base_idx, width=width), lower, upper,
                             res=res, unit_mode=True)
        diff = tid_list[1] - tid_list[0]
        for idx in range(1, len(tid_list) - 1):
            if tid_list[idx + 1] - tid_list[idx] != diff:
                raise ValueError('pitch mismatch.')
        pitch = diff // 2 if diff % 2 == 0 else diff / 2

        return WireArray(TrackID(layer, base_idx, width=width, num=len(tid_list), pitch=pitch),
                         lower, upper, res=res, unit_mode=True)

    @classmethod
    def single_warr_iter(cls, warr):
        if isinstance(warr, WireArray):
            yield from warr.warr_iter()
        else:
            for w in warr:
                yield from w.warr_iter()

    def get_immutable_key(self):
        return (self.__class__.__name__, self._track_id.get_immutable_key(), self._lower_unit,
                self._upper_unit, self._res)

    def to_warr_list(self):
        return list(self.warr_iter())

    def warr_iter(self):
        tid = self._track_id
        layer = tid.layer_id
        width = tid.width
        for tr in tid:
            yield WireArray(TrackID(layer, tr, width=width), self._lower_unit,
                            self._upper_unit, res=self._res, unit_mode=True)

    def get_bbox_array(self, grid):
        # type: ('RoutingGrid') -> BBoxArray
        """Returns the BBoxArray representing this WireArray.

        Parameters
        ----------
        grid : RoutingGrid
            the RoutingGrid of this WireArray.

        Returns
        -------
        bbox_arr : BBoxArray
            the BBoxArray of the wires.
        """
        track_id = self.track_id
        tr_w = track_id.width
        layer_id = track_id.layer_id
        base_idx = track_id.base_index
        num = track_id.num

        base_box = grid.get_bbox(layer_id, base_idx, self._lower_unit, self._upper_unit,
                                 width=tr_w, unit_mode=True)
        tot_pitch = (track_id.pitch_htr * grid.get_track_pitch(layer_id, unit_mode=True)) // 2
        if grid.get_direction(layer_id) == 'x':
            return BBoxArray(base_box, ny=num, spy=tot_pitch, unit_mode=True)
        else:
            return BBoxArray(base_box, nx=num, spx=tot_pitch, unit_mode=True)

    def wire_iter(self, grid):
        """Iterate over all wires in this WireArray as layer/BBox pair.

        Parameters
        ----------
        grid : :class:`bag.layout.routing.RoutingGrid`
            the RoutingGrid of this WireArray.

        Yields
        ------
        layer : string
            the wire layer name.
        bbox : :class:`bag.layout.util.BBox`
            the wire bounding box.
        """
        tr_w = self.track_id.width
        layer_id = self.layer_id
        for tr_idx in self.track_id:
            layer_name = grid.get_layer_name(layer_id, tr_idx)
            bbox = grid.get_bbox(layer_id, tr_idx, self._lower_unit, self._upper_unit,
                                 width=tr_w, unit_mode=True)
            yield layer_name, bbox

    def wire_arr_iter(self, grid):
        """Iterate over all wires in this WireArray as layer/BBoxArray pair.

        This method group all rectangles in the same layer together.

        Parameters
        ----------
        grid : :class:`bag.layout.routing.RoutingGrid`
            the RoutingGrid of this WireArray.

        Yields
        ------
        layer : string
            the wire layer name.
        bbox : :class:`bag.layout.util.BBoxArray`
            the wire bounding boxes.
        """
        res = self._res
        tid = self.track_id
        layer_id = tid.layer_id
        tr_width = tid.width
        track_pitch = grid.get_track_pitch(layer_id, unit_mode=True)
        is_x = grid.get_direction(layer_id) == 'x'
        for track_idx in tid.sub_tracks_iter(grid):
            base_idx = track_idx.base_index
            cur_layer = grid.get_layer_name(layer_id, base_idx)
            cur_num = track_idx.num
            wire_pitch = (track_idx.pitch_htr * track_pitch) // 2
            tl, tu = grid.get_wire_bounds(layer_id, base_idx, width=tr_width, unit_mode=True)
            if is_x:
                base_box = BBox(self._lower_unit, tl, self._upper_unit, tu, res, unit_mode=True)
                box_arr = BBoxArray(base_box, ny=cur_num, spy=wire_pitch, unit_mode=True)
            else:
                base_box = BBox(tl, self._lower_unit, tu, self._upper_unit, res, unit_mode=True)
                box_arr = BBoxArray(base_box, nx=cur_num, spx=wire_pitch, unit_mode=True)

            yield cur_layer, box_arr

    def transform(self, grid, loc=(0, 0), orient='R0', unit_mode=False):
        """Return a new transformed WireArray.

        Parameters
        ----------
        grid : :class:`bag.layout.routing.RoutingGrid`
            the RoutingGrid of this WireArray.
        loc : Tuple[Union[float, int], Union[float, int]]
            the X/Y coordinate shift.
        orient : str
            the new orientation.
        unit_mode : bool
            True if location is given in unit mode.
        """
        res = self._res
        if not unit_mode:
            loc = int(round(loc[0] / res)), int(round(loc[1] / res))

        layer_id = self.layer_id
        is_x = grid.get_direction(layer_id) == 'x'
        if orient == 'R0':
            lower, upper = self._lower_unit, self._upper_unit
        elif orient == 'MX':
            if is_x:
                lower, upper = self._lower_unit, self._upper_unit
            else:
                lower, upper = -self._upper_unit, -self._lower_unit
        elif orient == 'MY':
            if is_x:
                lower, upper = -self._upper_unit, -self._lower_unit
            else:
                lower, upper = self._lower_unit, self._upper_unit
        elif orient == 'R180':
            lower, upper = -self._upper_unit, -self._lower_unit
        else:
            raise ValueError('Unsupported orientation: %s' % orient)

        delta = loc[0] if is_x else loc[1]
        return WireArray(self.track_id.transform(grid, loc=loc, orient=orient, unit_mode=True),
                         lower + delta, upper + delta, res=res, unit_mode=True)


class Port(object):
    """A layout port.

    a port is a group of pins that represent the same net.
    The pins can be on different layers.

    Parameters
    ----------
    term_name : str
        the terminal name of the port.
    pin_dict : dict[int, list[bag.layout.routing.WireArray]]
        a dictionary from layer ID to pin geometries on that layer.
    """

    def __init__(self, term_name, pin_dict, label=''):
        self._term_name = term_name
        self._pin_dict = pin_dict
        self._label = label or term_name

    def __iter__(self):
        """Iterate through all pin geometries in this port.

        the iteration order is not guaranteed.
        """
        for geo_list in self._pin_dict.values():
            yield from geo_list

    def get_single_layer(self):
        # type: () -> Union[int, str]
        """Returns the layer of this port if it only has a single layer."""
        if len(self._pin_dict) > 1:
            raise ValueError('This port has more than one layer.')
        return next(iter(self._pin_dict))

    def _get_layer(self, layer):
        """Get the layer number."""
        if isinstance(layer, numbers.Integral):
            return self.get_single_layer() if layer < 0 else layer
        else:
            return self.get_single_layer() if not layer else layer

    @property
    def net_name(self):
        """Returns the net name of this port."""
        return self._term_name

    @property
    def label(self):
        """Returns the label of this port."""
        return self._label

    def get_pins(self, layer=-1):
        """Returns the pin geometries on the given layer.

        Parameters
        ----------
        layer : int
            the layer ID.  If Negative, check if this port is on a single layer,
            then return the result.

        Returns
        -------
        track_bus_list : Union[WireArray, BBox]
            pins on the given layer representing as WireArrays.
        """
        layer = self._get_layer(layer)
        return self._pin_dict.get(layer, [])

    def get_bounding_box(self, grid, layer=-1):
        """Calculate the overall bounding box of this port on the given layer.

        Parameters
        ----------
        grid : :class:`~bag.layout.routing.RoutingGrid`
            the RoutingGrid of this Port.
        layer : int
            the layer ID.  If Negative, check if this port is on a single layer,
            then return the result.

        Returns
        -------
        bbox : BBox
            the bounding box.
        """
        layer = self._get_layer(layer)
        box = BBox.get_invalid_bbox()
        for geo in self._pin_dict[layer]:
            if isinstance(geo, BBox):
                box = box.merge(geo)
            else:
                box = box.merge(geo.get_bbox_array(grid).get_overall_bbox())
        return box

    def transform(self, grid, loc=(0, 0), orient='R0', unit_mode=False):
        # type: (RoutingGrid, Tuple[Union[float, int], Union[float, int]], str, bool) -> Port
        """Return a new transformed Port.

        Parameters
        ----------
        grid : RoutingGrid
            the RoutingGrid of this Port.
        loc : Tuple[Union[float, int], Union[float, int]]
            the X/Y coordinate shift.
        orient : str
            the new orientation.
        unit_mode: bool
            True if location is in resolution units.
        """
        if not unit_mode:
            res = grid.resolution
            loc = (int(round(loc[0] / res)), int(round(loc[1] / res)))

        new_pin_dict = {}
        for lay, geo_list in self._pin_dict.items():
            new_geo_list = []
            for geo in geo_list:
                if isinstance(geo, BBox):
                    new_geo_list.append(geo.transform(loc=loc, orient=orient, unit_mode=True))
                else:
                    new_geo_list.append(geo.transform(grid, loc=loc, orient=orient, unit_mode=True))
            new_pin_dict[lay] = new_geo_list

        return Port(self._term_name, new_pin_dict, label=self._label)


class TrackManager(object):
    """A class that makes it easy to compute track locations.

    This class provides many helper methods for computing track locations and spacing when
    each track could have variable width.  All methods in this class accepts a "track_type",
    which is either a string in the track dictionary or an integer representing the track
    width.

    Parameters
    ----------
    grid : RoutingGrid
        the RoutingGrid object.
    tr_widths : Dict[str, Dict[int, int]]
        dictionary from wire types to its width on each layer.
    tr_spaces : Dict[Union[str, Tuple[str, str]], Dict[int, Union[float, int]]]
        dictionary from wire types to its spaces on each layer.
    **kwargs :
        additional options.
    """

    def __init__(self,
                 grid,  # type: RoutingGrid
                 tr_widths,  # type: Dict[str, Dict[int, int]]
                 tr_spaces,  # type: Dict[Union[str, Tuple[str, str]], Dict[int, Union[float, int]]]
                 **kwargs
                 ):
        # type: (...) -> None
        half_space = kwargs.get('half_space', False)

        self._grid = grid
        self._tr_widths = tr_widths
        self._tr_spaces = tr_spaces
        self._half_space = half_space

    @property
    def grid(self):
        # type: () -> RoutingGrid
        return self._grid

    @property
    def half_space(self):
        # type: () -> bool
        return self._half_space

    def get_width(self, layer_id, track_type):
        # type: (int, Union[str, int]) -> int
        """Returns the track width.

        Parameters
        ----------
        layer_id : int
            the track layer ID.
        track_type : Union[str, int]
            the track type.
        """
        if isinstance(track_type, int):
            return track_type
        if track_type not in self._tr_widths:
            return 1
        return self._tr_widths[track_type].get(layer_id, 1)

    def get_space(self,  # type: TrackManager
                  layer_id,  # type: int
                  type_tuple,  # type: Union[str, int, Tuple[Union[str, int], Union[str, int]]]
                  **kwargs):
        # type: (...) -> Union[int, float]
        """Returns the track spacing.

        Parameters
        ----------
        layer_id : int
            the track layer ID.
        type_tuple : Union[str, int, Tuple[Union[str, int], Union[str, int]]]
            If a single track type is given, will return the minimum spacing needed around that
            track type.  If a tuple of two types are given, will return the specific spacing
            between those two track types if specified.  Otherwise, returns the maximum of all the
            valid spacing.
        **kwargs:
            optional parameters.
        """
        half_space = kwargs.get('half_space', self._half_space)
        sp_override = kwargs.get('sp_override', None)

        if isinstance(type_tuple, tuple):
            # if two specific wires are given, first check if any specific rules exist
            ans = self._get_space_from_tuple(layer_id, type_tuple, sp_override)
            if ans is not None:
                return ans
            ans = self._get_space_from_tuple(layer_id, type_tuple, self._tr_spaces)
            if ans is not None:
                return ans
            # no specific rules, so return max of wire spacings.
            ans = 0
            for wtype in type_tuple:
                cur_space = self._get_space_from_type(layer_id, wtype, sp_override)
                if cur_space is None:
                    cur_space = self._get_space_from_type(layer_id, wtype, self._tr_spaces)
                if cur_space is None:
                    cur_space = 0
                cur_width = self.get_width(layer_id, wtype)
                ans = max(ans, cur_space, self._grid.get_num_space_tracks(layer_id, cur_width,
                                                                          half_space=half_space))
            return ans
        else:
            cur_space = self._get_space_from_type(layer_id, type_tuple, sp_override)
            if cur_space is None:
                cur_space = self._get_space_from_type(layer_id, type_tuple, self._tr_spaces)
            if cur_space is None:
                cur_space = 0
            cur_width = self.get_width(layer_id, type_tuple)
            return max(cur_space, self._grid.get_num_space_tracks(layer_id, cur_width,
                                                                  half_space=half_space))

    @classmethod
    def _get_space_from_tuple(cls, layer_id, ntup, sp_dict):
        if sp_dict is not None:
            if ntup in sp_dict:
                return sp_dict[ntup].get(layer_id, None)
            ntup = (ntup[1], ntup[0])
            if ntup in sp_dict:
                return sp_dict[ntup].get(layer_id, None)
        return None

    @classmethod
    def _get_space_from_type(cls, layer_id, wtype, sp_dict):
        if sp_dict is None:
            return None
        if wtype in sp_dict:
            test = sp_dict[wtype]
        else:
            key = (wtype, '')
            if key in sp_dict:
                test = sp_dict[key]
            else:
                key = ('', wtype)
                if key in sp_dict:
                    test = sp_dict[key]
                else:
                    test = None

        if test is None:
            return None
        return test.get(layer_id, None)

    def get_next_track(self,  # type: TrackManager
                       layer_id,  # type: int
                       cur_idx,  # type: Union[float, int]
                       cur_type,  # type: Union[str, int]
                       next_type,  # type: Union[str, int]
                       up=True,  # type: bool
                       **kwargs):
        # type: (...) -> Union[float, int]
        """Compute the track location of a wire next to a given one.

        Parameters
        ----------
        layer_id : int
            the layer ID.
        cur_idx : Union[float, int]
            the current wire track index.
        cur_type : Union[str, int]
            the current wire type.
        next_type : Union[str, int]
            the next wire type.
        up : bool
            True to return the next track index that is larger than cur_idx.
        **kwargs :
            optional parameters.

        Returns
        -------
        next_int : Union[float, int]
            the next track index.
        """
        cur_width = self.get_width(layer_id, cur_type)
        next_width = self.get_width(layer_id, next_type)
        space = self.get_space(layer_id, (cur_type, next_type), **kwargs)
        if up:
            par_test = int(round(2 * cur_idx + 2 * space + cur_width + next_width))
        else:
            par_test = int(round(2 * cur_idx - 2 * space - cur_width - next_width))

        return par_test // 2 if par_test % 2 == 0 else par_test / 2

    def place_wires(self,  # type: TrackManager
                    layer_id,  # type: int
                    type_list,  # type: Sequence[Union[str, int]]
                    start_idx=0,  # type: Union[float, int]
                    **kwargs):
        # type: (...) -> Tuple[Union[float, int], List[Union[float, int]]]
        """Place the given wires next to each other.

        Parameters
        ----------
        layer_id : int
            the layer of the tracks.
        type_list : Sequence[Union[str, int]]
            list of wire types.
        start_idx : Union[float, int]
            the starting track index.
        **kwargs:
            optional parameters for get_num_space_tracks() method of RoutingGrid.

        Returns
        -------
        num_tracks : Union[float, int]
            number of tracks used.
        locations : List[Union[float, int]]
            the center track index of each wire.
        """
        if not type_list:
            return 0, []

        prev_type = type_list[0]
        w0 = self.get_width(layer_id, prev_type)
        par_test = int(round(2 * start_idx + w0 - 1))
        mid_idx = par_test // 2 if par_test % 2 == 0 else par_test / 2
        ans = [mid_idx]
        for idx in range(1, len(type_list)):
            ans.append(self.get_next_track(layer_id, ans[-1], type_list[idx - 1],
                                           type_list[idx], up=True, **kwargs))

        w1 = self.get_width(layer_id, type_list[-1])
        par_test = int(round(w0 + w1 + 2 * (ans[-1] - ans[0])))
        ntr = par_test // 2 if par_test % 2 == 0 else par_test / 2

        return ntr, ans

    @classmethod
    def _get_align_delta(cls, tot_ntr, num_used, alignment):
        if alignment == -1 or num_used == tot_ntr:
            # we already aligned to left
            return 0
        elif alignment == 0:
            # center tracks
            delta_htr = int((tot_ntr - num_used) * 2) // 2
            return delta_htr / 2 if delta_htr % 2 == 1 else delta_htr // 2
        elif alignment == 1:
            # align to right
            return tot_ntr - num_used
        else:
            raise ValueError('Unknown alignment code: %d' % alignment)

    def align_wires(self,  # type: TrackManager
                    layer_id,  # type: int
                    type_list,  # type: Sequence[Union[str, int]]
                    tot_ntr,  # type: Union[float, int]
                    alignment=0,  # type: int
                    start_idx=0,  # type: Union[float, int]
                    **kwargs):
        # type: (...) -> List[Union[float, int]]
        """Place the given wires in the given space with the specified alignment.

        Parameters
        ----------
        layer_id : int
            the layer of the tracks.
        type_list : Sequence[Union[str, int]]
            list of wire types.
        tot_ntr : Union[float, int]
            total available space in number of tracks.
        alignment : int
            If alignment == -1, will "left adjust" the wires (left is the lower index direction).
            If alignment == 0, will center the wires in the middle.
            If alignment == 1, will "right adjust" the wires.
        start_idx : Union[float, int]
            the starting track index.
        **kwargs:
            optional parameters for place_wires().

        Returns
        -------
        locations : List[Union[float, int]]
            the center track index of each wire.
        """
        num_used, idx_list = self.place_wires(layer_id, type_list, start_idx=start_idx, **kwargs)
        if num_used > tot_ntr:
            raise ValueError('Given tracks occupy more space than given.')

        delta = self._get_align_delta(tot_ntr, num_used, alignment)
        return [idx + delta for idx in idx_list]

    def spread_wires(self,  # type: TrackManager
                     layer_id,  # type: int
                     type_list,  # type: Sequence[Union[str, int]]
                     tot_ntr,  # type: Union[float, int]
                     sp_type,  # type: Union[str, int, Tuple[Union[str, int], Union[str, int]]]
                     alignment=0,  # type: int
                     start_idx=0,  # type: Union[float, int]
                     max_sp=10000,  # type: int
                     sp_override=None,
                     ):
        # type: (...) -> List[Union[float, int]]
        """Spread out the given wires in the given space.

        This method tries to spread out wires by increasing the space around the given
        wire/combination of wires.

        Parameters
        ----------
        layer_id : int
            the layer of the tracks.
        type_list : Sequence[Union[str, int]]
            list of wire types.
        tot_ntr : Union[float, int]
            total available space in number of tracks.
        sp_type : Union[str, Tuple[str, str]]
            The space to increase.
        alignment : int
            If alignment == -1, will "left adjust" the wires (left is the lower index direction).
            If alignment == 0, will center the wires in the middle.
            If alignment == 1, will "right adjust" the wires.
        start_idx : Union[float, int]
            the starting track index.
        max_sp : int
            maximum space.
        sp_override :
            tracking spacing override dictionary.

        Returns
        -------
        locations : List[Union[float, int]]
            the center track index of each wire.
        """
        if not sp_override:
            sp_override = {sp_type: {layer_id: 0}}
        else:
            sp_override = sp_override.copy()
            sp_override[sp_type] = {layer_id: 0}
        cur_sp = int(round(2 * self.get_space(layer_id, sp_type)))
        bin_iter = BinaryIterator(cur_sp, None)
        while bin_iter.has_next():
            new_sp = bin_iter.get_next()
            if new_sp > 2 * max_sp:
                break
            sp_override[sp_type][layer_id] = new_sp / 2 if new_sp % 2 == 1 else new_sp // 2
            tmp = self.place_wires(layer_id, type_list, start_idx=start_idx,
                                   sp_override=sp_override)
            if tmp[0] > tot_ntr:
                bin_iter.down()
            else:
                bin_iter.save_info(tmp)
                bin_iter.up()

        if bin_iter.get_last_save_info() is None:
            raise ValueError('No solution found.')

        num_used, idx_list = bin_iter.get_last_save_info()
        delta = self._get_align_delta(tot_ntr, num_used, alignment)
        return [idx + delta for idx in idx_list]
