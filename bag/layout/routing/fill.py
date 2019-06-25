# -*- coding: utf-8 -*-

"""This module defines classes that provides automatic fill utility on a grid.
"""

from typing import TYPE_CHECKING, Optional, Union, List, Tuple, Any, Generator

from rtree.index import Index, Property

from bag.layout.util import BBox
from bag.util.search import BinaryIterator, minimize_cost_golden

if TYPE_CHECKING:
    from bag.layout.util import BBoxArray

    from .grid import RoutingGrid


class RectIndex(object):
    """A R-tree that stores all tracks on a layer."""

    def __init__(self, resolution, basename=None, overwrite=False):
        # type: (float, Optional[str], bool) -> None
        self._res = resolution
        self._cnt = 0
        if basename is None:
            self._index = Index(interleaved=True)
        else:
            p = Property(overwrite=overwrite)
            self._index = Index(basename, interleaved=True, properties=p)

    @property
    def bound_box(self):
        # type: () -> BBox
        xl, yb, xr, yt = self._index.bounds
        return BBox(int(xl), int(yb), int(xr), int(yt), self._res, unit_mode=True)

    def close(self):
        self._index.close()

    def record_box(self, box, dx, dy):
        # type: (BBox, int, int) -> None
        """Record the given BBox."""
        sp_box = box.expand(dx=dx, dy=dy, unit_mode=True)
        bnds = sp_box.get_bounds(unit_mode=True)
        obj = (box.left_unit, box.bottom_unit, box.right_unit, box.top_unit, dx, dy)
        self._index.insert(self._cnt, bnds, obj=obj)
        self._cnt += 1

    def rect_iter(self):
        # type: () -> Generator[Tuple[BBox, int, int], None, None]
        for xl, yb, xr, yt, sdx, sdy in self._index.intersection(self._index.bounds, objects='raw'):
            box_real = BBox(xl, yb, xr, yt, self._res, unit_mode=True)
            yield box_real, sdx, sdy

    def intersection_iter(self, box, dx=0, dy=0):
        # type: (BBox, int, int) -> Generator[BBox, None, None]
        """Finds all bounding box that intersects the given box."""
        res = self._res
        test_box = box.expand(dx=dx, dy=dy, unit_mode=True)
        box_iter = self._index.intersection(test_box.get_bounds(unit_mode=True), objects='raw')
        for xl, yb, xr, yt, sdx, sdy in box_iter:
            box_real = BBox(xl, yb, xr, yt, res, unit_mode=True)
            box_sp = box_real.expand(dx=sdx, dy=sdy, unit_mode=True)
            if box_sp.overlaps(box) or test_box.overlaps(box_real):
                yield box_real.expand(dx=max(dx, sdx), dy=max(dy, sdy), unit_mode=True)

    def intersection_rect_iter(self, box):
        # type: (BBox) -> Generator[BBox, None, None]
        """Finds all bounding box that intersects the given box."""
        res = self._res
        box_iter = self._index.intersection(box.get_bounds(unit_mode=True), objects='raw')
        for xl, yb, xr, yt, sdx, sdy in box_iter:
            yield BBox(xl, yb, xr, yt, res, unit_mode=True)


class UsedTracks(object):
    """A R-tree that stores all tracks in a template.
    """

    def __init__(self, save_file_basename=None, overwrite=False):
        # type: (Optional[str], bool) -> None
        self._idx_table = {}
        self._save_file_basename = save_file_basename
        self._overwrite = overwrite

    def __iter__(self):
        return self._idx_table.keys()

    def get_track_bbox(self, layer_id):
        # type: (int) -> BBox
        if layer_id not in self._idx_table:
            return BBox.get_invalid_bbox()
        return self._idx_table[layer_id].bound_box

    def track_box_iter(self):
        # type: () -> Generator[Tuple[int, BBox], None, None]
        for layer_id, rect_idx in self._idx_table.items():
            yield layer_id, rect_idx.bound_box

    def record_box(self, layer_id, box, dx, dy, res):
        # type: (int, BBox, int, int, float) -> None
        if layer_id not in self._idx_table:
            if self._save_file_basename is None:
                basename = None
            else:
                basename = self._save_file_basename + ('_%d' % layer_id)
            index = self._idx_table[layer_id] = RectIndex(res, basename, self._overwrite)
        else:
            index = self._idx_table[layer_id]
        index.record_box(box, dx, dy)

    def close(self):
        for index in self._idx_table.values():
            index.close()

    def record_rect(self, grid, layer_name, box_arr, dx=-1, dy=-1):
        # type: (RoutingGrid, Union[Tuple[str, str], str], BBoxArray, int, int) -> Optional[int]
        """Record the given bounding box array.  Returns the added layer ID."""
        tech_info = grid.tech_info

        if isinstance(layer_name, tuple):
            # TODO: find more process-portable fix?
            if layer_name[1] == 'exclude':
                return None
            layer_name = layer_name[0]
        try:
            layer_id = tech_info.get_layer_id(layer_name)
        except ValueError:
            return None

        if layer_id not in grid:
            return None

        if layer_id not in self._idx_table:
            if self._save_file_basename is None:
                basename = None
            else:
                basename = self._save_file_basename + ('_%d' % layer_id)
            index = self._idx_table[layer_id] = RectIndex(grid.resolution, basename,
                                                          self._overwrite)
        else:
            index = self._idx_table[layer_id]

        layer_type = tech_info.get_layer_type(layer_name)
        if grid.get_direction(layer_id) == 'x':
            w = box_arr.base.height_unit
            dx0 = tech_info.get_min_line_end_space(layer_type, w, unit_mode=True)
            dy0 = tech_info.get_min_space(layer_type, w, unit_mode=True, same_color=False)
        else:
            w = box_arr.base.width_unit
            dy0 = tech_info.get_min_line_end_space(layer_type, w, unit_mode=True)
            dx0 = tech_info.get_min_space(layer_type, w, unit_mode=True, same_color=False)

        if dx < 0:
            dx = dx0
        if dy < 0:
            dy = dy0

        for box in box_arr:
            index.record_box(box, dx, dy)

        return layer_id

    def all_rect_iter(self):
        # type: () -> Generator[Tuple[int, BBox, int, int], None, None]
        for layer_id, index in self._idx_table.items():
            for box, dx, dy, in index.rect_iter():
                yield layer_id, box, dx, dy

    def intersection_rect_iter(self, layer_id, box):
        # type: (int, BBox) -> Generator[BBox, None, None]
        """Finds all bounding box that intersects the given box."""
        if layer_id in self._idx_table:
            yield from self._idx_table[layer_id].intersection_rect_iter(box)

    def blockage_iter(self, layer_id, test_box, spx=0, spy=0):
        # type: (int, BBox, int, int) -> Generator[BBox, None, None]
        if layer_id in self._idx_table:
            yield from self._idx_table[layer_id].intersection_iter(test_box, dx=spx, dy=spy)


def fill_symmetric_const_space(area, sp_max, n_min, n_max, offset=0):
    # type: (int, int, int, int, int) -> List[Tuple[int, int]]
    """Fill the given 1-D area given maximum space spec alone.

    The method draws the minimum number of fill blocks needed to satisfy maximum spacing spec.
    The given area is filled with the following properties:

    1. all spaces are as close to the given space as possible (differ by at most 1),
       without exceeding it.
    2. the filled area is as uniform as possible.
    3. the filled area is symmetric about the center.
    4. fill is drawn as much as possible given the above constraints.

    fill is drawn such that space blocks abuts both area boundaries.

    Parameters
    ----------
    area : int
        the 1-D area to fill.
    sp_max : int
        the maximum space.
    n_min : int
        minimum fill length.
    n_max : int
        maximum fill length
    offset : int
        the fill area starting coordinate.

    Returns
    -------
    fill_intv : List[Tuple[int, int]]
        list of fill intervals.
    """
    if n_min > n_max:
        raise ValueError('min fill length = %d > %d = max fill length' % (n_min, n_max))

    # suppose we draw N fill blocks, then the filled area is A - (N + 1) * sp.
    # therefore, to maximize fill, with A and sp given, we need to minimize N.
    # since N = (A - sp) / (f + sp), where f is length of the fill, this tells
    # us we want to try filling with max block.
    # so we calculate the maximum number of fill blocks we'll use if we use
    # largest fill block.
    num_fill = -(-(area - sp_max) // (n_max + sp_max))
    if num_fill == 0:
        # we don't need fill; total area is less than sp_max.
        return []

    # at this point, using (num_fill - 1) max blocks is not enough, but num_fill
    # max blocks either fits perfectly or exceeds area.

    # calculate the fill block length if we use num_fill fill blocks, and sp_max
    # between blocks.
    blk_len = (area - (num_fill + 1) * sp_max) // num_fill
    if blk_len >= n_min:
        # we can draw fill using num_fill fill blocks.
        return fill_symmetric_helper(area, num_fill, sp_max, offset=offset, inc_sp=False,
                                     invert=False, fill_on_edge=False, cyclic=False)[0]

    # trying to draw num_fill fill blocks with sp_max between them results in fill blocks
    # that are too small.  This means we need to reduce the space between fill blocks.
    sp_max, remainder = divmod(area - num_fill * n_min, num_fill + 1)
    # we can achieve the new sp_max using fill with length n_min or n_min + 1.
    if n_max > n_min or remainder == 0:
        # if everything divides evenly or we can use two different fill lengths,
        # then we're done.
        return fill_symmetric_helper(area, num_fill, sp_max, offset=offset, inc_sp=False,
                                     invert=False, fill_on_edge=False, cyclic=False)[0]
    # If we're here, then we must use only one fill length
    # fill by inverting fill/space to try to get only one fill length
    sol, num_diff_sp = fill_symmetric_helper(area, num_fill + 1, n_max, offset=offset, inc_sp=False,
                                             invert=True, fill_on_edge=True, cyclic=False)
    if num_diff_sp == 0:
        # we manage to fill using only one fill length
        return sol

    # If we're here, that means num_fill + 1 is even.  So using num_fill + 2 will
    # guarantee solution.
    return fill_symmetric_helper(area, num_fill + 2, n_max, offset=offset, inc_sp=False,
                                 invert=True, fill_on_edge=True, cyclic=False)[0]


def fill_symmetric_min_density_info(area, targ_area, n_min, n_max, sp_min,
                                    sp_max=None, fill_on_edge=True, cyclic=False):
    # type: (int, int, int, int, int, Optional[int], bool, bool) -> Tuple[Tuple[Any, ...], bool]
    """Fill the given 1-D area as little as possible.

    Compute fill location such that the given area is filled with the following properties:

    1. the area is as uniform as possible.
    2. the area is symmetric with respect to the center
    3. all fill blocks have lengths between n_min and n_max.
    4. all fill blocks are at least sp_min apart.

    Parameters
    ----------
    area : int
        total number of space we need to fill.
    targ_area : int
        target minimum fill area.  If not achievable, will do the best that we can.
    n_min : int
        minimum length of the fill block.  Must be less than or equal to n_max.
    n_max : int
        maximum length of the fill block.
    sp_min : int
        minimum space between each fill block.
    sp_max : Optional[int]
        if given, make sure space between blocks does not exceed this value.
        Must be greater than sp_min
    fill_on_edge : bool
        If True, we put fill blocks on area boundary.  Otherwise, we put space block on
        area boundary.
    cyclic : bool
        If True, we assume we're filling in a cyclic area (it wraps around).

    Returns
    -------
    info : Tuple[Any, ...]
        the fill information tuple.
    invert : bool
        True if space/fill is inverted.
    """
    # first, fill as much as possible
    max_result = fill_symmetric_max_density_info(area, targ_area, n_min, n_max, sp_min,
                                                 sp_max=sp_max, fill_on_edge=fill_on_edge,
                                                 cyclic=cyclic)

    fill_area, nfill_opt = max_result[0][:2]
    if fill_area <= targ_area:
        # we cannot/barely meet area spec; return max result
        return max_result

    # now, reduce fill by doing binary search on n_max
    n_max_iter = BinaryIterator(n_min, n_max)
    while n_max_iter.has_next():
        n_max_cur = n_max_iter.get_next()
        try:
            info, invert = fill_symmetric_max_num_info(area, nfill_opt, n_min, n_max_cur, sp_min,
                                                       fill_on_edge=fill_on_edge, cyclic=cyclic)
            fill_area_cur = area - info[0] if invert else info[0]
            if invert:
                _, sp_cur = _get_min_max_blk_len(info)
            else:
                sp_cur = sp_min if info[1][2] == 0 else sp_min + 1
            if fill_area_cur >= targ_area and (sp_max is None or sp_cur <= sp_max):
                # both specs passed
                n_max_iter.save_info((info, invert))
                n_max_iter.down()
            else:
                # reduce n_max too much
                n_max_iter.up()

        except ValueError:
            # get here if n_min == n_max and there's no solution.
            n_max_iter.up()

    last_save = n_max_iter.get_last_save_info()
    if last_save is None:
        # no solution, return max result
        return max_result

    # return new minimum solution
    info, invert = last_save
    fill_area = area - info[0] if invert else info[0]
    return (fill_area, nfill_opt, info[1]), invert


def fill_symmetric_max_density_info(area, targ_area, n_min, n_max, sp_min,
                                    sp_max=None, fill_on_edge=True, cyclic=False):
    # type: (int, int, int, int, int, Optional[int], bool, bool) -> Tuple[Tuple[Any, ...], bool]
    """Fill the given 1-D area as much as possible.

    Compute fill location such that the given area is filled with the following properties:

    1. the area is as uniform as possible.
    2. the area is symmetric with respect to the center
    3. all fill blocks have lengths between n_min and n_max.
    4. all fill blocks are at least sp_min apart.

    Parameters
    ----------
    area : int
        total number of space we need to fill.
    targ_area : int
        target minimum fill area.  If not achievable, will do the best that we can.
    n_min : int
        minimum length of the fill block.  Must be less than or equal to n_max.
    n_max : int
        maximum length of the fill block.
    sp_min : int
        minimum space between each fill block.
    sp_max : Optional[int]
        if given, make sure space between blocks does not exceed this value.
        Must be greater than sp_min
    fill_on_edge : bool
        If True, we put fill blocks on area boundary.  Otherwise, we put space block on
        area boundary.
    cyclic : bool
        If True, we assume we're filling in a cyclic area (it wraps around).

    Returns
    -------
    info : Tuple[Any, ...]
        the fill information tuple.
    invert : bool
        True if space/fill is inverted.
    """

    # min area test
    nfill_min = 1
    try:
        try:
            fill_symmetric_max_num_info(area, nfill_min, n_min, n_max, sp_min,
                                        fill_on_edge=fill_on_edge, cyclic=cyclic)
        except (NoFillAbutEdgeError, NoFillChoiceError):
            # we need at least 2 fiils
            nfill_min = 2
            fill_symmetric_max_num_info(area, nfill_min, n_min, n_max, sp_min,
                                        fill_on_edge=fill_on_edge, cyclic=cyclic)
    except InsufficientAreaError:
        # cannot fill at all
        info, invert = fill_symmetric_max_num_info(area, 0, n_min, n_max, sp_min,
                                                   fill_on_edge=fill_on_edge, cyclic=cyclic)
        return (0, 0, info[1]), invert

    # fill area first monotonically increases with number of fill blocks, then monotonically
    # decreases (as we start adding more space than fill).  Therefore, a golden section search
    # can be done on the number of fill blocks to determine the optimum.
    def golden_fun(nfill):
        try:
            info2, invert2 = fill_symmetric_max_num_info(area, nfill, n_min, n_max, sp_min,
                                                         fill_on_edge=fill_on_edge, cyclic=cyclic)
        except ValueError:
            return 0
        if invert2:
            return area - info2[0]
        else:
            return info2[0]

    if sp_max is not None:
        if sp_max <= sp_min:
            raise ValueError('Cannot have sp_max = %d <= %d = sp_min' % (sp_max, sp_min))

        # find minimum nfill that meets sp_max spec

        def golden_fun2(nfill):
            try:
                info2, invert2 = fill_symmetric_max_num_info(area, nfill, n_min, n_max, sp_min,
                                                             fill_on_edge=fill_on_edge,
                                                             cyclic=cyclic)
                if invert2:
                    _, sp_cur = _get_min_max_blk_len(info2)
                else:
                    sp_cur = sp_min if info2[1][2] == 0 else sp_min + 1
                return -sp_cur
            except ValueError:
                return -sp_max - 1

        min_result = minimize_cost_golden(golden_fun2, -sp_max, offset=nfill_min, maxiter=None)
        if min_result.x is None:
            # try even steps
            min_result = minimize_cost_golden(golden_fun2, -sp_max, offset=nfill_min,
                                              step=2, maxiter=None)
            nfill_min = min_result.x
            if nfill_min is None:
                # should never get here...
                raise ValueError('No solution for sp_max = %d' % sp_max)
        else:
            nfill_min = min_result.x

    min_result = minimize_cost_golden(golden_fun, targ_area, offset=nfill_min, maxiter=None)
    nfill_opt = min_result.x
    if nfill_opt is None:
        nfill_opt = min_result.xmax
    info, invert = fill_symmetric_max_num_info(area, nfill_opt, n_min, n_max, sp_min,
                                               fill_on_edge=fill_on_edge, cyclic=cyclic)
    fill_area = area - info[0] if invert else info[0]
    return (fill_area, nfill_opt, info[1]), invert


def fill_symmetric_max_density(area,  # type: int
                               targ_area,  # type: int
                               n_min,  # type: int
                               n_max,  # type: int
                               sp_min,  # type: int
                               offset=0,  # type: int
                               sp_max=None,  # type: Optional[int]
                               fill_on_edge=True,  # type: bool
                               cyclic=False,  # type: bool
                               ):
    # type: (...) -> Tuple[List[Tuple[int, int]], int]
    """Fill the given 1-D area as much as possible.

    Compute fill location such that the given area is filled with the following properties:

    1. the area is as uniform as possible.
    2. the area is symmetric with respect to the center
    3. all fill blocks have lengths between n_min and n_max.
    4. all fill blocks are at least sp_min apart.

    Parameters
    ----------
    area : int
        total number of space we need to fill.
    targ_area : int
        target minimum fill area.  If not achievable, will do the best that we can.
    n_min : int
        minimum length of the fill block.  Must be less than or equal to n_max.
    n_max : int
        maximum length of the fill block.
    sp_min : int
        minimum space between each fill block.
    offset : int
        the starting coordinate of the total interval.
    sp_max : Optional[int]
        if given, make sure space between blocks does not exceed this value.
        Must be greater than sp_min
    fill_on_edge : bool
        If True, we put fill blocks on area boundary.  Otherwise, we put space block on
        area boundary.
    cyclic : bool
        If True, we assume we're filling in a cyclic area (it wraps around).

    Returns
    -------
    fill_interval : List[Tuple[int, int]]
        a list of [start, stop) intervals that needs to be filled.
    fill_area : int
        total filled area.  May or may not meet minimum density requirement.
    """
    max_result = fill_symmetric_max_density_info(area, targ_area, n_min, n_max, sp_min,
                                                 sp_max=sp_max, fill_on_edge=fill_on_edge,
                                                 cyclic=cyclic)
    (fill_area, _, args), invert = max_result
    return fill_symmetric_interval(*args, offset=offset, invert=invert)[0], fill_area


class InsufficientAreaError(ValueError):
    pass


class FillTooSmallError(ValueError):
    pass


class NoFillAbutEdgeError(ValueError):
    pass


class NoFillChoiceError(ValueError):
    pass


class EmptyRegionError(ValueError):
    pass


def fill_symmetric_max_num_info(tot_area, nfill, n_min, n_max, sp_min,
                                fill_on_edge=True, cyclic=False):
    # type: (int, int, int, int, int, bool, bool) -> Tuple[Tuple[Any, ...], bool]
    """Fill the given 1-D area as much as possible with given number of fill blocks.

    Compute fill location such that the given area is filled with the following properties:

    1. the area is as uniform as possible.
    2. the area is symmetric with respect to the center
    3. the area is filled as much as possible with exactly nfill blocks,
       with lengths between n_min and n_max.
    4. all fill blocks are at least sp_min apart.

    Parameters
    ----------
    tot_area : int
        total number of space we need to fill.
    nfill : int
        number of fill blocks to draw.
    n_min : int
        minimum length of the fill block.  Must be less than or equal to n_max.
    n_max : int
        maximum length of the fill block.
    sp_min : int
        minimum space between each fill block.
    fill_on_edge : bool
        If True, we put fill blocks on area boundary.  Otherwise, we put space block on
        area boundary.
    cyclic : bool
        If True, we assume we're filling in a cyclic area (it wraps around).

    Returns
    -------
    info : Tuple[Any, ...]
        the fill information tuple.
    invert : bool
        True if space/fill is inverted.
    """
    # error checking
    if nfill < 0:
        raise ValueError('nfill = %d < 0' % nfill)
    if n_min > n_max:
        raise ValueError('n_min = %d > %d = n_max' % (n_min, n_max))
    if n_min <= 0:
        raise ValueError('n_min = %d <= 0' % n_min)

    if nfill == 0:
        # no fill at all
        return _fill_symmetric_info(tot_area, 0, tot_area, inc_sp=False,
                                    fill_on_edge=False, cyclic=False), False

    # check no solution
    sp_delta = 0 if cyclic else (-1 if fill_on_edge else 1)
    nsp = nfill + sp_delta
    if n_min * nfill + nsp * sp_min > tot_area:
        raise InsufficientAreaError('Cannot draw %d fill blocks with n_min = %d' % (nfill, n_min))

    # first, try drawing nfill blocks without block length constraint.
    # may throw exception if no solution
    info = _fill_symmetric_info(tot_area, nfill, sp_min, inc_sp=True,
                                fill_on_edge=fill_on_edge, cyclic=cyclic)
    bmin, bmax = _get_min_max_blk_len(info)
    if bmin < n_min:
        # could get here if cyclic = True, fill_on_edge = True, n_min is odd
        # in this case actually no solution
        raise FillTooSmallError('Cannot draw %d fill blocks with n_min = %d' % (nfill, n_min))
    if bmax <= n_max:
        # we satisfy block length constraint, just return
        return info, False

    # we broke maximum block length constraint, so we flip
    # space and fill to have better control on fill length
    if nsp == 0 and n_max != tot_area and n_max - 1 != tot_area:
        # we get here only if nfill = 1 and fill_on_edge is True.
        # In this case there's no way to draw only one fill and abut both edges
        raise NoFillAbutEdgeError('Cannot draw only one fill abutting both edges.')
    info = _fill_symmetric_info(tot_area, nsp, n_max, inc_sp=False,
                                fill_on_edge=not fill_on_edge, cyclic=cyclic)
    num_diff_sp = info[1][2]
    if num_diff_sp > 0 and n_min == n_max:
        # no solution with same fill length, but we must have same fill length everywhere.
        raise NoFillChoiceError('Cannot draw %d fill blocks with '
                                'n_min = n_max = %d' % (nfill, n_min))
    return info, True


def _fill_symmetric_info(tot_area, num_blk_tot, sp, inc_sp=True, fill_on_edge=True, cyclic=False):
    # type: (int, int, int, bool, bool, bool) -> Tuple[int, Tuple[Any, ...]]
    """Calculate symmetric fill information.

    This method computes fill information without generating fill interval list.  This makes
    it fast to explore various fill settings.  See fill_symmetric_helper() to see a description
    of the fill algorithm.

    Parameters
    ----------
    tot_area : int
        the fill area length.
    num_blk_tot : int
        total number of fill blocks to use.
    sp : int
        space between blocks.  We will try our best to keep this spacing constant.
    inc_sp : bool
        If True, then we use sp + 1 if necessary.  Otherwise, we use sp - 1
        if necessary.
    fill_on_edge : bool
        If True, we put fill blocks on area boundary.  Otherwise, we put space block on
        area boundary.
    cyclic : bool
        If True, we assume we're filling in a cyclic area (it wraps around).

    Returns
    -------
    fill_area : int
        total filled area.
    args : Tuple[Any, ...]
        input arguments to _fill_symmetric_interval()
    """
    # error checking
    if num_blk_tot < 0:
        raise ValueError('num_blk_tot = %d < 0' % num_blk_tot)

    adj_sp_sgn = 1 if inc_sp else -1
    if num_blk_tot == 0:
        # special case, no fill at all
        if sp == tot_area:
            return 0, (tot_area, tot_area, 0, tot_area, 0, 0, 0, 0, -1, tot_area, False, False)
        elif sp == tot_area - adj_sp_sgn:
            return 0, (tot_area, tot_area, 1, tot_area, 0, 0, 0, 0, -1, tot_area, False, False)
        else:
            raise EmptyRegionError('Cannot have empty region = %d with sp = %d' % (tot_area, sp))

    # determine the number of space blocks
    if cyclic:
        num_sp_tot = num_blk_tot
    else:
        if fill_on_edge:
            num_sp_tot = num_blk_tot - 1
        else:
            num_sp_tot = num_blk_tot + 1

    # compute total fill area
    fill_area = tot_area - num_sp_tot * sp

    # find minimum fill length
    blk_len, num_blk1 = divmod(fill_area, num_blk_tot)
    # find number of fill intervals
    if cyclic and fill_on_edge:
        # if cyclic and fill on edge, number of intervals = number of blocks + 1,
        # because the interval on the edge double counts.
        num_blk_interval = num_blk_tot + 1
    else:
        num_blk_interval = num_blk_tot

    # find space length on edge, if applicable
    num_diff_sp = 0
    sp_edge = sp
    if cyclic and not fill_on_edge and sp_edge % 2 == 1:
        # edge space must be even.  To fix, we convert space to fill
        num_diff_sp += 1
        sp_edge += adj_sp_sgn
        num_blk1 += -adj_sp_sgn
        fill_area += -adj_sp_sgn
        if num_blk1 == num_blk_tot:
            blk_len += 1
            num_blk1 = 0
        elif num_blk1 < 0:
            blk_len -= 1
            num_blk1 += num_blk_tot

    mid_blk_len = mid_sp_len = -1
    # now we have num_blk_tot blocks with length blk0.  We have num_blk1 fill units
    # remaining that we need to distribute to the fill blocks
    if num_blk_interval % 2 == 0:
        # we have even number of fill intervals, so we have a space block in the middle
        mid_sp_len = sp
        # test condition for cyclic and fill_on_edge is different than other cases
        test_val = num_blk1 + blk_len if cyclic and fill_on_edge else num_blk1
        if test_val % 2 == 1:
            # we cannot distribute remaining fill units evenly, have to convert to space
            num_diff_sp += 1
            mid_sp_len += adj_sp_sgn
            num_blk1 += -adj_sp_sgn
            fill_area += -adj_sp_sgn
            if num_blk1 == num_blk_tot:
                blk_len += 1
                num_blk1 = 0
            elif num_blk1 < 0:
                blk_len -= 1
                num_blk1 += num_blk_tot
        if num_blk1 % 2 == 1:
            # the only way we get here is if cyclic and fill_on_edge is True.
            # in this case, we need to add one to fill unit to account
            # for edge fill double counting.
            num_blk1 += 1

        # get number of half fill intervals
        m = num_blk_interval // 2
    else:
        # we have odd number of fill intervals, so we have a fill block in the middle
        mid_blk_len = blk_len
        if cyclic and fill_on_edge:
            # special handling for this case, because edge fill block must be even
            if blk_len % 2 == 0 and num_blk1 % 2 == 1:
                # assign one fill unit to middle block
                mid_blk_len += 1
                num_blk1 -= 1
            elif blk_len % 2 == 1:
                # edge fill block is odd; we need odd number of fill units so we can
                # correct this.
                if num_blk1 % 2 == 0:
                    # we increment middle fill block to get odd number of fill units
                    mid_blk_len += 1
                    num_blk1 -= 1
                    if num_blk1 < 0:
                        # we get here only if num_blk1 == 0.  This means middle blk
                        # borrow one unit from edge block.  So we set num_blk1 to
                        # num_blk_tot - 2 to make sure rest of the blocks are one
                        # larger than edge block.
                        blk_len -= 1
                        num_blk1 = num_blk_tot - 2
                    else:
                        # Add one to account for edge fill double counting.
                        num_blk1 += 1
                else:
                    # Add one to account for edge fill double counting.
                    num_blk1 += 1
        elif num_blk1 % 2 == 1:
            # assign one fill unit to middle block
            mid_blk_len += 1
            num_blk1 -= 1

        m = (num_blk_interval - 1) // 2

    if blk_len <= 0:
        raise InsufficientAreaError('Insufficent area; cannot draw fill with length <= 0.')

    # now we need to distribute the fill units evenly.  We do so using cumulative modding
    num_large = num_blk1 // 2
    num_small = m - num_large
    if cyclic and fill_on_edge:
        # if cyclic and fill is on the edge, we need to make sure left-most block is even length
        if blk_len % 2 == 0:
            blk1, blk0 = blk_len, blk_len + 1
            k = num_small
        else:
            blk0, blk1 = blk_len, blk_len + 1
            k = num_large
    else:
        # make left-most fill interval be the most frequenct fill length
        if num_large >= num_small:
            blk0, blk1 = blk_len, blk_len + 1
            k = num_large
        else:
            blk1, blk0 = blk_len, blk_len + 1
            k = num_small

    return fill_area, (tot_area, sp, num_diff_sp, sp_edge, blk0, blk1, k, m,
                       mid_blk_len, mid_sp_len, fill_on_edge, cyclic)


def _get_min_max_blk_len(fill_info):
    """Helper method to get minimum/maximum fill lengths used."""
    blk0, blk1, blkm = fill_info[1][4], fill_info[1][5], fill_info[1][8]
    if blkm < 0:
        blkm = blk0
    return min(blk0, blk1, blkm), max(blk0, blk1, blkm)


def fill_symmetric_interval(tot_area, sp, num_diff_sp, sp_edge, blk0, blk1, k, m, mid_blk_len,
                            mid_sp_len, fill_on_edge, cyclic, offset=0, invert=False):
    """Helper function, construct interval list from output of _fill_symmetric_info().

    num_diff_sp = number of space blocks that has length different than sp
    sp_edge = if cyclic and not fill on edge, the edge space length.
    m = number of half fill blocks.
    blk1 = length of left-most fill block.
    blk0 = the second possible fill block length.
    k = number of half fill blocks with length = blk1.
    mid_blk_len = if > 0, length of middle fill block.  This is either blk0 or blk1.
    """
    ans = []
    if cyclic:
        if fill_on_edge:
            marker = offset - blk1 // 2
        else:
            marker = offset - sp_edge // 2
    else:
        marker = offset
    cur_sum = 0
    prev_sum = 1
    for fill_idx in range(m):
        # determine current fill length from cumulative modding result
        if cur_sum <= prev_sum:
            cur_len = blk1
        else:
            cur_len = blk0

        cur_sp = sp_edge if fill_idx == 0 else sp
        # record fill/space interval
        if invert:
            if fill_on_edge:
                ans.append((marker + cur_len, marker + cur_sp + cur_len))
            else:
                ans.append((marker, marker + cur_sp))
        else:
            if fill_on_edge:
                ans.append((marker, marker + cur_len))
            else:
                ans.append((marker + cur_sp, marker + cur_sp + cur_len))

        marker += cur_len + cur_sp
        prev_sum = cur_sum
        cur_sum = (cur_sum + k) % m

    # add middle fill or space
    if mid_blk_len >= 0:
        # fill in middle
        if invert:
            if not fill_on_edge:
                # we have one more space block before reaching middle block
                cur_sp = sp_edge if m == 0 else sp
                ans.append((marker, marker + cur_sp))
            half_len = len(ans)
        else:
            # we don't want to replicate middle fill, so get half length now
            half_len = len(ans)
            if fill_on_edge:
                ans.append((marker, marker + mid_blk_len))
            else:
                cur_sp = sp_edge if m == 0 else sp
                ans.append((marker + cur_sp, marker + cur_sp + mid_blk_len))
    else:
        # space in middle
        if invert:
            if fill_on_edge:
                # the last space we added is wrong, we need to remove
                del ans[-1]
                marker -= sp
            # we don't want to replicate middle space, so get half length now
            half_len = len(ans)
            ans.append((marker, marker + mid_sp_len))
        else:
            # don't need to do anything if we're recording blocks
            half_len = len(ans)

    # now add the second half of the list
    shift = tot_area + offset * 2
    for idx in range(half_len - 1, -1, -1):
        start, stop = ans[idx]
        ans.append((shift - stop, shift - start))

    return ans, num_diff_sp


def fill_symmetric_helper(tot_area, num_blk_tot, sp, offset=0, inc_sp=True, invert=False,
                          fill_on_edge=True, cyclic=False):
    # type: (int, int, int, int, bool, bool, bool, bool) -> Tuple[List[Tuple[int, int]], int]
    """Helper method for all fill symmetric methods.

    This method fills an area with given number of fill blocks such that the space between
    blocks is equal to the given space.  Other fill_symmetric methods basically transpose
    the constraints into this problem, with the proper options.

    The solution has the following properties:

    1. it is symmetric about the center.
    2. it is as uniform as possible.
    3. it uses at most 3 consecutive values of fill lengths.
    4. it uses at most 2 consecutive values of space lengths.  If inc_sp is True,
       we use sp and sp + 1.  If inc_sp is False, we use sp - 1 and sp.  In addition,
       at most two space blocks have length different than sp.

    Here are all the scenarios that affect the number of different fill/space lengths:

    1. All spaces will be equal to sp under the following condition:
       i. cyclic is False, and num_blk_tot is odd.
       ii. cyclic is True, fill_on_edge is True, and num_blk_tot is even.
       iii. cyclic is True, fill_on_edge is False, sp is even, and num_blk_tot is odd.

       In particular, this means if you must have the same space between fill blocks, you
       can change num_blk_tot by 1.
    2. The only case where at most 2 space blocks have length different than sp is
       when cyclic is True, fill_on_edge is False, sp is odd, and num_blk_tot is even.
    3. In all other cases, at most 1 space block have legnth different than sp.
    4, The only case where at most 3 fill lengths are used is when cyclic is True,
       fill_on_edge is True, and num_blk_tot is even,

    Parameters
    ----------
    tot_area : int
        the fill area length.
    num_blk_tot : int
        total number of fill blocks to use.
    sp : int
        space between blocks.  We will try our best to keep this spacing constant.
    offset : int
        the starting coordinate of the area interval.
    inc_sp : bool
        If True, then we use sp + 1 if necessary.  Otherwise, we use sp - 1
        if necessary.
    invert : bool
        If True, we return space intervals instead of fill intervals.
    fill_on_edge : bool
        If True, we put fill blocks on area boundary.  Otherwise, we put space block on
        area boundary.
    cyclic : bool
        If True, we assume we're filling in a cyclic area (it wraps around).

    Returns
    -------
    ans : List[(int, int)]
        list of fill or space intervals.
    num_diff_sp : int
        number of space intervals with length different than sp.  This is an integer
        between 0 and 2.
    """
    fill_info = _fill_symmetric_info(tot_area, num_blk_tot, sp, inc_sp=inc_sp,
                                     fill_on_edge=fill_on_edge, cyclic=cyclic)

    _, args = fill_info
    return fill_symmetric_interval(*args, offset=offset, invert=invert)
