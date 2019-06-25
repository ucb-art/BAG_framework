# -*- coding: utf-8 -*-

"""This module provides data structure that keeps track of intervals.
"""

from typing import List, Optional, Tuple, Any, Iterable, Generator

import bisect


class IntervalSet(object):
    """A data structure that keeps track of disjoint 1D integer intervals.

    Each interval has a value associated with it.  If not specified, the value defaults to None.

    Parameters
    ----------
    intv_list : Optional[Iterable[Tuple[int, int]]]
        the sorted initial interval list.
    val_list : Optional[Iterable[Any]]
        the initial values list.
    """

    def __init__(self, intv_list=None, val_list=None):
        # type: (Optional[Iterable[Tuple[int, int]]], Optional[Iterable[Any]]) -> None
        self._start_list = []  # type: List[int]
        self._end_list = []  # type: List[int]
        if intv_list is None:
            self._val_list = []  # type: List[Any]
        else:
            for v0, v1 in intv_list:
                self._start_list.append(v0)
                self._end_list.append(v1)
            if val_list is None:
                self._val_list = [None] * len(self._start_list)
            else:
                self._val_list = list(val_list)

    def __contains__(self, key):
        # type: (Tuple[int, int]) -> bool
        """Returns True if this IntervalSet contains the given interval.

        Parameters
        ----------
        key : Tuple[int, int]
            the interval to test.

        Returns
        -------
        contains : bool
            True if this IntervalSet contains the given interval.
        """
        idx = self._get_first_overlap_idx(key)
        return idx >= 0 and key[0] == self._start_list[idx] and key[1] == self._end_list[idx]

    def __getitem__(self, intv):
        # type: (Tuple[int, int]) -> Any
        """Returns the value associated with the given interval.

        Raises KeyError if the given interval is not in this IntervalSet.

        Parameters
        ----------
        intv : Tuple[int, int]
            the interval to query.

        Returns
        -------
        val : Any
            the value associated with the given interval.
        """
        idx = self._get_first_overlap_idx(intv)
        if idx < 0 or intv[0] != self._start_list[idx] or intv[1] != self._end_list[idx]:
            raise KeyError('Invalid interval: %s' % repr(intv))
        return self._val_list[idx]

    def __setitem__(self, intv, value):
        # type: (Tuple[int, int], Any) -> None
        """Update the value associated with the given interval.

        Raises KeyError if the given interval is not in this IntervalSet.

        Parameters
        ----------
        intv : Tuple[int, int]
            the interval to update.
        value : Any
            the new value.
        """
        idx = self._get_first_overlap_idx(intv)
        if idx < 0:
            self.add(intv, value)
        elif intv[0] != self._start_list[idx] or intv[1] != self._end_list[idx]:
            raise KeyError('Invalid interval: %s' % repr(intv))
        else:
            self._val_list[idx] = value

    def __iter__(self):
        # type: () -> Iterable[Tuple[int, int]]
        """Iterates over intervals in this IntervalSet in increasing order.

        Yields
        ------
        intv : Tuple[int, int]
            the next interval.
        """
        return zip(self._start_list, self._end_list)

    def __len__(self):
        # type: () -> int
        """Returns the number of intervals in this IntervalSet.

        Returns
        -------
        length : int
            number of intervals in this set.
        """
        return len(self._start_list)

    def get_start(self):
        # type: () -> int
        """Returns the start of the first interval.

        Returns
        -------
        start : int
            the start of the first interval.
        """
        return self._start_list[0]

    def get_end(self):
        # type: () -> int
        """Returns the end of the last interval.

        Returns
        -------
        end : int
            the end of the last interval.
        """
        return self._end_list[-1]

    def get_interval(self, idx):
        # type: (int) -> Tuple[int, int]
        if idx < 0:
            idx += len(self._start_list)
        if idx < 0:
            raise IndexError('Invalid index: %d' % idx)
        if idx >= len(self._start_list):
            raise IndexError('Invalid index: %d' % idx)

        return self._start_list[idx], self._end_list[idx]

    def copy(self):
        # type: () -> IntervalSet
        """Create a copy of this interval set.

        Returns
        -------
        intv_set : IntervalSet
            a copy of this IntervalSet.
        """
        return IntervalSet(intv_list=list(zip(self._start_list, self._end_list)),
                           val_list=self._val_list)

    def _get_first_overlap_idx(self, intv, abut=False):
        # type: (Tuple[int, int], bool) -> int
        """Returns the index of the first interval that overlaps with the given interval.

        Parameters
        ----------
        intv : Tuple[int, int]
            the given interval.
        abut : bool
            True to return abutted interval too.

        Returns
        -------
        idx : int
            the index of the overlapping interval.  If no overlapping intervals are
            found, -(idx + 1) is returned, where idx is the index to insert the interval.
        """
        start, end = intv
        if not self._start_list:
            return -1
        # find the smallest start index greater than start
        idx = bisect.bisect_right(self._start_list, start)
        if idx == 0:
            # all interval's starting point is greater than start
            test = self._start_list[0]
            return 0 if test < end or (abut and test == end) else -1

        # interval where start index is less than or equal to start
        test_idx = idx - 1
        test = self._end_list[test_idx]
        if start < test or (abut and start == test):
            # start is covered by the interval; overlaps.
            return test_idx
        elif idx < len(self._start_list) and \
                (self._start_list[idx] < end or (abut and self._start_list[idx] == end)):
            # _start_list[idx] covered by interval.
            return idx
        else:
            # no overlap interval found
            return -(idx + 1)

    def _get_last_overlap_idx(self, intv, abut=False):
        # type: (Tuple[int, int], bool) -> int
        """Returns the index of the last interval that overlaps with the given interval.

        Parameters
        ----------
        intv : Tuple[int, int]
            the given interval.
        abut : bool
            True to return abutted interval too.

        Returns
        -------
        idx : int
            the index of the overlapping interval.  If no overlapping intervals are
            found, -(idx + 1) is returned, where idx is the index to insert the interval.
        """
        start, end = intv
        if not self._start_list:
            return -1
        # find the smallest start index greater than end
        idx = bisect.bisect_right(self._start_list, end)
        if idx == 0:
            # all interval's starting point is greater than end
            return -1

        # interval where start index is less than or equal to end
        test_idx = idx - 1
        test = self._end_list[test_idx]
        if test > start or (abut and test == start):
            return test_idx
        return -(idx + 1)

    def has_overlap(self, intv):
        # type: (Tuple[int, int]) -> bool
        """Returns True if the given interval overlaps at least one interval in this set.

        Parameters
        ----------
        intv : Tuple[int, int]
            the given interval.

        Returns
        -------
        has_overlap : bool
            True if there is at least one interval in this set that overlaps with the given one.
        """
        return self._get_first_overlap_idx(intv) >= 0

    def has_single_cover(self, intv):
        # type: (Tuple[int, int]) -> bool
        """Returns True if the given interval is completed covered by a single interval."""
        idx = self._get_first_overlap_idx(intv)
        if idx < 0:
            return False
        return self._start_list[idx] <= intv[0] and self._end_list[idx] >= intv[1]

    def remove(self, intv):
        # type: (Tuple[int, int]) -> bool
        """Removes the given interval from this IntervalSet.

        Parameters
        ----------
        intv : Tuple[int, int]
            the interval to remove.

        Returns
        -------
        success : bool
            True if the given interval is found and removed.  False otherwise.
        """
        idx = self._get_first_overlap_idx(intv)
        if idx < 0:
            return False
        if intv[0] == self._start_list[idx] and intv[1] == self._end_list[idx]:
            del self._start_list[idx]
            del self._end_list[idx]
            del self._val_list[idx]
            return True
        return False

    def get_intersection(self, other):
        # type: (IntervalSet) -> IntervalSet
        """Returns the intersection of two IntervalSets.

        the new IntervalSet will have all values set to None.

        Parameters
        ----------
        other : IntervalSet
            the other IntervalSet.

        Returns
        -------
        intersection : IntervalSet
            a new IntervalSet containing all intervals present in both sets.
        """
        idx1 = idx2 = 0
        len1 = len(self._start_list)
        len2 = len(other._start_list)
        intvs = []
        while idx1 < len1 and idx2 < len2:
            intv1 = self._start_list[idx1], self._end_list[idx1]
            intv2 = other._start_list[idx2], other._end_list[idx2]
            test = max(intv1[0], intv2[0]), min(intv1[1], intv2[1])
            if test[1] > test[0]:
                intvs.append(test)
            if intv1[1] < intv2[1]:
                idx1 += 1
            elif intv2[1] < intv1[1]:
                idx2 += 1
            else:
                idx1 += 1
                idx2 += 1

        return IntervalSet(intv_list=intvs)

    def get_complement(self, total_intv):
        # type: (Tuple[int, int]) -> IntervalSet
        """Returns a new IntervalSet that's the complement of this one.

        The new IntervalSet will have all values set to None.

        Parameters
        ----------
        total_intv : Tuple[int, int]
            the universal interval.  All intervals in this IntervalSet must be as subinterval
            of the universal interval.

        Returns
        -------
        complement : IntervalSet
            the complement of this IntervalSet.
        """
        return IntervalSet(intv_list=self.complement_iter(total_intv))

    def complement_iter(self, total_intv):
        # type: (Tuple[int, int]) -> Generator[Tuple[int, int], None, None]
        """Iterate over all intervals that;s the complement of this one."""
        if not self._start_list:
            yield total_intv
        elif self._start_list[0] < total_intv[0] or total_intv[1] < self._end_list[-1]:
            raise ValueError('The given interval [{0}, {1}) is '
                             'not a valid universal interval'.format(*total_intv))
        else:
            marker = total_intv[0]
            for start, end in zip(self._start_list, self._end_list):
                if marker < start:
                    yield marker, start
                marker = end

            if marker < total_intv[1]:
                yield marker, total_intv[1]

    def remove_all_overlaps(self, intv):
        # type: (Tuple[int, int]) -> None
        """Remove all intervals in this set that overlaps with the given interval.

        Parameters
        ----------
        intv : Tuple[int, int]
            the given interval
        """
        sidx = self._get_first_overlap_idx(intv)
        if sidx >= 0:
            eidx = self._get_last_overlap_idx(intv) + 1
            del self._start_list[sidx:eidx]
            del self._end_list[sidx:eidx]
            del self._val_list[sidx:eidx]

    def add(self, intv, val=None, merge=False, abut=False):
        # type: (Tuple[int, int], Any, bool, bool) -> bool
        """Adds the given interval to this IntervalSet.

        Can only add interval that does not overlap with any existing ones, unless merge is True.

        Parameters
        ----------
        intv : Tuple[int, int]
            the interval to add.
        val : Any
            the value associated with the given interval.
        merge : bool
            If true, the given interval will be merged with any existing intervals
            that overlaps with it.  The merged interval will have the given value.
        abut : bool
            True to count merge abutting intervals.

        Returns
        -------
        success : bool
            True if the given interval is added.
        """
        abut = abut and merge
        bidx = self._get_first_overlap_idx(intv, abut=abut)
        if bidx >= 0:
            if not merge:
                return False
            eidx = self._get_last_overlap_idx(intv, abut=abut)
            new_start = min(self._start_list[bidx], intv[0])
            new_end = max(self._end_list[eidx], intv[1])
            del self._start_list[bidx:eidx + 1]
            del self._end_list[bidx:eidx + 1]
            del self._val_list[bidx:eidx + 1]
            self._start_list.insert(bidx, new_start)
            self._end_list.insert(bidx, new_end)
            self._val_list.insert(bidx, val)
            return True
        else:
            # insert interval
            idx = -bidx - 1
            self._start_list.insert(idx, intv[0])
            self._end_list.insert(idx, intv[1])
            self._val_list.insert(idx, val)
            return True

    def subtract(self, intv):
        # type: (Tuple[int, int]) -> List[Tuple[int, int]]
        """Subtract the given interval from this IntervalSet.

        Parameters
        ----------
        intv : Tuple[int, int]
            the interval to subtract.

        Returns
        -------
        remaining_intvs : List[Tuple[int, int]]
            intervals created from subtraction.
        """
        bidx = self._get_first_overlap_idx(intv)
        insert_intv = []
        if bidx >= 0:
            eidx = self._get_last_overlap_idx(intv)
            insert_val = []
            if self._start_list[bidx] < intv[0]:
                insert_intv.append((self._start_list[bidx], intv[0]))
                insert_val.append(self._val_list[bidx])
            if intv[1] < self._end_list[eidx]:
                insert_intv.append((intv[1], self._end_list[eidx]))
                insert_val.append(self._val_list[eidx])
            del self._start_list[bidx:eidx + 1]
            del self._end_list[bidx:eidx + 1]
            del self._val_list[bidx:eidx + 1]
            insert_idx = bidx
            for (new_start, new_end), val in zip(insert_intv, insert_val):
                self._start_list.insert(insert_idx, new_start)
                self._end_list.insert(insert_idx, new_end)
                self._val_list.insert(insert_idx, val)
                insert_idx += 1

        return insert_intv

    def items(self):
        # type: () -> Iterable[Tuple[Tuple[int, int], Any]]
        """Iterates over intervals and values in this IntervalSet

        The intervals are returned in increasing order.

        Yields
        ------
        intv : Tuple[Tuple[int, int]
            the interval.
        val : Any
            the value associated with the interval.
        """
        return zip(self.__iter__(), self._val_list)

    def intervals(self):
        # type: () -> Iterable[Tuple[int, int]]
        """Iterates over intervals in this IntervalSet

        The intervals are returned in increasing order.

        Yields
        ------
        intv : Tuple[int, int]
            the interval.
        """
        return self.__iter__()

    def values(self):
        # type: () -> Iterable[Any]
        """Iterates over values in this IntervalSet

        The values correspond to intervals in increasing order.

        Yields
        ------
        val : Any
            the value.
        """
        return self._val_list.__iter__()

    def overlap_items(self, intv):
        # type: (Tuple[int, int]) -> Generator[Tuple[Tuple[int, int], Any], None, None]
        """Iterates over intervals and values overlapping the given interval.

        Parameters
        ----------
        intv : Tuple[int, int]
            the interval.

        Yields
        -------
        ovl_intv : Tuple[int, int]
            the overlapping interval.
        val : Any
            value associated with ovl_intv.
        """
        sidx = self._get_first_overlap_idx(intv)
        if sidx >= 0:
            eidx = self._get_last_overlap_idx(intv) + 1
            for idx in range(sidx, eidx):
                yield (self._start_list[idx], self._end_list[idx]), self._val_list[idx]

    def overlap_intervals(self, intv):
        # type: (Tuple[int, int]) -> Generator[Tuple[int, int], None, None]
        """Iterates over intervals overlapping the given interval.

        Parameters
        ----------
        intv : Tuple[int, int]
            the interval.

        Yields
        -------
        ovl_intv : Tuple[int, int]
            the overlapping interval.
        """
        sidx = self._get_first_overlap_idx(intv)
        if sidx >= 0:
            eidx = self._get_last_overlap_idx(intv) + 1
            for idx in range(sidx, eidx):
                yield self._start_list[idx], self._end_list[idx]

    def overlap_values(self, intv):
        # type: (Tuple[int, int]) -> Generator[Any, None, None]
        """Iterates over values of intervals overlapping the given interval.

        Parameters
        ----------
        intv : Tuple[int, int]
            the interval.

        Yields
        -------
        ovl_intv : Tuple[int, int]
            the overlapping interval.
        """
        sidx = self._get_first_overlap_idx(intv)
        if sidx >= 0:
            eidx = self._get_last_overlap_idx(intv) + 1
            for idx in range(sidx, eidx):
                yield self._val_list[idx]

    def get_first_overlap_item(self, intv):
        # type: (Tuple[int, int]) -> Optional[Tuple[Tuple[int, int], Any]]
        """Returns the first item with interval that overlaps the given one."""
        idx = self._get_first_overlap_idx(intv)
        if idx < 0:
            return None
        return (self._start_list[idx], self._end_list[idx]), self._val_list[idx]

    def transform(self, scale=1, shift=0):
        # type: (int, int) -> IntervalSet
        """Return a new IntervalSet under the given transformation.

        Parameters
        ----------
        scale : int
            multiple all interval values by this scale.  Either 1 or -1.
        shift : int
            add this amount to all intervals.

        Returns
        -------
        intv_set : IntervalSet
            the transformed IntervalSet.
        """
        if scale < 0:
            new_start = [-v + shift for v in reversed(self._end_list)]
            new_end = [-v + shift for v in reversed(self._start_list)]
            new_val = list(reversed(self._val_list))
        else:
            new_start = [v + shift for v in self._start_list]
            new_end = [v + shift for v in self._end_list]
            new_val = list(self._val_list)

        result = self.__class__.__new__(self.__class__)
        result._start_list = new_start
        result._end_list = new_end
        result._val_list = new_val

        return result
