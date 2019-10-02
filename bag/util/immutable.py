"""This module defines various immutable and hashable data types.
"""

from __future__ import annotations

from typing import TypeVar, Any, Generic, Dict, Iterable, Tuple, Union, Optional, overload

import sys
import bisect
from collections import Hashable, Mapping, Sequence

T = TypeVar('T')
U = TypeVar('U')
ImmutableType = Union[None, Hashable, Tuple[Hashable, ...]]


def combine_hash(a: int, b: int) -> int:
    """Combine the two given hash values.

    Parameter
    ---------
    a : int
        the first hash value.
    b : int
        the second hash value.

    Returns
    -------
    hash : int
        the combined hash value.
    """
    # algorithm taken from boost::hash_combine
    return sys.maxsize & (a ^ (b + 0x9e3779b9 + (a << 6) + (a >> 2)))


class ImmutableList(Hashable, Sequence, Generic[T]):
    """An immutable homogeneous list."""

    def __init__(self, values: Optional[Sequence[T]] = None) -> None:
        if values is None:
            self._content = []
            self._hash = 0
        elif isinstance(values, ImmutableList):
            self._content = values._content
            self._hash = values._hash
        else:
            self._content = values
            self._hash = 0
            for v in values:
                self._hash = combine_hash(self._hash, hash(v))

    @classmethod
    def sequence_equal(cls, a: Sequence[T], b: Sequence[T]) -> bool:
        if len(a) != len(b):
            return False
        for av, bv in zip(a, b):
            if av != bv:
                return False
        return True

    def __repr__(self) -> str:
        return repr(self._content)

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, ImmutableList) and self._hash == other._hash and
                self.sequence_equal(self._content, other._content))

    def __hash__(self) -> int:
        return self._hash

    def __bool__(self) -> bool:
        return len(self) > 0

    def __len__(self) -> int:
        return len(self._content)

    def __iter__(self) -> Iterable[T]:
        return iter(self._content)

    @overload
    def __getitem__(self, idx: int) -> T: ...
    @overload
    def __getitem__(self, idx: slice) -> ImmutableList[T]: ...

    def __getitem__(self, idx) -> T:
        if isinstance(idx, int):
            return self._content[idx]
        return ImmutableList(self._content[idx])

    def __contains__(self, val: Any) -> bool:
        return val in self._content


class ImmutableSortedDict(Hashable, Mapping, Generic[T, U]):
    """An immutable dictionary with sorted keys."""

    def __init__(self,
                 table: Optional[Mapping[T, Any]] = None) -> None:
        if table is not None:
            if isinstance(table, ImmutableSortedDict):
                self._keys = table._keys
                self._vals = table._vals
                self._hash = table._hash
            else:
                self._keys = ImmutableList(sorted(table.keys()))
                self._vals = ImmutableList([to_immutable(table[k]) for k in self._keys])
                self._hash = combine_hash(hash(self._keys), hash(self._vals))
        else:
            self._keys = ImmutableList([])
            self._vals = ImmutableList([])
            self._hash = combine_hash(hash(self._keys), hash(self._vals))

    def __repr__(self) -> str:
        return repr(list(zip(self._keys, self._vals)))

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, ImmutableSortedDict) and
                self._hash == other._hash and
                self._keys == other._keys and
                self._vals == other._vals)

    def __hash__(self) -> int:
        return self._hash

    def __bool__(self) -> bool:
        return len(self) > 0

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self) -> Iterable[T]:
        return iter(self._keys)

    def __contains__(self, item: Any) -> bool:
        idx = bisect.bisect_left(self._keys, item)
        return idx != len(self._keys) and self._keys[idx] == item

    def __getitem__(self, item: T) -> U:
        idx = bisect.bisect_left(self._keys, item)
        if idx == len(self._keys) or self._keys[idx] != item:
            raise KeyError('Key not found: {}'.format(item))
        return self._vals[idx]

    def get(self, item: T, default: Optional[U] = None) -> Optional[U]:
        idx = bisect.bisect_left(self._keys, item)
        if idx == len(self._keys) or self._keys[idx] != item:
            return default
        return self._vals[idx]

    def keys(self) -> Iterable[T]:
        return iter(self._keys)

    def values(self) -> Iterable[U]:
        return iter(self._vals)

    def items(self) -> Iterable[Tuple[T, U]]:
        return zip(self._keys, self._vals)

    def copy(self, append: Optional[Dict[T, Any]] = None) -> ImmutableSortedDict[T, U]:
        if append is None:
            return self.__class__(self)
        else:
            tmp = self.to_dict()
            tmp.update(append)
            return self.__class__(tmp)

    def to_dict(self) -> Dict[T, U]:
        return dict(zip(self._keys, self._vals))


Param = ImmutableSortedDict[str, Any]


def to_immutable(obj: Any) -> ImmutableType:
    """Convert the given Python object into an immutable type."""
    if obj is None:
        return obj
    if isinstance(obj, Hashable):
        # gets around cases of tuple of un-hashable types.
        try:
            hash(obj)
            return obj
        except TypeError:
            pass
    if isinstance(obj, tuple):
        return tuple((to_immutable(v) for v in obj))
    if isinstance(obj, list):
        return ImmutableList([to_immutable(v) for v in obj])
    if isinstance(obj, set):
        return ImmutableList([to_immutable(v) for v in sorted(obj)])
    if isinstance(obj, dict):
        return ImmutableSortedDict(obj)

    raise ValueError('Cannot convert the following object to immutable type: {}'.format(obj))
