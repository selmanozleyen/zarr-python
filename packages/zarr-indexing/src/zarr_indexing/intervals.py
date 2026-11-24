"""The `Intervals` selector, and the `ArrayMap` subclass that stores it as intervals."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from zarr_indexing._affine import checked_affine
from zarr_indexing.output_map import ArrayMap, frozen_copy

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = ["Intervals"]

_INTP_MAX = int(np.iinfo(np.intp).max)


class Intervals:
    """Half-open intervals `[lo[i], hi[i])` along one axis, read in order.

    An orthogonal selector: `view.oindex[Intervals(lo, hi), :]` selects what
    `view.oindex[np.r_[lo[0]:hi[0], lo[1]:hi[1], ...], :]` does, without listing
    every index. Positions are not wrapped, so they must be non-negative.
    Overlapping and repeated intervals are allowed; empty ones are dropped and
    touching ones merged.
    The `[lo, hi)` pairs are ndsel's `intervals`. Immutable.

    Examples
    --------
    >>> from zarr_indexing import LazyArray
    >>> view = LazyArray.from_numpy(np.arange(10) * 10)
    >>> view.oindex[Intervals(lo=[7, 1], hi=[9, 3])].result()
    array([70, 80, 10, 20])
    """

    __slots__ = ("hi", "lo", "size")

    lo: npt.NDArray[np.intp]
    """First index of each interval, after dropping empty ones and merging touching ones."""

    hi: npt.NDArray[np.intp]
    """One past the last index of each interval, likewise."""

    size: int
    """How many indices the intervals select."""

    def __init__(self, lo: npt.ArrayLike, hi: npt.ArrayLike) -> None:
        lo_array, hi_array = np.asarray(lo), np.asarray(hi)
        if lo_array.ndim != 1 or lo_array.shape != hi_array.shape:
            raise ValueError(
                f"lo and hi must be 1-D and equally long, got shapes {lo_array.shape} "
                f"and {hi_array.shape}"
            )
        for name, array in (("lo", lo_array), ("hi", hi_array)):
            if array.size > 0 and not np.issubdtype(array.dtype, np.integer):
                raise TypeError(f"{name} must have an integer dtype, got {array.dtype}")
        lo_array, hi_array = checked_affine(0, 1, lo_array), checked_affine(0, 1, hi_array)
        if (hi_array < lo_array).any():
            raise ValueError("every interval needs lo <= hi")
        # With hi >= lo the difference fits uint64 exactly; the sum is exact in
        # Python integers when it could overflow.
        lengths = hi_array.astype(np.uint64) - lo_array.astype(np.uint64)
        n = lengths.size
        fits = n == 0 or int(lengths.max()) <= _INTP_MAX // n
        size = int(lengths.sum()) if fits else sum(lengths.tolist())
        if size > _INTP_MAX:
            raise OverflowError(f"the intervals select {size} indices, more than np.intp holds")
        keep = lengths > 0
        lo_array, hi_array = lo_array[keep], hi_array[keep]
        # Merge touching intervals: [a, b) then [b, c) is [a, c).
        head = np.ones(lo_array.size, dtype=bool)
        head[1:] = lo_array[1:] != hi_array[:-1]
        tail = np.ones(lo_array.size, dtype=bool)
        tail[:-1] = head[1:]
        object.__setattr__(self, "lo", frozen_copy(lo_array[head]))
        object.__setattr__(self, "hi", frozen_copy(hi_array[tail]))
        object.__setattr__(self, "size", size)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Intervals is immutable")

    def __reduce__(self) -> tuple[object, tuple[object, object]]:
        """Rebuild through `__init__`, so a copy is read-only too."""
        return (Intervals, (self.lo, self.hi))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Intervals):
            return NotImplemented
        return bool(np.array_equal(self.lo, other.lo) and np.array_equal(self.hi, other.hi))

    def __hash__(self) -> int:
        return hash((self.lo.tobytes(), self.hi.tobytes()))

    def __repr__(self) -> str:
        return f"Intervals(lo={self.lo.tolist()}, hi={self.hi.tolist()})"

    def shifted(self, offset: int) -> Intervals:
        """The same intervals moved by `offset`."""
        return Intervals(checked_affine(offset, 1, self.lo), checked_affine(offset, 1, self.hi))

    def indices(self) -> npt.NDArray[np.intp]:
        """Every selected index, in order: `np.r_[lo[0]:hi[0], lo[1]:hi[1], ...]`."""
        return concat_ranges(self.lo, self.hi)


def concat_ranges(lo: npt.NDArray[np.intp], hi: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
    """`np.r_[lo[0]:hi[0], lo[1]:hi[1], ...]` for `lo <= hi`."""
    lengths = hi - lo
    starts = np.repeat(lo - (np.cumsum(lengths) - lengths), lengths)
    return starts + np.arange(starts.size, dtype=np.intp)


def index_extremes(m: ArrayMap) -> tuple[int, int]:
    """The smallest and largest entry of a non-empty map's index array, without building it."""
    if isinstance(m, IntervalArrayMap):
        return int(m.intervals.lo.min()), int(m.intervals.hi.max()) - 1
    return int(m.index_array.min()), int(m.index_array.max())


class IntervalArrayMap(ArrayMap):
    """An `ArrayMap` whose index array is `Intervals` along one input axis, stored as intervals.

    It is the same map as the `ArrayMap` of `intervals.indices()`: it compares,
    serializes and composes as that map, and builds the index array the first
    time an operation asks for it. Chunk planning and pickling use the intervals.
    """

    __slots__ = ("_index_array", "_shape", "axis", "intervals")

    intervals: Intervals
    """The selected indices, as intervals."""

    axis: int
    """The input axis the indices vary along."""

    _shape: tuple[int, ...]
    _index_array: npt.NDArray[np.intp] | None

    def __init__(
        self, intervals: Intervals, axis: int, ndim: int, offset: int, stride: int
    ) -> None:
        shape = (1,) * axis + (intervals.size,) + (1,) * (ndim - axis - 1)
        object.__setattr__(self, "intervals", intervals)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "_shape", shape)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "stride", stride)
        object.__setattr__(self, "_index_array", None)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("IntervalArrayMap is immutable")

    @property
    def shape(self) -> tuple[int, ...]:
        """The index array's shape, known without building it."""
        return self._shape

    @property
    def index_array(self) -> npt.NDArray[np.intp]:  # type: ignore[override]
        """The full index array, built on first use."""
        values = self._index_array
        if values is None:
            values = frozen_copy(self.intervals.indices().reshape(self._shape))
            object.__setattr__(self, "_index_array", values)
        return values

    def _with_affine(self, offset: int, stride: int) -> ArrayMap:
        """The same intervals under a different affine adjustment, still stored as intervals."""
        return IntervalArrayMap(self.intervals, self.axis, len(self._shape), offset, stride)

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:  # type: ignore[override]
        """Pickle the intervals, not the index array."""
        args = (self.intervals, self.axis, len(self._shape), self.offset, self.stride)
        return (IntervalArrayMap, args)

    def __repr__(self) -> str:
        return (
            f"IntervalArrayMap({self.intervals!r}, shape={self._shape}, "
            f"offset={self.offset}, stride={self.stride})"
        )
