"""`Intervals`: select runs of consecutive indices along one axis, stored as runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from zarr_indexing._affine import checked_affine
from zarr_indexing.output_map import ArrayMap

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = ["Intervals"]

_INTP_MAX = int(np.iinfo(np.intp).max)


class Intervals:
    """Half-open intervals `[lo[i], hi[i])` along one axis, read in order.

    An orthogonal selector: `view.oindex[Intervals(lo, hi), :]` selects what
    `view.oindex[np.r_[lo[0]:hi[0], lo[1]:hi[1], ...], :]` does, without listing
    every index. Positions are not wrapped, so they must be non-negative.
    Overlapping and repeated intervals are allowed; empty ones are dropped.
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
    """First index of each non-empty interval."""

    hi: npt.NDArray[np.intp]
    """One past the last index of each non-empty interval."""

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
            if array.size and not np.issubdtype(array.dtype, np.integer):
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
        keep = lengths > 0  # boolean indexing copies, so the caller's arrays are not kept
        self._init(_read_only(lo_array[keep]), _read_only(hi_array[keep]), size)

    def _init(self, lo: npt.NDArray[np.intp], hi: npt.NDArray[np.intp], size: int) -> None:
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)
        object.__setattr__(self, "size", size)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Intervals is immutable")

    def __reduce__(self) -> tuple[object, tuple[object, object]]:
        """Rebuild through `__init__`, so a copy is read-only too."""
        return (Intervals, (self.lo, self.hi))

    def __repr__(self) -> str:
        return f"Intervals(lo={self.lo.tolist()}, hi={self.hi.tolist()})"

    def shifted(self, offset: int) -> Intervals:
        """The same intervals moved by `offset`."""
        moved = object.__new__(Intervals)
        moved._init(
            _read_only(checked_affine(offset, 1, self.lo)),
            _read_only(checked_affine(offset, 1, self.hi)),
            self.size,
        )
        return moved

    def indices(self) -> npt.NDArray[np.intp]:
        """Every selected index, in order: `np.r_[lo[0]:hi[0], lo[1]:hi[1], ...]`."""
        return concat_ranges(self.lo, self.hi)


def concat_ranges(lo: npt.NDArray[np.intp], hi: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
    """`np.r_[lo[0]:hi[0], lo[1]:hi[1], ...]` for `lo <= hi`."""
    lengths = hi - lo
    starts = np.repeat(lo - (np.cumsum(lengths) - lengths), lengths)
    return starts + np.arange(starts.size, dtype=np.intp)


def _read_only(array: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
    """`array` behind a read-only buffer, so its WRITEABLE flag cannot be re-enabled.

    Only for arrays nothing else holds: it does not copy.
    """
    view = memoryview(np.ascontiguousarray(array)).toreadonly()
    return np.frombuffer(view, dtype=np.intp).reshape(array.shape)


class IntervalArrayMap(ArrayMap):
    """An `ArrayMap` whose index array is `Intervals` along one input axis, stored as intervals.

    It is the same map as the `ArrayMap` of `intervals.indices()`: it compares,
    serializes and composes as that map, and builds the index array the first
    time an operation asks for it. Chunk planning and pickling use the intervals.
    """

    __slots__ = ("_index_array", "_shape", "intervals")

    intervals: Intervals
    """The selected indices, as intervals."""

    _shape: tuple[int, ...]
    _index_array: npt.NDArray[np.intp] | None

    def __init__(
        self, intervals: Intervals, axis: int, ndim: int, offset: int, stride: int
    ) -> None:
        shape = (1,) * axis + (intervals.size,) + (1,) * (ndim - axis - 1)
        object.__setattr__(self, "intervals", intervals)
        object.__setattr__(self, "_shape", shape)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "stride", stride)
        object.__setattr__(self, "_index_array", None)

    @property
    def shape(self) -> tuple[int, ...]:
        """The index array's shape, known without building it."""
        return self._shape

    @property
    def index_array(self) -> npt.NDArray[np.intp]:  # type: ignore[override]
        """The full index array, built on first use."""
        values = self._index_array
        if values is None:
            values = _read_only(self.intervals.indices().reshape(self._shape))
            object.__setattr__(self, "_index_array", values)
        return values

    def _with_affine(self, offset: int, stride: int) -> ArrayMap:
        """The same intervals under a different affine adjustment, still stored as intervals."""
        return IntervalArrayMap(
            self.intervals, self.dependency_axes[0], len(self._shape), offset, stride
        )

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:  # type: ignore[override]
        """Pickle the intervals, not the index array."""
        axis = self.dependency_axes[0]
        return (
            IntervalArrayMap,
            (self.intervals, axis, len(self._shape), self.offset, self.stride),
        )

    def __repr__(self) -> str:
        return (
            f"IntervalArrayMap({self.intervals!r}, shape={self._shape}, "
            f"offset={self.offset}, stride={self.stride})"
        )
