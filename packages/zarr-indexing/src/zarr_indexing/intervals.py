"""`Intervals` — select runs of consecutive indices along one axis, stored as runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from zarr_indexing.output_map import ArrayMap

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = ["IntervalArrayMap", "Intervals"]


class Intervals:
    """Half-open intervals `[lo[i], hi[i])` along one axis, read in order.

    An orthogonal selector: `view.oindex[Intervals(lo, hi), :]` selects what
    `view.oindex[np.r_[lo[0]:hi[0], lo[1]:hi[1], ...], :]` does, without listing
    every index. Empty, overlapping and repeated intervals are allowed. The same
    `[lo, hi)` pairs are ndsel's `intervals`.

    Examples
    --------
    >>> from zarr_indexing import LazyArray
    >>> view = LazyArray.from_numpy(np.arange(10) * 10)
    >>> view.oindex[Intervals(lo=[7, 1], hi=[9, 3])].result()
    array([70, 80, 10, 20])
    """

    __slots__ = ("hi", "lo")

    lo: npt.NDArray[np.intp]
    """First index of each interval."""

    hi: npt.NDArray[np.intp]
    """One past the last index of each interval."""

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
        self.lo = lo_array.astype(np.intp)
        self.hi = hi_array.astype(np.intp)
        if (self.hi < self.lo).any():
            raise ValueError("every interval needs lo <= hi")

    def __repr__(self) -> str:
        return f"Intervals(lo={self.lo.tolist()}, hi={self.hi.tolist()})"

    @property
    def size(self) -> int:
        """How many indices the intervals select."""
        return int((self.hi - self.lo).sum())

    def shifted(self, offset: int) -> Intervals:
        """The same intervals moved by `offset`."""
        return Intervals(self.lo + offset, self.hi + offset)

    def indices(self) -> npt.NDArray[np.intp]:
        """Every selected index, in order: `np.r_[lo[0]:hi[0], lo[1]:hi[1], ...]`."""
        lengths = self.hi - self.lo
        offsets = np.repeat(self.lo - (np.cumsum(lengths) - lengths), lengths)
        return np.asarray(offsets + np.arange(self.size, dtype=np.intp), dtype=np.intp)


class IntervalArrayMap(ArrayMap):
    """An `ArrayMap` whose index array is `Intervals` along one input axis, stored as intervals.

    It is the same map as the `ArrayMap` of `intervals.indices()`: it compares,
    serializes and composes as that map, and builds the index array the first
    time an operation asks for it. Chunk planning reads the intervals instead,
    so planning costs per interval rather than per index.
    """

    __slots__ = ("_index_array", "axis", "intervals", "rank")

    intervals: Intervals
    """The selected indices, as intervals."""

    axis: int
    """The input axis the indices vary along."""

    rank: int
    """The input rank of the enclosing transform."""

    _index_array: npt.NDArray[np.intp] | None

    def __init__(
        self, intervals: Intervals, axis: int, rank: int, offset: int = 0, stride: int = 1
    ) -> None:
        object.__setattr__(self, "intervals", intervals)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "stride", stride)
        object.__setattr__(self, "_index_array", None)

    @property
    def shape(self) -> tuple[int, ...]:
        """The index array's shape, known without building it."""
        return (1,) * self.axis + (self.intervals.size,) + (1,) * (self.rank - self.axis - 1)

    @property
    def index_array(self) -> npt.NDArray[np.intp]:  # type: ignore[override]
        """The full index array, built on first use."""
        cached = self._index_array
        if cached is None:
            values = self.intervals.indices()
            cached = np.frombuffer(values.tobytes(), dtype=np.intp).reshape(self.shape)
            object.__setattr__(self, "_index_array", cached)
        return cached

    def _with_affine(self, offset: int, stride: int) -> ArrayMap:
        """The same intervals under a different affine adjustment, still stored as intervals."""
        return IntervalArrayMap(self.intervals, self.axis, self.rank, offset, stride)

    def __reduce__(self) -> tuple[object, tuple[object, int, int]]:
        """Pickle as the plain `ArrayMap` it equals."""
        return (ArrayMap, (self.index_array, self.offset, self.stride))

    def __repr__(self) -> str:
        return (
            f"IntervalArrayMap({self.intervals!r}, axis={self.axis}, rank={self.rank}, "
            f"offset={self.offset}, stride={self.stride})"
        )
