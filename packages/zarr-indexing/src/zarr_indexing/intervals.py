"""`Intervals` — select runs of consecutive indices along one axis, stored as runs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from zarr_indexing._affine import checked_affine
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

    __slots__ = ("hi", "lo", "size")

    lo: npt.NDArray[np.intp]
    """First index of each interval (read-only)."""

    hi: npt.NDArray[np.intp]
    """One past the last index of each interval (read-only)."""

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
        # Own immutable copies, as `ArrayMap` does, so the caller's arrays can
        # change without changing a transform built from them.
        self.lo, self.hi = (_frozen(checked_affine(0, 1, a)) for a in (lo_array, hi_array))
        # Exact Python integers: `hi - lo` and its sum can overflow np.intp.
        # ponytail: ~55 ms per 1e6 intervals; add an intp fast path when lo >= 0 if that shows up.
        lengths = self.hi.astype(object) - self.lo.astype(object)
        if (lengths < 0).any():
            raise ValueError("every interval needs lo <= hi")
        self.size = int(lengths.sum())
        if self.size > np.iinfo(np.intp).max:
            raise OverflowError(
                f"the intervals select {self.size} indices, more than np.intp holds"
            )

    def __repr__(self) -> str:
        return f"Intervals(lo={self.lo.tolist()}, hi={self.hi.tolist()})"

    def span(self) -> tuple[int, int] | None:
        """The smallest `lo` and largest `hi` of the non-empty intervals, or None if all are empty."""
        nonempty = self.hi > self.lo
        if not nonempty.any():
            return None
        return int(self.lo[nonempty].min()), int(self.hi[nonempty].max())

    def shifted(self, offset: int) -> Intervals:
        """The same intervals moved by `offset`."""
        return Intervals(checked_affine(offset, 1, self.lo), checked_affine(offset, 1, self.hi))

    def indices(self) -> npt.NDArray[np.intp]:
        """Every selected index, in order: `np.r_[lo[0]:hi[0], lo[1]:hi[1], ...]`."""
        lengths = self.hi - self.lo
        offsets = np.repeat(self.lo - (np.cumsum(lengths) - lengths), lengths)
        return np.asarray(offsets + np.arange(self.size, dtype=np.intp), dtype=np.intp)


def _frozen(array: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
    """A read-only copy backed by immutable bytes, so its WRITEABLE flag cannot be re-enabled."""
    return np.frombuffer(array.tobytes(), dtype=np.intp).reshape(array.shape)


class IntervalArrayMap(ArrayMap):
    """An `ArrayMap` whose index array is `Intervals` along one input axis, stored as intervals.

    It is the same map as the `ArrayMap` of `intervals.indices()`: it compares,
    serializes and composes as that map, and builds the index array the first
    time an operation asks for it. Chunk planning reads the intervals instead,
    so planning costs per interval rather than per index. Rebuilding it from its
    fields (`dataclasses.replace`, pickle) gives that plain `ArrayMap`.
    """

    __slots__ = ("_index_array", "_shape", "intervals")

    intervals: Intervals
    """The selected indices, as intervals."""

    _shape: tuple[int, ...]
    _index_array: npt.NDArray[np.intp] | None

    def __new__(
        cls,
        intervals: Intervals | None = None,
        shape: tuple[int, ...] = (),
        offset: int = 0,
        stride: int = 1,
        *,
        index_array: npt.NDArray[np.integer[Any]] | None = None,
    ) -> ArrayMap:
        # `dataclasses.replace` rebuilds from the dataclass fields, the first of
        # which is `index_array`; that gives the plain map this one equals.
        if index_array is not None:
            return ArrayMap(index_array, offset, stride)
        return object.__new__(cls)

    def __init__(
        self,
        intervals: Intervals | None = None,
        shape: tuple[int, ...] = (),
        offset: int = 0,
        stride: int = 1,
        *,
        index_array: npt.NDArray[np.integer[Any]] | None = None,
    ) -> None:
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
        cached = self._index_array
        if cached is None:
            cached = _frozen(self.intervals.indices().reshape(self._shape))
            object.__setattr__(self, "_index_array", cached)
        return cached

    def _with_affine(self, offset: int, stride: int) -> ArrayMap:
        """The same intervals under a different affine adjustment, still stored as intervals."""
        return IntervalArrayMap(self.intervals, self._shape, offset, stride)

    def __reduce__(self) -> tuple[object, tuple[object, int, int]]:
        """Pickle as the plain `ArrayMap` it equals."""
        return (ArrayMap, (self.index_array, self.offset, self.stride))

    def __repr__(self) -> str:
        return (
            f"IntervalArrayMap({self.intervals!r}, shape={self._shape}, "
            f"offset={self.offset}, stride={self.stride})"
        )
