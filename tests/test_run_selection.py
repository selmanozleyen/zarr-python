from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import zarr
from zarr.abc.codec import CodecPipeline
from zarr.core.buffer import default_buffer_prototype
from zarr.core.indexing import RunIndexer
from zarr.core.sync import sync
from zarr.errors import BoundsCheckError


def expected(values: np.ndarray, starts: list[int], lengths: list[int]) -> np.ndarray:
    parts = [values[s : s + n] for s, n in zip(starts, lengths, strict=True)]
    return np.concatenate([*parts, values[:0]])


@pytest.fixture(params=["chunked", "sharded"])
def arr_1d(request: pytest.FixtureRequest) -> tuple[zarr.Array[Any], np.ndarray]:
    values = np.arange(100, dtype="i4")
    shards = (20,) if request.param == "sharded" else None
    a = zarr.create_array(store={}, shape=values.shape, chunks=(4,), shards=shards, dtype="i4")
    a[:] = values
    return a, values


CASES = [
    ([0], [100]),  # the whole array
    ([10, 30, 50], [5, 1, 30]),  # the last crosses several chunks and a shard
    ([50, 10, 50], [7, 3, 7]),  # unordered and repeated
    ([10, 12], [5, 5]),  # overlapping
    ([10, 15, 20], [5, 5, 5]),  # touching: one read
    ([3, 7], [0, 2]),  # an empty range among others
    ([], []),
    ([99], [1]),
]


@pytest.mark.parametrize(("starts", "lengths"), CASES)
def test_matches_numpy(
    arr_1d: tuple[zarr.Array[Any], np.ndarray], starts: list[int], lengths: list[int]
) -> None:
    a, values = arr_1d
    assert_array_equal(a.get_range_selection(starts, lengths), expected(values, starts, lengths))


def test_trailing_axes_whole() -> None:
    values = np.arange(20 * 6, dtype="f8").reshape(20, 6)
    a = zarr.create_array(store={}, shape=values.shape, chunks=(4, 4), dtype="f8")
    a[:] = values
    starts, lengths = [15, 2, 3], [4, 3, 1]
    got = a.get_range_selection(starts, lengths)
    assert got.shape == (8, 6)
    assert_array_equal(got, expected(values, starts, lengths))


def test_unwritten_chunks_read_as_fill() -> None:
    a = zarr.create_array(store={}, shape=(40,), chunks=(4,), dtype="i2", fill_value=7)
    a[:6] = np.arange(6)
    assert_array_equal(a.get_range_selection([4, 30], [4, 2]), [4, 5, 7, 7, 7, 7])


def test_into_a_buffer(arr_1d: tuple[zarr.Array[Any], np.ndarray]) -> None:
    a, values = arr_1d
    out = default_buffer_prototype().nd_buffer.empty(shape=(6,), dtype=np.dtype("i4"))
    a.get_range_selection([40, 3], [4, 2], out=out)
    assert_array_equal(out.as_ndarray_like(), expected(values, [40, 3], [4, 2]))


def test_async_matches_sync(arr_1d: tuple[zarr.Array[Any], np.ndarray]) -> None:
    a, values = arr_1d
    got = sync(a.async_array.get_range_selection([60, 1], [9, 9]))
    assert_array_equal(got, expected(values, [60, 1], [9, 9]))


@pytest.mark.parametrize(
    ("starts", "lengths", "error"),
    [
        ([98], [3], BoundsCheckError),  # past the end
        ([-1], [1], BoundsCheckError),
        ([1], [-1], BoundsCheckError),
        ([1, 2], [1], IndexError),  # unequal lengths
        ([[1]], [[1]], IndexError),  # not one-dimensional
        ([1.5], [1], IndexError),  # not integers
    ],
)
def test_refusals(
    arr_1d: tuple[zarr.Array[Any], np.ndarray],
    starts: list[Any],
    lengths: list[Any],
    error: type[Exception],
) -> None:
    a, _ = arr_1d
    with pytest.raises(error):
        a.get_range_selection(starts, lengths)


def test_out_of_the_wrong_shape(arr_1d: tuple[zarr.Array[Any], np.ndarray]) -> None:
    a, _ = arr_1d
    out = default_buffer_prototype().nd_buffer.empty(shape=(5,), dtype=np.dtype("i4"))
    with pytest.raises(ValueError, match="shape of out"):
        a.get_range_selection([0], [4], out=out)


def test_indexer_merges_touching_ranges() -> None:
    a = zarr.create_array(store={}, shape=(100,), chunks=(10,), dtype="u1")
    indexer = RunIndexer((([10, 15, 20, 50, 60], [5, 5, 5, 3, 0]),), a.shape, a._chunk_grid)
    (starts, lengths), = indexer.runs
    assert starts.tolist() == [10, 50]
    assert lengths.tolist() == [15, 3]
    # 10..25 crosses two chunks, 50..53 one: a projection per chunk crossed, not per row.
    assert len(list(indexer)) == 3


def test_pipeline_hook_serves_the_read(
    arr_1d: tuple[zarr.Array[Any], np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    a, _ = arr_1d
    calls: list[list[tuple[list[int], list[int]]]] = []

    async def read_runs(self, store_path, metadata, runs, out, **kwargs) -> None:  # type: ignore[no-untyped-def]
        calls.append([(s.tolist(), n.tolist()) for s, n in runs])
        out.as_ndarray_like()[...] = -1

    monkeypatch.setattr(type(a.async_array.codec_pipeline), "read_runs", read_runs)
    got = a.get_range_selection([10, 15, 40], [5, 5, 2])
    assert calls == [[([10, 40], [10, 2])]]
    assert_array_equal(got, np.full(12, -1))


def test_pipeline_hook_can_defer_to_the_default(
    arr_1d: tuple[zarr.Array[Any], np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    a, values = arr_1d
    calls = []

    async def read_runs(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[no-untyped-def]
        calls.append(1)
        await CodecPipeline.read_runs(self, *args, **kwargs)

    monkeypatch.setattr(type(a.async_array.codec_pipeline), "read_runs", read_runs)
    assert_array_equal(a.get_range_selection([5, 30], [10, 3]), expected(values, [5, 30], [10, 3]))
    assert calls == [1]



@pytest.mark.parametrize("shards", [None, (20, 10, 10)])
def test_runs_on_several_axes(shards: tuple[int, ...] | None) -> None:
    values = np.arange(60 * 10 * 20, dtype="i4").reshape(60, 10, 20)
    a = zarr.create_array(store={}, shape=values.shape, chunks=(4, 5, 5), shards=shards, dtype="i4")
    a[:] = values
    selection = (([0, 40, 3], [5, 5, 2]), slice(2, 6), ([10, 0], [3, 2]))
    rows, cols, depth = np.r_[0:5, 40:45, 3:5], np.r_[2:6], np.r_[10:13, 0:2]
    assert_array_equal(a.get_run_selection(selection), values[np.ix_(rows, cols, depth)])


def test_run_selection_refusals(arr_1d: tuple[zarr.Array[Any], np.ndarray]) -> None:
    a, _ = arr_1d
    with pytest.raises(IndexError, match="step 1"):
        a.get_run_selection((slice(0, 10, 2),))
    with pytest.raises(IndexError, match="too many axes"):
        a.get_run_selection((slice(None), slice(None)))


@pytest.fixture
def served_2d(monkeypatch: pytest.MonkeyPatch) -> tuple[zarr.Array[Any], np.ndarray, list[int]]:
    """A 2-D array whose pipeline overrides read_runs, counting the calls."""
    values = np.arange(12 * 10, dtype="i4").reshape(12, 10)
    a = zarr.create_array(store={}, shape=values.shape, chunks=(5, 4), dtype="i4")
    a[:] = values
    calls: list[int] = []

    async def read_runs(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[no-untyped-def]
        calls.append(1)
        await CodecPipeline.read_runs(self, *args, **kwargs)

    monkeypatch.setattr(type(a.async_array.codec_pipeline), "read_runs", read_runs)
    return a, values, calls


@pytest.mark.parametrize(
    "selection",
    [
        np.s_[2:5, 1:7],
        np.s_[3, 1:7],
        np.s_[..., 2],
        np.s_[-1, :],
        np.s_[4:4, :],
    ],
)
def test_basic_reads_go_through_read_runs(
    served_2d: tuple[zarr.Array[Any], np.ndarray, list[int]], selection: Any
) -> None:
    a, values, calls = served_2d
    assert_array_equal(a[selection], values[selection])
    assert calls == [1]


@pytest.mark.parametrize(
    "selection",
    [
        ([5, 1, 2, 3], slice(2, 4)),
        ([-1, 0, 0], [3, 1]),
        (np.arange(12) % 3 == 0, [0, 9]),
        (7, [2, 3, 4]),
    ],
)
def test_orthogonal_reads_go_through_read_runs(
    served_2d: tuple[zarr.Array[Any], np.ndarray, list[int]], selection: Any
) -> None:
    a, values, calls = served_2d
    rows, cols = (np.arange(n)[s] if not isinstance(s, int) else s for s, n in zip(selection, values.shape))
    want = values[rows][:, cols] if not isinstance(rows, int) else values[rows][cols]
    assert_array_equal(a.oindex[selection], want)
    assert calls == [1]


@pytest.mark.parametrize("selection", [np.s_[::2, :], np.s_[3, 4]])
def test_other_reads_keep_their_path(
    served_2d: tuple[zarr.Array[Any], np.ndarray, list[int]], selection: Any
) -> None:
    a, values, calls = served_2d
    assert_array_equal(a[selection], values[selection])
    assert calls == []


def test_no_dispatch_without_an_override() -> None:
    a = zarr.create_array(store={}, shape=(8, 8), chunks=(4, 4), dtype="u1")
    assert type(a.async_array.codec_pipeline).read_runs is CodecPipeline.read_runs
    assert_array_equal(a[1:3, 2:5], np.zeros((2, 3), dtype="u1"))
