from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import zarr
from zarr.abc.codec import CodecPipeline
from zarr.core.buffer import default_buffer_prototype
from zarr.core.indexing import RangeIndexer
from zarr.core.sync import sync
from zarr.errors import BoundsCheckError


def expected(values: np.ndarray, starts: list[int], lengths: list[int]) -> np.ndarray:
    return np.concatenate([values[s : s + n] for s, n in zip(starts, lengths, strict=True)] + [values[:0]])


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
    ([5], [0]),
    ([99], [1]),
]


@pytest.mark.parametrize(("starts", "lengths"), CASES)
def test_matches_numpy(
    arr_1d: tuple[zarr.Array[Any], np.ndarray], starts: list[int], lengths: list[int]
) -> None:
    a, values = arr_1d
    assert_array_equal(a.get_range_selection(starts, lengths), expected(values, starts, lengths))


def test_trailing_axes_whole() -> None:
    """A trailing axis split over several chunks is read whole, for every range."""
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
    indexer = RangeIndexer([10, 15, 20, 50, 60], [5, 5, 5, 3, 0], a.shape, a._chunk_grid)
    assert indexer.starts.tolist() == [10, 50]
    assert indexer.lengths.tolist() == [15, 3]
    assert indexer.out_starts.tolist() == [0, 15]
    # 10..25 crosses two chunks, 50..53 one: a projection per chunk crossed, not per row.
    assert len(list(indexer)) == 3


def test_pipeline_hook_serves_the_read(
    arr_1d: tuple[zarr.Array[Any], np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pipeline that overrides the hook is handed the merged ranges and reads them itself."""
    a, _ = arr_1d
    calls: list[tuple[list[int], list[int]]] = []

    async def read_ranges(self, store_path, metadata, starts, lengths, out, **kwargs) -> None:  # type: ignore[no-untyped-def]
        calls.append((starts.tolist(), lengths.tolist()))
        out.as_ndarray_like()[...] = -1

    monkeypatch.setattr(type(a.async_array.codec_pipeline), "read_ranges", read_ranges)
    got = a.get_range_selection([10, 15, 40], [5, 5, 2])
    assert calls == [([10, 40], [10, 2])]
    assert_array_equal(got, np.full(12, -1))


def test_pipeline_hook_can_defer_to_the_default(
    arr_1d: tuple[zarr.Array[Any], np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An override that does not serve an array calls the base method, which reads it."""
    a, values = arr_1d
    calls = []

    async def read_ranges(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[no-untyped-def]
        calls.append(1)
        await CodecPipeline.read_ranges(self, *args, **kwargs)

    monkeypatch.setattr(type(a.async_array.codec_pipeline), "read_ranges", read_ranges)
    assert_array_equal(a.get_range_selection([5, 30], [10, 3]), expected(values, [5, 30], [10, 3]))
    assert calls == [1]


def test_default_hook_reads_the_ranges(arr_1d: tuple[zarr.Array[Any], np.ndarray]) -> None:
    a, values = arr_1d
    pipeline = a.async_array.codec_pipeline
    out = default_buffer_prototype().nd_buffer.empty(shape=(7,), dtype=np.dtype("i4"))
    sync(
        CodecPipeline.read_ranges(
            pipeline,
            a.store_path,
            a.metadata,
            np.array([90, 3]),
            np.array([4, 3]),
            out,
            config=a.async_array.config,
            chunk_grid=a._chunk_grid,
            prototype=default_buffer_prototype(),
        )
    )
    assert_array_equal(out.as_ndarray_like(), expected(values, [90, 3], [4, 3]))
