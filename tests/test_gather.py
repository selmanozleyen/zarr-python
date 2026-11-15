"""Tests for the experimental `zarr.experimental.gather`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings
from numpy.testing import assert_array_equal

import zarr
from zarr.abc.codec import CodecPipeline
from zarr.errors import ChunkNotFoundError, LazyIndexingDisabledError
from zarr.experimental import gather
from zarr.storage import LoggingStore, MemoryStore
from zarr.testing.strategies import rectilinear_arrays, sharded_arrays, simple_arrays

if TYPE_CHECKING:
    from collections.abc import Iterator

    from zarr.types import AnyArray

zarr_indexing = pytest.importorskip("zarr_indexing")
selections = pytest.importorskip("zarr_indexing.testing.strategies")

pytestmark = pytest.mark.filterwarnings(
    "ignore::zarr.core.dtype.common.UnstableSpecificationWarning"
)

any_array = st.one_of(
    simple_arrays(shapes=npst.array_shapes(max_dims=3, min_side=1, max_side=10)),
    rectilinear_arrays(shapes=npst.array_shapes(max_dims=3, min_side=1, max_side=20)),
    sharded_arrays(shapes=npst.array_shapes(min_dims=1, max_dims=3, min_side=1, max_side=8)),
)
DATA = np.arange(17 * 13, dtype="int32").reshape(17, 13)


@pytest.fixture(autouse=True)
def lazy_enabled() -> Iterator[None]:
    with zarr.config.set({"experimental_lazy_indexing": True}):
        yield


def make_array(store: Any = None, **kwargs: Any) -> AnyArray:
    arr = zarr.create_array(
        store if store is not None else {}, shape=DATA.shape, chunks=(4, 4), dtype="int32", **kwargs
    )
    arr[:] = DATA
    return arr


@st.composite
def views(draw: st.DrawFn, zarray: Any) -> Any:
    mode = draw(st.sampled_from(["basic", "orthogonal", "vectorized"]))
    if mode == "basic":
        return zarray.lazy[draw(selections.basic_selections(zarray.shape))]
    if mode == "orthogonal":
        return zarray.lazy.oindex[draw(selections.orthogonal_selections(zarray.shape))]
    return zarray.lazy.vindex[draw(selections.vectorized_selections(zarray.shape))]


@settings(deadline=None)
@given(data=st.data())
def test_matches_result(data: st.DataObject) -> None:
    # One array per example: the strategies reuse a single store, so a second draw overwrites it.
    zarray = data.draw(any_array)
    picked = data.draw(st.lists(views(zarray), min_size=1, max_size=4))
    for got, view in zip(gather(picked), picked, strict=True):
        assert_array_equal(got, view.result())


def test_several_arrays() -> None:
    a = make_array()
    b = make_array(shards=(8, 8))
    got = gather([a.lazy[1:5], b.lazy.oindex[[16, 2], :], a.lazy[::-4, 3]])
    assert_array_equal(got[0], DATA[1:5])
    assert_array_equal(got[1], DATA[[16, 2]])
    assert_array_equal(got[2], DATA[::-4, 3])


@pytest.mark.parametrize("shards", [None, (8, 8)])
def test_reads_each_chunk_once(shards: tuple[int, ...] | None) -> None:
    store = LoggingStore(store=MemoryStore())
    arr = make_array(store, shards=shards)
    view = arr.lazy[2:10, 1:9]
    store.counter.clear()
    view.result()
    once = store.counter["get"]
    store.counter.clear()
    results = gather([view, view, arr.lazy[3:9, 2:8]])
    assert store.counter["get"] <= once
    assert_array_equal(results[0], DATA[2:10, 1:9])
    assert_array_equal(results[2], DATA[3:9, 2:8])


def test_pipeline_gets_each_chunk_once_with_all_its_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    arr = make_array()
    calls: list[list[Any]] = []
    real = CodecPipeline.read_gather

    async def spy(self: CodecPipeline, requests: Any) -> Any:
        requests = list(requests)
        calls.append(requests)
        return await real(self, requests)

    monkeypatch.setattr(CodecPipeline, "read_gather", spy)
    gather([arr.lazy[0:4, 0:8], arr.lazy[2:4, 4:8]])
    (requests,) = calls
    keys = [str(byte_getter) for byte_getter, _, _ in requests]
    assert len(keys) == len(set(keys)) == 2
    assert sorted(len(reads) for _, _, reads in requests) == [1, 2]


def test_missing_chunks() -> None:
    arr = zarr.create_array({}, shape=(8,), chunks=(4,), dtype="int32", fill_value=7)
    arr[:4] = 1
    assert_array_equal(gather([arr.lazy[2:6]])[0], [1, 1, 7, 7])
    strict = arr.with_config({"read_missing_chunks": False})
    with pytest.raises(ChunkNotFoundError, match="c/1"):
        gather([strict.lazy[2:6]])


def test_zarr_v2_fortran_order() -> None:
    arr = zarr.create_array(
        {}, shape=DATA.shape, chunks=(4, 4), dtype="int32", zarr_format=2, order="F"
    )
    arr[:] = DATA
    assert_array_equal(gather([arr.lazy[1:15:3, ::-2]])[0], DATA[1:15:3, ::-2])


def test_empty_and_scalar_views() -> None:
    arr = make_array()
    empty, scalar = gather([arr.lazy[3:3], arr.lazy[4, 5]])
    assert empty.shape == (0, 13)
    assert scalar.shape == ()
    assert scalar == DATA[4, 5]


def test_flag_off_raises() -> None:
    view = make_array().lazy[0:2]
    with zarr.config.set({"experimental_lazy_indexing": False}):
        with pytest.raises(LazyIndexingDisabledError, match="gather"):
            gather([view])


def test_rejects_views_of_other_arrays() -> None:
    with pytest.raises(TypeError, match="zarr arrays"):
        gather([zarr_indexing.LazyArray.from_numpy(DATA)[0:2]])


@pytest.mark.parametrize("shards", [None, (8, 8)])
def test_fused_pipeline(shards: tuple[int, ...] | None) -> None:
    with zarr.config.set({"codec_pipeline.path": "zarr.core.codec_pipeline.FusedCodecPipeline"}):
        arr = make_array(shards=shards)
        got = gather([arr.lazy[2:10, 1:9], arr.lazy.vindex[[0, 16], [12, 3]]])
    assert_array_equal(got[0], DATA[2:10, 1:9])
    assert_array_equal(got[1], DATA[[0, 16], [12, 3]])
