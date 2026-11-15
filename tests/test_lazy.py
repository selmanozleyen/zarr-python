"""Tests for the experimental `Array.lazy` accessor."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING, Any

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings
from numpy.testing import assert_array_equal

import zarr
from zarr.storage import LoggingStore, MemoryStore
from zarr.testing.strategies import (
    orthogonal_indices,
    rectilinear_arrays,
    sharded_arrays,
    simple_arrays,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from zarr.types import AnyArray

pytest.importorskip("zarr_indexing")

pytestmark = pytest.mark.filterwarnings(
    "ignore::zarr.core.dtype.common.UnstableSpecificationWarning"
)

nonempty_shapes = npst.array_shapes(max_dims=4, min_side=1, max_side=10)
any_array = st.one_of(
    simple_arrays(shapes=nonempty_shapes),
    rectilinear_arrays(shapes=npst.array_shapes(max_dims=3, min_side=1, max_side=20)),
    sharded_arrays(shapes=npst.array_shapes(min_dims=1, max_dims=3, min_side=1, max_side=8)),
)


@pytest.fixture(autouse=True)
def lazy_enabled() -> Iterator[None]:
    with zarr.config.set({"experimental_lazy_indexing": True}):
        yield


def make_array(*, shards: tuple[int, ...] | None = None, store: Any = None) -> AnyArray:
    arr = zarr.create_array(
        store if store is not None else {},
        shape=(17, 13),
        chunks=(4, 4),
        shards=shards,
        dtype="int32",
    )
    arr[:] = np.arange(17 * 13, dtype="int32").reshape(17, 13)
    return arr


@settings(deadline=None)
@given(data=st.data())
def test_basic_parity(data: st.DataObject) -> None:
    zarray = data.draw(any_array)
    nparray = zarray[...]
    # Negative steps included: lazy indexing supports them, eager zarr does not.
    sel = data.draw(npst.basic_indices(shape=nparray.shape))
    view = zarray.lazy[sel]
    expected = nparray[sel]
    assert view.shape == np.shape(expected)
    assert_array_equal(view.result(), expected)


@settings(deadline=None)
@given(data=st.data())
def test_oindex_parity(data: st.DataObject) -> None:
    zarray = data.draw(any_array)
    nparray = zarray[...]
    zsel, npsel = data.draw(orthogonal_indices(shape=nparray.shape))
    assert_array_equal(zarray.lazy.oindex[zsel].result(), nparray[npsel])


@settings(deadline=None)
@given(data=st.data())
def test_vindex_parity(data: st.DataObject) -> None:
    zarray = data.draw(any_array)
    nparray = zarray[...]
    sel = data.draw(npst.integer_array_indices(shape=nparray.shape))
    assert_array_equal(zarray.lazy.vindex[sel].result(), nparray[sel])


@pytest.mark.parametrize("shards", [None, (8, 8)])
def test_mask_and_composition(shards: tuple[int, ...] | None) -> None:
    arr = make_array(shards=shards)
    expected = np.asarray(arr[...])
    mask = np.zeros(arr.shape, dtype=bool)
    mask[::3, 1::4] = True
    assert_array_equal(arr.lazy.vindex[mask].result(), expected[mask])
    assert_array_equal(arr.lazy[2:15][1::2, 3:].result(), expected[2:15][1::2, 3:])
    assert_array_equal(arr.lazy[2:15][::-2, 5].result(), expected[2:15][::-2, 5])
    assert_array_equal(
        arr.lazy[1:][::2].oindex[[0, 3, 3], [1, 0]].result(),
        expected[1:][::2][np.ix_([0, 3, 3], [1, 0])],
    )


def test_sharded_reads_by_inner_chunk() -> None:
    # The read grid is the inner-chunk grid: one reader call per touched inner chunk,
    # not per shard. A known cost here, not a correctness issue.
    arr = make_array(shards=(8, 8))
    parts = list(arr.lazy[0:8, 0:8].parts())
    assert len(parts) == 4


def test_no_io_before_result() -> None:
    store = LoggingStore(store=MemoryStore())
    arr = make_array(store=store)
    store.counter.clear()
    view = arr.lazy[1:15, ::2].oindex[[0, 2, 2], :][1:]
    view = view.vindex[[0, 1], [1, 2]]
    assert view.shape == (2,)
    assert dict(store.counter) == {}
    expected = np.asarray(arr[...])[1:15, ::2][[2, 2]][[0, 1], [1, 2]]
    assert_array_equal(np.asarray(view), expected)
    assert store.counter["get"] > 0


def test_async_array_lazy() -> None:
    arr = make_array()
    view = arr.async_array.lazy[3:7, 1]
    assert_array_equal(view.result(), arr[3:7, 1])

    async def read() -> Any:
        return await asyncio.to_thread(arr.async_array.lazy[3:7, 1].result)

    assert_array_equal(asyncio.run(read()), arr[3:7, 1])


def test_flag_off_raises() -> None:
    arr = make_array()
    with zarr.config.set({"experimental_lazy_indexing": False}):
        assert zarr.config.get("experimental_lazy_indexing") is False
        with pytest.raises(RuntimeError, match="ZARR_EXPERIMENTAL_LAZY_INDEXING"):
            _ = arr.lazy
        with pytest.raises(RuntimeError, match="experimental_lazy_indexing"):
            _ = arr.async_array.lazy
        assert_array_equal(arr[2:4, 1], [27, 40])


def test_flag_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZARR_EXPERIMENTAL_LAZY_INDEXING", "True")
    zarr.config.reset()
    assert zarr.config.get("experimental_lazy_indexing") is True
    arr = make_array()
    assert_array_equal(arr.lazy[2:4, 1].result(), [27, 40])


def test_flag_default_off() -> None:
    zarr.config.reset()
    assert zarr.config.get("experimental_lazy_indexing") is False


def test_zarr_indexing_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    arr = make_array()
    monkeypatch.setitem(sys.modules, "zarr_indexing", None)
    with pytest.raises(ImportError, match=r"zarr\[lazy\]"):
        _ = arr.lazy
