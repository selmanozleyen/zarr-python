"""Tests for the experimental `Array.lazy` accessor."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings
from numpy.testing import assert_array_equal

import zarr
from zarr.errors import LazyIndexingDisabledError
from zarr.storage import LoggingStore, MemoryStore
from zarr.testing.strategies import (
    orthogonal_indices,
    rectilinear_arrays,
    sharded_arrays,
    simple_arrays,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

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
DATA = np.arange(17 * 13, dtype="int32").reshape(17, 13)


@pytest.fixture(autouse=True)
def lazy_enabled() -> Iterator[None]:
    with zarr.config.set({"experimental_lazy_indexing": True}):
        yield


def make_array(*, shards: tuple[int, ...] | None = None, store: Any = None) -> AnyArray:
    arr = zarr.create_array(
        store if store is not None else {},
        shape=DATA.shape,
        chunks=(4, 4),
        shards=shards,
        dtype="int32",
    )
    arr[:] = DATA
    return arr


# The property tests use `zarray[...]` as the oracle, as tests/test_properties.py does:
# the strategies draw the NumPy data internally and do not return it.


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


MASK = np.zeros(DATA.shape, dtype=bool)
MASK[::3, 1::4] = True


@pytest.mark.parametrize("shards", [None, (8, 8)])
@pytest.mark.parametrize(
    ("lazy", "expected"),
    [
        (lambda v: v.vindex[MASK], DATA[MASK]),
        (lambda v: v[2:15][1::2, 3:], DATA[2:15][1::2, 3:]),
        (lambda v: v[2:15][::-2, 5], DATA[2:15][::-2, 5]),
        (lambda v: v[1:][::2].oindex[[0, 3, 3], [1, 0]], DATA[1:][::2][np.ix_([0, 3, 3], [1, 0])]),
    ],
    ids=["mask", "compose", "reverse", "compose-oindex"],
)
def test_selection(
    shards: tuple[int, ...] | None, lazy: Callable[[Any], Any], expected: np.ndarray[Any, Any]
) -> None:
    assert_array_equal(lazy(make_array(shards=shards).lazy).result(), expected)


def test_no_io_before_result() -> None:
    store = LoggingStore(store=MemoryStore())
    arr = make_array(store=store)
    store.counter.clear()
    view = arr.lazy[1:15, ::2].oindex[[0, 2, 2], :][1:]
    view = view.vindex[[0, 1], [1, 2]]
    assert view.shape == (2,)
    assert dict(store.counter) == {}
    assert_array_equal(np.asarray(view), DATA[1:15, ::2][[2, 2]][[0, 1], [1, 2]])
    assert store.counter["get"] > 0


def test_write() -> None:
    arr = make_array()
    arr.lazy[2:4, ::5] = -1
    expected = DATA.copy()
    expected[2:4, ::5] = -1
    assert_array_equal(arr[...], expected)


def test_async_array_lazy() -> None:
    assert_array_equal(make_array().async_array.lazy[3:7, 1].result(), DATA[3:7, 1])


def test_flag_off_raises() -> None:
    arr = make_array()
    with zarr.config.set({"experimental_lazy_indexing": False}):
        with pytest.raises(LazyIndexingDisabledError, match="ZARR_EXPERIMENTAL_LAZY_INDEXING"):
            _ = arr.lazy
        with pytest.raises(LazyIndexingDisabledError):
            _ = arr.async_array.lazy


def test_flag_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZARR_EXPERIMENTAL_LAZY_INDEXING", "True")
    zarr.config.reset()
    assert zarr.config.get("experimental_lazy_indexing") is True
    assert_array_equal(make_array().lazy[2:4, 1].result(), DATA[2:4, 1])


def test_zarr_indexing_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    arr = make_array()
    monkeypatch.setitem(sys.modules, "zarr_indexing", None)
    with pytest.raises(ImportError, match=r"zarr\[lazy\]"):
        _ = arr.lazy
