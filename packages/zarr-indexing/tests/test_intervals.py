from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from zarr_indexing import (
    ArrayMap,
    BoundsCheckError,
    ConstantMap,
    DimensionMap,
    IndexedSet,
    IndexTransform,
    Intervals,
    IntervalSet,
    LazyArray,
    plan_chunks,
)
from zarr_indexing.grid import dimension_grids_from_chunks
from zarr_indexing.intervals import IntervalArrayMap


def r_(lo: list[int], hi: list[int]) -> np.ndarray[Any, Any]:
    """The oracle: what `np.r_[lo[0]:hi[0], ...]` selects, built without `Intervals`."""
    pieces = [np.arange(a, b) for a, b in zip(lo, hi, strict=True)]
    return np.concatenate(pieces).astype(np.intp) if pieces else np.empty(0, dtype=np.intp)


def test_indices_are_np_r() -> None:
    assert Intervals([5, 0, 3, 3], [8, 2, 3, 4]).indices().tolist() == [5, 6, 7, 0, 1, 3]
    assert Intervals([], []).indices().tolist() == []
    assert repr(Intervals([1], [3])) == "Intervals(lo=[1], hi=[3])"


@pytest.mark.parametrize(
    ("lo", "hi", "error"),
    [
        ([0, 1], [1], ValueError),
        ([[0]], [[1]], ValueError),
        ([3], [2], ValueError),
        ([0.5], [1], TypeError),
    ],
)
def test_validation(lo: Any, hi: Any, error: type[Exception]) -> None:
    with pytest.raises(error):
        Intervals(lo, hi)


@st.composite
def cases(
    draw: st.DrawFn,
) -> tuple[LazyArray, np.ndarray[Any, Any], tuple[Any, ...], np.ndarray[Any, Any]]:
    """A view with a nonzero origin, an oindex key holding `Intervals`, and the expected values."""
    rank = draw(st.integers(1, 3))
    axis = draw(st.integers(0, rank - 1))
    extents = [draw(st.integers(1, 3)) for _ in range(rank)]
    extents[axis] = draw(st.integers(1, 20))
    offset = draw(st.integers(0, 3))
    source = np.arange(int(np.prod(extents)) * (offset + 1)).reshape(
        [e * (offset + 1) if j == 0 else e for j, e in enumerate(extents)]
    )
    view_values = source[offset : offset + extents[0]]
    lo = draw(st.lists(st.integers(0, extents[axis]), max_size=6))
    hi = [draw(st.integers(a, extents[axis])) for a in lo]
    parts = tuple(draw(st.integers(1, n)) for n in source.shape)
    view = LazyArray.from_numpy(source).with_parts(parts)[offset : offset + extents[0]]
    key = tuple(Intervals(lo, hi) if j == axis else slice(None) for j in range(rank))
    return view, view_values, key, np.take(view_values, r_(lo, hi), axis=axis)


@settings(deadline=None)
@given(case=cases())
def test_matches_numpy(case: Any) -> None:
    view, _, key, expected = case
    selected = view.oindex[key]
    np.testing.assert_array_equal(selected.result(), expected)
    np.testing.assert_array_equal(selected.unpartitioned().result(), expected)


def test_planning_and_reading_never_build_the_index_array() -> None:
    source = np.arange(40 * 3).reshape(40, 3)
    view = (
        LazyArray.from_numpy(source)
        .with_parts((8, 3))
        .oindex[Intervals([30, 2, 10], [34, 8, 15]), :]
    )
    m = view.transform.output[0]
    assert isinstance(m, IntervalArrayMap)
    np.testing.assert_array_equal(view.result(), source[r_([30, 2, 10], [34, 8, 15])])
    assert m._index_array is None


def test_plan_of_huge_intervals_is_per_interval() -> None:
    n = 10**12
    transform = IndexTransform.from_shape((n,)).oindex[Intervals([0, n - 10**9], [10**9, n])]
    (axis,) = (
        plan_chunks(transform, dimension_grids_from_chunks((10**6,), shape=(n,))).partition().sets
    )
    assert isinstance(axis, IntervalSet)
    assert axis.extent.size == 2000
    assert int(axis.extent.sum()) == 2 * 10**9


def test_same_map_as_the_index_list() -> None:
    t = IndexTransform.from_shape((40, 3))
    by_intervals = t.oindex[Intervals([1, 9], [4, 11]), :]
    by_list = t.oindex[np.r_[1:4, 9:11], :]
    assert by_intervals == by_list
    assert hash(by_intervals) == hash(by_list)
    assert by_intervals.to_json() == by_list.to_json()
    restored = pickle.loads(pickle.dumps(by_intervals.output[0]))
    assert type(restored) is ArrayMap
    assert restored == by_list.output[0]


def test_pieces_merge_and_single_pieces_are_slices() -> None:
    # [0, 2) and [2, 4) continue each other: one piece filling the chunk.
    transform = IndexTransform.from_shape((8,)).oindex[Intervals([0, 2, 5], [2, 4, 7])]
    plan = plan_chunks(transform, dimension_grids_from_chunks((4,), shape=(8,)))
    axis = plan.partition().sets[0]
    assert isinstance(axis, IntervalSet)
    assert axis.extent.tolist() == [4, 2]
    first, second = list(plan)
    assert isinstance(first.chunk_transform.output[0], DimensionMap)
    assert first.coverage == "full"
    assert second.coverage == "partial"


def test_no_merge_when_not_neighbours_in_the_request() -> None:
    # [0, 2) and [2, 4) touch in storage but [5, 6) sits between them in the request.
    source = np.arange(8)
    view = LazyArray.from_numpy(source).with_parts((4,)).oindex[Intervals([0, 5, 2], [2, 6, 4])]
    np.testing.assert_array_equal(view.result(), [0, 1, 5, 2, 3])


def test_one_index_is_a_constant_and_none_is_empty() -> None:
    t = IndexTransform.from_shape((10,))
    assert t.oindex[Intervals([4, 7], [5, 7])].output[0] == ConstantMap(4)
    assert LazyArray.from_numpy(np.arange(10)).oindex[Intervals([3], [3])].shape == (0,)


def test_strided_view_falls_back_to_the_index_list() -> None:
    source = np.arange(40)
    view = LazyArray.from_numpy(source).with_parts((8,))[::2].oindex[Intervals([3, 0], [6, 2])]
    (axis,) = (
        plan_chunks(view.transform, dimension_grids_from_chunks((8,), shape=(40,))).partition().sets
    )
    assert isinstance(axis, IndexedSet)
    np.testing.assert_array_equal(view.result(), source[::2][r_([3, 0], [6, 2])])


def test_further_selections_stay_correct() -> None:
    source = np.arange(40 * 3).reshape(40, 3)
    rows = source[r_([30, 2, 10], [34, 8, 15])]
    view = LazyArray.from_numpy(source).oindex[Intervals([30, 2, 10], [34, 8, 15]), :]
    np.testing.assert_array_equal(view[2:9, ::-1].result(), rows[2:9, ::-1])
    np.testing.assert_array_equal(view.vindex[[0, 14], [2, 1]].result(), rows[[0, 14], [2, 1]])


def test_errors() -> None:
    view = LazyArray.from_numpy(np.arange(10))
    with pytest.raises(IndexError, match="out of bounds"):
        view.oindex[Intervals([8], [11])]
    with pytest.raises(IndexError, match="oindex"):
        view[Intervals([1], [3])]
    with pytest.raises(IndexError, match="oindex"):
        view.vindex[Intervals([1], [3])]
    with pytest.raises(BoundsCheckError, match="outside"):
        IndexTransform.from_shape((10,)).oindex[Intervals([8], [11])]


@pytest.mark.parametrize(
    "shift",
    [lambda t: t, lambda t: t.translate((5,)), lambda t: t.translate_domain_by((3,))],
    ids=["plain", "storage-offset", "domain-origin"],
)
def test_plan_maps_every_cell_like_the_index_list(shift: Any) -> None:
    lo, hi = [9, 1, 4, 13], [12, 3, 8, 13]
    transform = shift(IndexTransform.from_shape((20,)).oindex[Intervals(lo, hi)])
    expected = shift(IndexTransform.from_shape((20,)).oindex[r_(lo, hi)])
    assert isinstance(transform.output[0], IntervalArrayMap)
    grids = dimension_grids_from_chunks((4,), shape=(25,))
    mapped = {}
    for p in plan_chunks(transform, grids):
        points = np.arange(p.chunk_transform.domain.shape[0]).reshape(-1, 1)
        cells = p.cell_transform.apply_many(points)[:, 0]
        storage = p.chunk_transform.apply_many(points)[:, 0] + p.chunk_domain.inclusive_min[0]
        mapped.update(zip(cells.tolist(), storage.tolist(), strict=True))
    lo_cell = expected.domain.inclusive_min[0]
    cells = np.arange(lo_cell, expected.domain.exclusive_max[0]).reshape(-1, 1)
    assert mapped == dict(
        zip(cells[:, 0].tolist(), expected.apply_many(cells)[:, 0].tolist(), strict=True)
    )
