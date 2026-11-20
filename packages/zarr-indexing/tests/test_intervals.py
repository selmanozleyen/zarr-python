from __future__ import annotations

import copy
import pickle
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from zarr_indexing import (
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
    assert repr(Intervals([3, 1], [3, 3])) == "Intervals(lo=[1], hi=[3])"  # empty ones dropped


@pytest.mark.parametrize(
    ("lo", "hi", "error"),
    [
        ([0, 1], [1], ValueError),
        ([[0]], [[1]], ValueError),
        ([3], [2], ValueError),
        ([0.5], [1], TypeError),
        (np.array([0], np.uint64), np.array([2**63 + 5], np.uint64), OverflowError),
        ([0, 0], [2**62, 2**62], OverflowError),
        ([0, 0, 0], [2**63 - 1] * 3, OverflowError),
    ],
    ids=["lengths", "2-d", "hi-below-lo", "float", "uint64-value", "size", "size-past-uint64"],
)
def test_validation(lo: Any, hi: Any, error: type[Exception]) -> None:
    with pytest.raises(error):
        Intervals(lo, hi)


@st.composite
def cases(
    draw: st.DrawFn,
) -> tuple[LazyArray, np.ndarray[Any, Any], tuple[Any, ...], np.ndarray[Any, Any]]:
    """A strided or reversed view with a nonzero origin, an oindex key holding `Intervals`,
    and the expected values."""
    rank = draw(st.integers(1, 3))
    axis = draw(st.integers(0, rank - 1))
    extents = [draw(st.integers(1, 3)) for _ in range(rank)]
    extents[axis] = draw(st.integers(1, 20))
    offset = draw(st.integers(1, 3))
    step = draw(st.sampled_from([1, 2, -1]))
    rows = [offset + 2 * extents[0], *extents[1:]]
    source = np.arange(int(np.prod(rows))).reshape(rows)
    first = offset + extents[0] - 1 if step < 0 else offset
    window = slice(first, first + step * extents[0], step)
    view_values = source[window]

    def intervals(extent: int) -> tuple[Intervals, np.ndarray[Any, Any]]:
        lo = draw(st.lists(st.integers(0, extent), max_size=6))
        hi = [draw(st.integers(a, extent)) for a in lo]
        return Intervals(lo, hi), r_(lo, hi)

    key: list[Any] = []
    takes: list[Any] = []
    for j in range(rank):
        kind = (
            "intervals"
            if j == axis
            else draw(st.sampled_from(["slice", "int", "list", "intervals"]))
        )
        if kind == "intervals":
            entry, take = intervals(extents[j])
        elif kind == "int":
            entry = take = draw(st.integers(0, extents[j] - 1))
        elif kind == "list":
            entry = take = draw(st.lists(st.integers(0, extents[j] - 1), min_size=1, max_size=4))
        else:
            entry, take = slice(None), np.arange(extents[j])
        key.append(entry)
        takes.append(take)
    expected = view_values
    for j in reversed(range(rank)):  # last axis first, so a dropped axis shifts none still to come
        expected = np.take(expected, takes[j], axis=j)
    parts = tuple(draw(st.integers(1, n)) for n in source.shape)
    view = LazyArray.from_numpy(source).with_parts(parts)[window]
    return view, view_values, tuple(key), expected


@settings(deadline=None)
@given(case=cases())
def test_matches_numpy(case: Any) -> None:
    view, _, key, expected = case
    selected = view.oindex[key]
    np.testing.assert_array_equal(selected.result(), expected)
    np.testing.assert_array_equal(selected.unpartitioned().result(), expected)


def test_planning_and_reading_never_build_the_index_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = np.arange(40 * 3).reshape(40, 3)
    view = (
        LazyArray.from_numpy(source)
        .with_parts((8, 3))
        .oindex[Intervals([30, 2, 10], [34, 8, 15]), :]
    )
    assert isinstance(view.transform.output[0], IntervalArrayMap)

    def indices(self: Intervals) -> np.ndarray[Any, Any]:
        raise AssertionError("built the full index array")

    monkeypatch.setattr(Intervals, "indices", indices)
    np.testing.assert_array_equal(view.result(), source[r_([30, 2, 10], [34, 8, 15])])


@pytest.mark.parametrize(("offset", "coordinate"), [(15, 26), (-5, -4)], ids=["above", "below"])
def test_out_of_grid_error_names_the_coordinate(offset: int, coordinate: int) -> None:
    selected = IndexTransform.from_shape((20,)).oindex[Intervals([1, 9], [4, 12])]
    transform = selected.translate((offset,))
    with pytest.raises(IndexError, match=f"index {coordinate} is out of bounds"):
        list(plan_chunks(transform, dimension_grids_from_chunks((4,), shape=(25,))))


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
    assert pickle.loads(pickle.dumps(by_intervals)) == by_list


def test_planning_near_the_intp_limit_overflows_cleanly() -> None:
    transform = (
        IndexTransform.from_shape((20,)).oindex[Intervals([0], [20])].translate((2**63 - 10,))
    )
    with pytest.raises(OverflowError):
        list(plan_chunks(transform, dimension_grids_from_chunks((4,), shape=(2**63 - 1,))))


def test_pickle_and_copy_keep_intervals(monkeypatch: pytest.MonkeyPatch) -> None:
    def indices(self: Intervals) -> np.ndarray[Any, Any]:
        raise AssertionError("built the full index array")

    monkeypatch.setattr(Intervals, "indices", indices)
    n = 10**12
    transform = IndexTransform.from_shape((n,)).oindex[Intervals([0, n - 10**9], [10**9, n])]
    data = pickle.dumps(transform)
    assert len(data) < 2000
    restored = pickle.loads(data).output[0]
    assert isinstance(restored, IntervalArrayMap)
    for selector in (
        pickle.loads(pickle.dumps(restored.intervals)),
        copy.deepcopy(restored.intervals),
    ):
        with pytest.raises(ValueError, match="read-only"):
            selector.hi[0] = 1


def test_intervals_cannot_change_after_selecting() -> None:
    lo, hi = np.array([1, 9]), np.array([4, 12])
    selector = Intervals(lo, hi)
    transform = IndexTransform.from_shape((20,)).oindex[selector]
    lo[0], hi[1] = 0, 10
    with pytest.raises(ValueError, match="read-only"):
        selector.hi[1] = 10
    with pytest.raises(ValueError, match="WRITEABLE"):
        selector.hi.flags.writeable = True
    with pytest.raises(AttributeError, match="immutable"):
        selector.hi = np.array([4, 10])
    assert transform == IndexTransform.from_shape((20,)).oindex[r_([1, 9], [4, 12])]


def test_empty_intervals_may_lie_outside_like_np_r() -> None:
    source = np.arange(12)
    view = LazyArray.from_numpy(source).oindex[Intervals([13, 0, 5], [13, 2, 7])]
    np.testing.assert_array_equal(view.result(), source[np.r_[13:13, 0:2, 5:7]])
    t = IndexTransform.from_shape((12,)).oindex[Intervals([13, 0, 5], [13, 2, 7])]
    assert t == IndexTransform.from_shape((12,)).oindex[[0, 1, 5, 6]]


def test_write_through_intervals() -> None:
    target = np.zeros((12, 2), dtype=int)
    values = np.arange(10).reshape(5, 2)
    LazyArray.from_numpy(target).with_parts((4, 2)).oindex[Intervals([9, 1], [11, 4]), :] = values
    expected = np.zeros((12, 2), dtype=int)
    expected[r_([9, 1], [11, 4])] = values
    np.testing.assert_array_equal(target, expected)


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


def test_two_dimensional_oindex_array_still_raises() -> None:
    # Unchanged from before `Intervals`: the array's length, not its size, sizes the axis.
    with pytest.raises(ValueError, match="neither 1 nor"):
        IndexTransform.from_shape((10, 3)).oindex[np.array([[1, 2]]), :]


@pytest.mark.parametrize(
    ("select", "error", "match"),
    [
        (lambda v, t: v.oindex[Intervals([8], [11])], IndexError, "intervals out of bounds"),
        (lambda v, t: v.oindex[Intervals([-3], [-1])], IndexError, "intervals out of bounds"),
        (lambda v, t: v[Intervals([1], [3])], IndexError, "oindex"),
        (lambda v, t: v.vindex[Intervals([1], [3])], IndexError, "oindex"),
        (lambda v, t: t.oindex[Intervals([8], [11])], BoundsCheckError, "out of bounds"),
        (lambda v, t: t.vindex[Intervals([1], [3])], IndexError, "oindex"),
        (lambda v, t: t.select(Intervals([1], [3]), mode="vectorized"), IndexError, "oindex"),
    ],
    ids=[
        "view-bounds",
        "view-negative",
        "view-basic",
        "view-vindex",
        "transform-bounds",
        "transform-vindex",
        "select-vectorized",
    ],
)
def test_errors(select: Any, error: type[Exception], match: str) -> None:
    view = LazyArray.from_numpy(np.arange(10))
    with pytest.raises(error, match=match):
        select(view, IndexTransform.from_shape((10,)))


@pytest.mark.parametrize(
    "shift",
    [
        lambda t: t,
        lambda t: t.translate((5,)),
        lambda t: t.translate_domain_by((3,)),
        lambda t: t.translate_domain_by((-7,)),
    ],
    ids=["plain", "storage-offset", "domain-origin", "negative-domain-origin"],
)
def test_plan_maps_every_cell_like_the_index_list(shift: Any) -> None:
    lo, hi = [9, 1, 4, 13], [12, 3, 8, 13]
    transform = shift(IndexTransform.from_shape((20,)).oindex[Intervals(lo, hi)])
    expected = shift(IndexTransform.from_shape((20,)).oindex[r_(lo, hi)])
    assert isinstance(transform.output[0], IntervalArrayMap)
    grids = dimension_grids_from_chunks((4,), shape=(25,))
    mapped: dict[int, int] = {}
    planned = 0
    for p in plan_chunks(transform, grids):
        points = np.arange(p.chunk_transform.domain.shape[0]).reshape(-1, 1)
        cells = p.cell_transform.apply_many(points)[:, 0]
        storage = p.chunk_transform.apply_many(points)[:, 0] + p.chunk_domain.inclusive_min[0]
        mapped.update(zip(cells.tolist(), storage.tolist(), strict=True))
        planned += len(cells)
    lo_cell = expected.domain.inclusive_min[0]
    cells = np.arange(lo_cell, expected.domain.exclusive_max[0]).reshape(-1, 1)
    assert planned == len(mapped)  # no cell is planned twice
    assert mapped == dict(
        zip(cells[:, 0].tolist(), expected.apply_many(cells)[:, 0].tolist(), strict=True)
    )
