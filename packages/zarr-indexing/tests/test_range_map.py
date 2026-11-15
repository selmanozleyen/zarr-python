from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from zarr_indexing import (
    ArrayMap,
    DimensionMap,
    IndexDomain,
    IndexTransform,
    LazyArray,
    RangeMap,
    RunSet,
    plan_chunks,
)
from zarr_indexing.grid import dimension_grids_from_chunks


def runs_transform(starts: list[int], lengths: list[int], ncols: int) -> IndexTransform:
    m = RangeMap(0, np.array(starts, dtype=np.intp), np.array(lengths, dtype=np.intp))
    return IndexTransform(IndexDomain.from_shape((m.size, ncols)), (m, DimensionMap(1)))


def oracle(starts: list[int], lengths: list[int]) -> np.ndarray[Any, Any]:
    pieces = [np.arange(s, s + n) for s, n in zip(starts, lengths, strict=True)]
    return np.concatenate(pieces).astype(np.intp) if pieces else np.empty(0, dtype=np.intp)


@st.composite
def runs_case(draw: st.DrawFn) -> tuple[int, int, list[int], list[int], tuple[int, int]]:
    nrows = draw(st.integers(1, 30))
    ncols = draw(st.integers(1, 4))
    starts = draw(st.lists(st.integers(0, nrows - 1), max_size=8))
    lengths = [draw(st.integers(0, nrows - s)) for s in starts]
    parts = (draw(st.integers(1, nrows)), draw(st.integers(1, ncols)))
    return nrows, ncols, starts, lengths, parts


@settings(deadline=None)
@given(case=runs_case())
def test_result_matches_numpy(case: tuple[int, int, list[int], list[int], tuple[int, int]]) -> None:
    # Overlapping, repeated, out-of-order and empty runs all included.
    nrows, ncols, starts, lengths, parts = case
    source = np.arange(nrows * ncols).reshape(nrows, ncols)
    view = LazyArray.from_numpy(source).with_parts(parts)[runs_transform(starts, lengths, ncols)]
    expected = source[oracle(starts, lengths)]
    assert isinstance(view.transform.output[0], RangeMap)
    np.testing.assert_array_equal(view.result(), expected)
    np.testing.assert_array_equal(view.unpartitioned().result(), expected)


def test_plan_is_linear_in_runs_not_cells() -> None:
    # 2e9 cells over a 1e12-long axis: an expanded plan would not fit in memory.
    n = 10**12
    m = RangeMap(0, np.array([0, n - 10**9]), np.array([10**9, 10**9]))
    transform = IndexTransform(IndexDomain.from_shape((m.size,)), (m,))
    (axis,) = (
        plan_chunks(transform, dimension_grids_from_chunks((10**6,), shape=(n,))).partition().sets
    )
    assert isinstance(axis, RunSet)
    assert axis.length.size == 2000
    assert int(axis.length.sum()) == m.size


def test_run_set_pieces() -> None:
    # Runs [6, 9) and [1, 3) over chunks of 4: three pieces, one per chunk.
    transform = runs_transform([6, 1], [3, 2], 1)
    plan = plan_chunks(transform, dimension_grids_from_chunks((4, 1), shape=(10, 1)))
    axis = plan.partition().sets[0]
    assert isinstance(axis, RunSet)
    assert axis.chunk.tolist() == [0, 1, 2]
    assert axis.local_start.tolist() == [1, 2, 0]
    assert axis.length.tolist() == [2, 2, 1]
    assert axis.position.tolist() == [3, 0, 2]
    # A chunk with one piece is a slice, so the part is a box.
    assert all(isinstance(p.chunk_transform.output[0], DimensionMap) for p in plan)


def test_run_set_two_pieces_in_one_chunk() -> None:
    transform = runs_transform([0, 2], [1, 1], 1)
    plan = plan_chunks(transform, dimension_grids_from_chunks((4, 1), shape=(4, 1)))
    (projection,) = list(plan)
    assert isinstance(projection.chunk_transform.output[0], ArrayMap)
    assert projection.coverage == "unknown"


def test_compose_onto_offset_view_stays_compact() -> None:
    source = np.arange(40 * 3).reshape(40, 3)
    # `[5:]` keeps literal coordinates [5, 40), so the runs are literal too.
    view = LazyArray.from_numpy(source)[5:][runs_transform([30, 5], [4, 6], 3)]
    assert isinstance(view.transform.output[0], RangeMap)
    np.testing.assert_array_equal(view.result(), source[oracle([30, 5], [4, 6])])


def test_compose_onto_strided_view_expands() -> None:
    source = np.arange(40 * 3).reshape(40, 3)
    view = LazyArray.from_numpy(source)[::2][runs_transform([3, 0], [4, 2], 3)]
    assert isinstance(view.transform.output[0], ArrayMap)
    np.testing.assert_array_equal(view.result(), source[::2][oracle([3, 0], [4, 2])])


def test_further_selection_expands() -> None:
    source = np.arange(40 * 3).reshape(40, 3)
    view = LazyArray.from_numpy(source)[runs_transform([30, 2], [4, 6], 3)]
    expected = source[oracle([30, 2], [4, 6])]
    np.testing.assert_array_equal(view[2:7, ::-1].result(), expected[2:7, ::-1])
    np.testing.assert_array_equal(view.oindex[[0, 9], [2]].result(), expected[np.ix_([0, 9], [2])])


def test_beside_correlated_index_arrays() -> None:
    # A run map and an index array varying over the same axis: the joint walk.
    source = np.arange(10 * 6).reshape(10, 6)
    columns = np.array([5, 0, 3, 3, 1])
    m = RangeMap(0, np.array([7, 1]), np.array([2, 3]))
    transform = IndexTransform(IndexDomain.from_shape((5,)), (m, ArrayMap(columns)))
    view = LazyArray.from_numpy(source).with_parts((3, 4))[transform]
    np.testing.assert_array_equal(view.result(), source[oracle([7, 1], [2, 3]), columns])


def test_transform_operations_match_expanded() -> None:
    transform = runs_transform([6, 1], [3, 2], 2)
    expanded = IndexTransform(
        transform.domain, (ArrayMap(oracle([6, 1], [3, 2]).reshape(5, 1)), DimensionMap(1))
    )
    assert transform.to_json() == expanded.to_json()
    assert transform.apply((3, 1)) == expanded.apply((3, 1))
    points = np.stack(np.meshgrid(np.arange(5), np.arange(2), indexing="ij"), axis=-1)
    np.testing.assert_array_equal(transform.apply_many(points), expanded.apply_many(points))
    assert transform[1:4] == expanded[1:4]
    assert IndexTransform.from_json(transform.to_json()) == expanded
    assert transform != expanded
    assert "runs(2)" in transform.selection_repr
    assert "runs(2)" in repr(transform)


def test_lazy_array_shape_queries() -> None:
    view = LazyArray.from_numpy(np.zeros((40, 3)))[runs_transform([30, 2, 0], [4, 6, 0], 3)]
    assert not view.is_box
    assert view.strides() is None
    assert view.bounding_box() == ((2, 34), (0, 3))


def test_write() -> None:
    source = np.zeros((10, 2), dtype=int)
    view = LazyArray.from_numpy(source)[runs_transform([6, 1], [3, 2], 2)]
    view.write(np.arange(10).reshape(5, 2))
    expected = np.zeros((10, 2), dtype=int)
    expected[oracle([6, 1], [3, 2])] = np.arange(10).reshape(5, 2)
    np.testing.assert_array_equal(source, expected)


def test_all_runs_empty() -> None:
    view = LazyArray.from_numpy(np.zeros((4, 2)))[runs_transform([1, 3], [0, 0], 2)]
    assert view.shape == (0, 2)
    assert view.result().shape == (0, 2)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"starts": np.array([0, 1]), "lengths": np.array([1])}, ValueError),
        ({"starts": np.array([[0]]), "lengths": np.array([[1]])}, ValueError),
        ({"starts": np.array([0]), "lengths": np.array([-1])}, ValueError),
        ({"starts": np.array([0.5]), "lengths": np.array([1])}, TypeError),
    ],
)
def test_range_map_validation(kwargs: dict[str, Any], error: type[Exception]) -> None:
    with pytest.raises(error):
        RangeMap(0, **kwargs)


def test_transform_validation() -> None:
    m = RangeMap(0, np.array([0]), np.array([3]))
    with pytest.raises(ValueError, match="cover 3 cells"):
        IndexTransform(IndexDomain.from_shape((4,)), (m,))
    with pytest.raises(ValueError, match="out of range"):
        IndexTransform(IndexDomain.from_shape((3,)), (RangeMap(1, m.starts, m.lengths),))


def test_out_of_bounds_run_is_rejected() -> None:
    with pytest.raises(Exception, match="outside"):
        LazyArray.from_numpy(np.zeros((4, 1)))[runs_transform([2], [3], 1)]


def test_equality_and_hash() -> None:
    a = RangeMap(0, np.array([1, 5]), np.array([2, 2]))
    b = RangeMap(0, np.array([1, 5]), np.array([2, 2]))
    assert a == b
    assert hash(a) == hash(b)
    assert a != RangeMap(0, np.array([1, 5]), np.array([2, 3]))


def test_snapshot_ignores_caller_mutation() -> None:
    starts = np.array([1, 5])
    m = RangeMap(0, starts, np.array([2, 2]))
    starts[0] = 9
    assert m.starts.tolist() == [1, 5]
    with pytest.raises(ValueError, match="read-only"):
        m.starts[0] = 9


def test_overflow_raises() -> None:
    with pytest.raises(OverflowError):
        RangeMap(0, np.array([2**63], dtype=np.uint64), np.array([1]))
    m = RangeMap(0, np.array([2**62]), np.array([1]))
    t = IndexTransform(IndexDomain.from_shape((1,)), (m,))
    with pytest.raises(OverflowError):
        t.translate((2**62,))
    inner = IndexTransform(
        IndexDomain(inclusive_min=(-(2**63),), exclusive_max=(2**63 - 1,)),
        (DimensionMap(0, offset=2**62),),
    )
    with pytest.raises(OverflowError):
        t.compose(inner)


def test_translate_moves_runs_and_translate_domain_keeps_them() -> None:
    t = runs_transform([6, 1], [3, 2], 1)
    moved = t.translate((10, 0))
    assert moved.output[0] == RangeMap(0, np.array([16, 11]), np.array([3, 2]))
    assert t.translate_domain_by((4, 0)).output[0] == t.output[0]


def test_adjacent_runs_merge() -> None:
    # Consecutive rows of a batch: one piece, a full slice of the chunk.
    transform = runs_transform([0, 2], [2, 2], 1)
    plan = plan_chunks(transform, dimension_grids_from_chunks((4, 1), shape=(4, 1)))
    axis = plan.partition().sets[0]
    assert isinstance(axis, RunSet)
    assert axis.counts.tolist() == [1]
    (projection,) = list(plan)
    assert isinstance(projection.chunk_transform.output[0], DimensionMap)
    assert projection.coverage == "full"


@pytest.mark.parametrize(
    ("starts", "lengths", "coverage"), [([0], [4], "full"), ([1], [2], "partial")]
)
def test_single_piece_coverage(starts: list[int], lengths: list[int], coverage: str) -> None:
    plan = plan_chunks(
        runs_transform(starts, lengths, 1), dimension_grids_from_chunks((4, 1), shape=(4, 1))
    )
    assert [p.coverage for p in plan] == [coverage]


def test_varying_grid() -> None:
    source = np.arange(20).reshape(20, 1)
    view = LazyArray.from_numpy(source).with_parts_per_axis(((3, 9, 1, 7), (1,)))
    view = view[runs_transform([15, 2, 11], [5, 10, 2], 1)]
    np.testing.assert_array_equal(view.result(), source[oracle([15, 2, 11], [5, 10, 2])])


def test_plan_out_of_bounds_names_coordinate() -> None:
    transform = runs_transform([2], [5], 1)
    with pytest.raises(IndexError, match="index 6"):
        plan_chunks(transform, dimension_grids_from_chunks((4, 1), shape=(4, 1))).partition()


@pytest.mark.parametrize("stride", [-1, 0, 2])
def test_compose_through_non_unit_stride(stride: int) -> None:
    outer = runs_transform([3, 0], [2, 3], 1)
    expanded = outer.expand_ranges()
    inner = IndexTransform(
        IndexDomain.from_shape((5, 1)), (DimensionMap(0, 10, stride), DimensionMap(1))
    )
    assert outer.compose(inner) == expanded.compose(inner)


def test_index_array_structure_and_diagonal() -> None:
    m = RangeMap(0, np.array([0]), np.array([3]))
    assert runs_transform([0], [3], 2).index_array_structure == "orthogonal"
    diagonal = IndexTransform(IndexDomain.from_shape((3,)), (m, DimensionMap(0)))
    with pytest.raises(ValueError, match="diagonal"):
        plan_chunks(diagonal, dimension_grids_from_chunks((2, 2), shape=(4, 4))).partition()


def test_correlated_plan_keeps_transform() -> None:
    m = RangeMap(0, np.array([7, 1]), np.array([2, 3]))
    transform = IndexTransform(
        IndexDomain.from_shape((5,)), (m, ArrayMap(np.array([5, 0, 3, 3, 1])))
    )
    plan = plan_chunks(transform, dimension_grids_from_chunks((3, 4), shape=(10, 6)))
    assert plan.partition().transform == plan.transform
