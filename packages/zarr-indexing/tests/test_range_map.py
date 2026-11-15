from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import zarr_indexing._composition
import zarr_indexing.chunk_resolution
import zarr_indexing.output_map
from zarr_indexing import (
    ArrayMap,
    BoundsCheckError,
    ConstantMap,
    DimensionMap,
    IndexDomain,
    IndexTransform,
    LazyArray,
    RangeMap,
    RangeSet,
    plan_chunks,
)
from zarr_indexing.grid import dimension_grids_from_chunks
from zarr_indexing.testing.strategies import (
    basic_selections,
    orthogonal_selections,
    vectorized_selections,
)


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
    assert not isinstance(view.transform.output[0], ArrayMap)
    np.testing.assert_array_equal(view.result(), expected)
    np.testing.assert_array_equal(view.unpartitioned().result(), expected)


@pytest.fixture
def expanded_cells(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Cells expanded from runs during the test, one entry per expansion."""
    cells: list[int] = []
    real = zarr_indexing.output_map._expand_runs

    def spy(starts: Any, lengths: Any) -> Any:
        cells.append(int(lengths.sum()))
        return real(starts, lengths)

    for module in (
        zarr_indexing.output_map,
        zarr_indexing._composition,
        zarr_indexing.chunk_resolution,
    ):
        monkeypatch.setattr(module, "_expand_runs", spy)
    return cells


def test_plan_is_linear_in_runs_not_cells(expanded_cells: list[int]) -> None:
    n = 10**12
    m = RangeMap(0, np.array([0, n - 10**9]), np.array([10**9, 10**9]))
    transform = IndexTransform(IndexDomain.from_shape((m.size,)), (m,))
    (axis,) = (
        plan_chunks(transform, dimension_grids_from_chunks((10**6,), shape=(n,))).partition().sets
    )
    assert isinstance(axis, RangeSet)
    assert axis.extent.size == 2000
    assert int(axis.extent.sum()) == m.size
    assert expanded_cells == []


def test_run_set_pieces() -> None:
    # Runs [6, 9) and [1, 3) over chunks of 4: three pieces, one per chunk.
    transform = runs_transform([6, 1], [3, 2], 1)
    plan = plan_chunks(transform, dimension_grids_from_chunks((4, 1), shape=(10, 1)))
    axis = plan.partition().sets[0]
    assert isinstance(axis, RangeSet)
    assert axis.chunk.tolist() == [0, 1, 2]
    assert axis.local_start.tolist() == [1, 2, 0]
    assert axis.extent.tolist() == [2, 2, 1]
    assert axis.origin.tolist() == [3, 0, 2]
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
    np.testing.assert_array_equal(
        transform[1:4].apply_many(points[1:4]), expanded[1:4].apply_many(points[1:4])
    )
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
    with pytest.raises(BoundsCheckError, match="outside"):
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
    with pytest.raises(OverflowError, match="cells"):
        RangeMap(0, np.zeros(3, dtype=np.intp), np.array([2**63 - 1, 2**63 - 1, 2]))
    with pytest.raises(OverflowError, match="ends past"):
        RangeMap(0, np.array([2**63 - 5]), np.array([10]))
    assert RangeMap(0, np.array([2**63 - 5]), np.array([5])).bounds() == (2**63 - 5, 2**63 - 1)
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
    assert isinstance(axis, RangeSet)
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


SELECTIONS = {
    "basic": (basic_selections, lambda v: v),
    "orthogonal": (orthogonal_selections, lambda v: v.oindex),
    "vectorized": (vectorized_selections, lambda v: v.vindex),
}


@settings(deadline=None)
@given(case=runs_case(), mode=st.sampled_from(sorted(SELECTIONS)), data=st.data())
def test_selection_on_run_view_matches_plain_view(
    case: tuple[int, int, list[int], list[int], tuple[int, int]], mode: str, data: st.DataObject
) -> None:
    # The oracle is the same selection on a view of the materialized rows, with no runs.
    nrows, ncols, starts, lengths, parts = case
    assume(sum(lengths) > 0)  # the selection strategies need a nonempty axis
    source = np.arange(nrows * ncols).reshape(nrows, ncols)
    view = LazyArray.from_numpy(source).with_parts(parts)[runs_transform(starts, lengths, ncols)]
    plain = LazyArray.from_numpy(source[oracle(starts, lengths)])
    strategy, accessor = SELECTIONS[mode]
    selection = data.draw(strategy(view.shape))
    np.testing.assert_array_equal(
        accessor(view)[selection].result(), accessor(plain)[selection].result()
    )


def run_view() -> LazyArray:
    return LazyArray.from_numpy(np.arange(40 * 3).reshape(40, 3))[
        runs_transform([30, 2, 10], [4, 6, 5], 3)
    ]


ROWS = np.arange(40 * 3).reshape(40, 3)[oracle([30, 2, 10], [4, 6, 5])]


@pytest.mark.parametrize(
    ("select", "kind"),
    [
        (lambda v: v[:, 1:3], RangeMap),  # another axis: runs untouched
        (lambda v: v[2:12], RangeMap),  # a window across runs
        (lambda v: v[5:9], DimensionMap),  # a window inside one run is a slice
        (lambda v: v[3], ConstantMap),
        (lambda v: v[::2], ArrayMap),  # strided: only the selected positions
        (lambda v: v[::-1], ArrayMap),
        (lambda v: v.oindex[[14, 0, 7], :], ArrayMap),
        (lambda v: v.vindex[[1, 2], [0, 2]], ArrayMap),
    ],
    ids=[
        "other-axis",
        "window",
        "window-one-run",
        "int",
        "strided",
        "reversed",
        "oindex",
        "vindex",
    ],
)
def test_selection_keeps_runs_where_it_can(select: Any, kind: type) -> None:
    view = select(run_view())
    expected = select(LazyArray.from_numpy(ROWS)).result()
    assert isinstance(view.transform.output[0], kind)
    np.testing.assert_array_equal(view.result(), expected)


def test_window_inside_one_run_is_a_box() -> None:
    view = run_view()[5:9]
    assert view.is_box
    assert view.strides() == (1, 1)


def test_newaxis_renumbers_runs() -> None:
    view = run_view()[None, 2:12]
    (m,) = [m for m in view.transform.output if isinstance(m, RangeMap)]
    assert m.input_dimension == 1
    np.testing.assert_array_equal(view.result(), ROWS[None, 2:12])


def test_window_of_huge_runs_does_not_expand(expanded_cells: list[int]) -> None:
    m = RangeMap(0, np.array([0, 10**12]), np.array([10**9, 10**9]))
    transform = IndexTransform(IndexDomain.from_shape((m.size,)), (m,))
    window = transform[10**9 - 2 : 10**9 + 3]
    assert window.output[0] == RangeMap(0, np.array([10**9 - 2, 10**12]), np.array([2, 3]))
    assert transform[10**9 + 7].output[0] == ConstantMap(10**12 + 7)
    assert expanded_cells == []


def test_oindex_on_other_axis_does_not_expand_runs(expanded_cells: list[int]) -> None:
    # Rows by genes: columns picked first, then a runs key over the rows.
    source = np.arange(40 * 3).reshape(40, 3)
    view = LazyArray.from_numpy(source).oindex[:, [0, 2]][runs_transform([30, 2, 10], [4, 6, 5], 2)]
    assert isinstance(view.transform.output[0], RangeMap)
    assert expanded_cells == []
    np.testing.assert_array_equal(
        view.result(), source[np.ix_(oracle([30, 2, 10], [4, 6, 5]), [0, 2])]
    )


def test_runs_onto_runs_stay_runs(expanded_cells: list[int]) -> None:
    view = run_view()[runs_transform([12, 0, 3], [3, 2, 5], 3)]
    assert isinstance(view.transform.output[0], RangeMap)
    assert expanded_cells == []
    np.testing.assert_array_equal(view.result(), ROWS[oracle([12, 0, 3], [3, 2, 5])])


def test_empty_key_onto_run_view() -> None:
    # An empty key may address coordinates outside the view, as an empty slice may.
    key = IndexTransform(IndexDomain.from_shape((20, 0)), (DimensionMap(0), DimensionMap(1)))
    assert run_view()[key].result().shape == (20, 0)


def test_merge_needs_contiguous_request_positions() -> None:
    # [0, 2) and [2, 4) touch in storage but are not neighbours in the request.
    source = np.arange(8).reshape(8, 1)
    view = LazyArray.from_numpy(source).with_parts((4, 1))[runs_transform([0, 5, 2], [2, 1, 2], 1)]
    np.testing.assert_array_equal(view.result(), source[[0, 1, 5, 2, 3]])


@st.composite
def run_views(draw: st.DrawFn) -> tuple[LazyArray, np.ndarray[Any, Any]]:
    """A view holding runs on any axis of a rank 1-3 domain with a nonzero origin, and its values."""
    rank = draw(st.integers(1, 3))
    k = draw(st.integers(0, rank - 1))
    extents = [draw(st.integers(1, 3)) for _ in range(rank)]
    extents[k] = draw(st.integers(1, 20))
    starts = draw(st.lists(st.integers(0, extents[k] - 1), max_size=6))
    lengths = [draw(st.integers(0, extents[k] - s)) for s in starts]
    origins = [draw(st.integers(-5, 5)) for _ in range(rank)]
    parts = tuple(draw(st.integers(1, e)) for e in extents)
    source = np.arange(int(np.prod(extents))).reshape(extents)
    m = RangeMap(k, np.array(starts, dtype=np.intp), np.array(lengths, dtype=np.intp))
    shape = [m.size if j == k else e for j, e in enumerate(extents)]
    domain = IndexDomain(
        inclusive_min=tuple(origins),
        exclusive_max=tuple(o + n for o, n in zip(origins, shape, strict=True)),
    )
    output = tuple(m if j == k else DimensionMap(j, offset=-origins[j]) for j in range(rank))
    view = LazyArray.from_numpy(source).with_parts(parts)[IndexTransform(domain, output)]
    return view, np.take(source, oracle(starts, lengths), axis=k)


def runs_key(
    domain: IndexDomain, axis: int, starts: list[int], lengths: list[int], origin: int
) -> IndexTransform:
    """A key reading runs at positions of `axis` of a view with `domain`, in its literal frame."""
    lo = domain.inclusive_min[axis]
    m = RangeMap(axis, np.array(starts, dtype=np.intp) + lo, np.array(lengths, dtype=np.intp))
    key_min = list(domain.inclusive_min)
    key_max = list(domain.exclusive_max)
    key_min[axis], key_max[axis] = origin, origin + m.size
    output = tuple(m if j == axis else DimensionMap(j) for j in range(domain.ndim))
    return IndexTransform(
        IndexDomain(inclusive_min=tuple(key_min), exclusive_max=tuple(key_max)), output
    )


@settings(deadline=None)
@given(case=run_views(), data=st.data())
def test_chained_selections_match_numpy(
    case: tuple[LazyArray, np.ndarray[Any, Any]], data: st.DataObject
) -> None:
    # Every step is checked against the same step on a plain view of the values.
    view, values = case
    plain = LazyArray.from_numpy(values)
    np.testing.assert_array_equal(view.result(), values)
    for _ in range(data.draw(st.integers(1, 3))):
        if view.ndim == 0 or 0 in view.shape:
            break
        step = data.draw(st.sampled_from([*sorted(SELECTIONS), "runs"]))
        if step == "runs":
            axis = data.draw(st.integers(0, view.ndim - 1))
            extent = view.shape[axis]
            starts = data.draw(st.lists(st.integers(0, extent - 1), max_size=4))
            lengths = [data.draw(st.integers(0, extent - s)) for s in starts]
            origin = data.draw(st.integers(-5, 5))
            view = view[runs_key(view.transform.domain, axis, starts, lengths, origin)]
            plain = LazyArray.from_numpy(
                np.take(plain.result(), oracle(starts, lengths), axis=axis)
            )
        else:
            strategy, accessor = SELECTIONS[step]
            selection = data.draw(strategy(view.shape))
            view, plain = accessor(view)[selection], accessor(plain)[selection]
        np.testing.assert_array_equal(view.result(), plain.result())


def test_pickle_round_trip_owns_read_only_arrays() -> None:
    import pickle

    m = pickle.loads(pickle.dumps(RangeMap(0, np.array([1, 7]), np.array([5, 2]))))
    assert m == RangeMap(0, np.array([1, 7]), np.array([5, 2]))
    assert m.size == 7
    assert not m.starts.flags.writeable
    assert not m.lengths.flags.writeable


def test_map_to_json_refuses_and_transform_to_json_expands() -> None:
    transform = runs_transform([6, 1], [3, 2], 1)
    with pytest.raises(TypeError, match="IndexTransform.to_json"):
        transform.output[0].to_json()
    assert transform.to_json()["output"][0]["index_array"] == [[6], [7], [8], [1], [2]]


@pytest.mark.parametrize("reader", ["basic_reader", "unit_step_reader", "numpy_reader"])
def test_readers_accept_runs_directly(reader: str) -> None:
    import zarr_indexing

    transform = IndexTransform(
        IndexDomain.from_shape((5,)), (RangeMap(0, np.array([6, 1]), np.array([3, 2])),)
    )
    out = np.empty(5, dtype=int)
    getattr(zarr_indexing, reader).read_into(
        np.arange(10), zarr_indexing.ReadContext(transform), out
    )
    assert out.tolist() == [6, 7, 8, 1, 2]


def test_dask_token_is_compact(expanded_cells: list[int]) -> None:
    pytest.importorskip("dask")
    m = RangeMap(0, np.array([0, 10**6]), np.array([10**6, 10**6]))
    source = LazyArray.from_numpy(np.zeros(2 * 10**6))
    view = source[IndexTransform(IndexDomain.from_shape((m.size,)), (m,))]
    view.__dask_tokenize__()
    assert expanded_cells == []
