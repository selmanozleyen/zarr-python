"""Select runs of consecutive indices with `Intervals`, planned per interval."""

import numpy as np

from zarr_indexing import IntervalSet, Intervals, LazyArray, plan_chunks
from zarr_indexing.grid import dimension_grids_from_chunks

source = np.arange(40 * 3).reshape(40, 3)

# --8<-- [start:intervals]
view = LazyArray.from_numpy(source).oindex[Intervals(lo=[30, 2, 10], hi=[34, 8, 15]), :]
assert (view.result() == source[np.r_[30:34, 2:8, 10:15]]).all()

# Planning splits each interval at chunk boundaries; no index list is built.
grids = dimension_grids_from_chunks((8, 3), shape=source.shape)
rows = plan_chunks(view.transform, grids).partition().sets[0]
assert isinstance(rows, IntervalSet)
assert rows.chunk.tolist() == [0, 1, 3, 4]
# --8<-- [end:intervals]
