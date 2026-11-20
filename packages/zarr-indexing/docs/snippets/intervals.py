"""Select runs of consecutive indices with `Intervals`."""

import numpy as np

from zarr_indexing import Intervals, LazyArray

source = np.arange(40 * 3).reshape(40, 3)

# --8<-- [start:intervals]
view = LazyArray.from_numpy(source).oindex[Intervals(lo=[30, 2, 10], hi=[34, 8, 15]), :]
assert (view.result() == source[np.r_[30:34, 2:8, 10:15]]).all()
# --8<-- [end:intervals]
