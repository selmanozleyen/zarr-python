"""Read many lazy views of zarr arrays in one call."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from zarr.core.array import Array, _get_chunk_spec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.config import config as zarr_config
from zarr.core.sync import sync
from zarr.errors import ChunkNotFoundError, LazyIndexingDisabledError

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt
    from zarr_indexing import IndexTransform, LazyArray

__all__ = ["gather", "gather_async"]

type _Reads = dict[tuple[int, ...], list[tuple[IndexTransform, npt.NDArray[Any]]]]


def gather(views: Sequence[LazyArray]) -> list[npt.NDArray[Any]]:
    """Read lazy views of zarr arrays together, one result per view.

    Each view is planned on its array's storage grid, and every stored chunk is
    fetched and decoded once however many views touch it. The reads of one
    array go to its codec pipeline in one
    [`read_gather`][zarr.abc.codec.CodecPipeline.read_gather] call. Each result
    equals `view.result()`.

    Experimental: needs `zarr.config.set({"experimental_lazy_indexing": True})`.
    """
    return sync(gather_async(views))


async def gather_async(views: Sequence[LazyArray]) -> list[npt.NDArray[Any]]:
    """The asynchronous form of [`gather`][zarr.experimental.gather.gather]."""
    if not zarr_config.get("experimental_lazy_indexing"):
        raise LazyIndexingDisabledError(
            "gather is experimental and disabled by default. Enable it with "
            "zarr.config.set({'experimental_lazy_indexing': True}) or "
            "ZARR_EXPERIMENTAL_LAZY_INDEXING=True"
        )
    outs = [np.empty(view.shape, dtype=np.dtype(view.dtype)) for view in views]
    scattered: list[tuple[npt.NDArray[Any], tuple[Any, ...], npt.NDArray[Any]]] = []
    arrays: dict[int, tuple[Array[Any], _Reads]] = {}
    for view, out in zip(views, outs, strict=True):
        array = view.array
        if not isinstance(array, Array):
            raise TypeError(f"gather reads views of zarr arrays, got one of {type(array).__name__}")
        if out.size == 0:
            continue
        _, reads = arrays.setdefault(id(array.async_array), (array, {}))
        for part in view.with_parts_per_axis(array.write_chunk_sizes).parts():
            selection = part.out_selection
            if all(isinstance(s, slice) for s in selection):
                destination = out if len(selection) == 0 else out[selection]
            else:
                destination = np.empty(part.view.shape, dtype=out.dtype)
                scattered.append((out, selection, destination))
            reads.setdefault(part.base_coords, []).append(
                (part.projection.chunk_transform, destination)
            )
    for array, reads in arrays.values():
        await _read(array, reads)
    for out, selection, values in scattered:
        out[selection] = values
    return outs


async def _read(array: Array[Any], reads: _Reads) -> None:
    """Send one array's reads to its codec pipeline, one request per stored chunk."""
    source = array.async_array
    metadata = source.metadata
    config = source.config
    if metadata.zarr_format == 2:
        config = replace(config, order=metadata.order)
    prototype = default_buffer_prototype()
    coords = list(reads)
    results = await source.codec_pipeline.read_gather(
        [
            (
                source.store_path / metadata.encode_chunk_key(chunk),
                _get_chunk_spec(metadata, source._chunk_grid, chunk, config, prototype),
                reads[chunk],
            )
            for chunk in coords
        ]
    )
    if config.read_missing_chunks is False:
        missing = [
            metadata.encode_chunk_key(chunk)
            for chunk, result in zip(coords, results, strict=True)
            if result["status"] == "missing"
        ]
        if missing:
            raise ChunkNotFoundError(
                f"{len(missing)} chunk(s) not found in store '{source.store_path}': "
                + ", ".join(missing)
            )
