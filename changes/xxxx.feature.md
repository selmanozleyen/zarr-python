Add an experimental `.lazy` accessor to `Array` and `AsyncArray`. `arr.lazy[selection]` returns a lazy
view, backed by the `zarr-indexing` package, that composes further selections without IO and reads
on `.result()`. Opt in with `zarr.config.set({"experimental_lazy_indexing": True})` (or
`ZARR_EXPERIMENTAL_LAZY_INDEXING=True`) and install `zarr[lazy]`. Eager indexing is unchanged.
