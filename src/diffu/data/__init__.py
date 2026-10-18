"""Data pipeline: (PAGE/ALTO XML  OR  pre-cropped lines) -> (image, text, writer_id) manifest + split.

Intentionally empty of re-exports (matches the package root, ``diffu/__init__.py``): import from the
submodule that owns the symbol, e.g. ``from diffu.data.dataset import HandwritingLineDataset``.
Keeping it bare keeps the pure-stdlib prep path (parse/ingest) importable without pulling in the
torch-adjacent modules.
"""
