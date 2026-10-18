"""Lance line-bank — pre-rendered, CER-gated lines for CPU-only page composition.

The 1M-page lever: lines are rendered + TrOCR-gated ONCE (the line pipeline in :mod:`diffu_runner.pipeline`
writes them to a Lance dataset) and pages become pure CPU compose over those fixed pixels — re-rendering
every line per page (~79 GPU-days at 1M pages) collapses to about a CPU-day of seating. :class:`LineBank`
keeps a light COLUMN INDEX in RAM (text/cer/ids — never ``png``, which is what makes the bank ~44 GB) and
reads pixel rows lazily via ``take``. Two sampling modes, auto-detected from the schema:

- **coherent** (``doc_id`` + ``line_idx`` columns): a page draws CONSECUTIVE surviving lines of one
  document — coherent in prose AND hand. A gap in ``line_idx`` is a line the bank gate rejected; a run
  ends there, so a page never silently skips a line mid-passage.
- **random bank** (the existing 344k-line bank: just text/png/cer/…): a page draws one writer's
  (``style_path``) lines — coherent in HAND though not in prose. With no ``style_path`` column either,
  the whole bank is one pool (no coherence to recover).

No ray imports — unit-testable off-cluster, like :mod:`diffu_runner.actors`.
"""

from __future__ import annotations

import io
import logging
import random
from collections.abc import Callable, Hashable, Sequence

import lance
from PIL import Image

logger = logging.getLogger(__name__)


def contiguous_runs(doc_ids: Sequence[Hashable], line_idxs: Sequence[int]) -> list[list[int]]:
    """Per ``doc_id``, the maximal runs of CONSECUTIVE ``line_idx``, as lists of dataset ROW positions
    (sorted by ``line_idx`` within a run). A gap in ``line_idx`` marks a line the bank's quality gate
    rejected — a run must NOT bridge it, or a composed page would skip a line of prose mid-passage."""
    by_doc: dict[Hashable, list[tuple[int, int]]] = {}
    for row, (doc, idx) in enumerate(zip(doc_ids, line_idxs, strict=True)):
        by_doc.setdefault(doc, []).append((int(idx), row))
    runs: list[list[int]] = []
    for pairs in by_doc.values():
        pairs.sort()  # bank row order is write order, not document order — restore the passage order
        run = [pairs[0][1]]
        prev = pairs[0][0]
        for idx, row in pairs[1:]:
            if idx == prev + 1:
                run.append(row)
            else:
                runs.append(run)
                run = [row]
            prev = idx
        runs.append(run)
    return runs


def gate_stored_cers(
    cers: Sequence[float], *, line_cer_max: float, max_bad_frac: float
) -> tuple[bool, float]:
    """The bank-mode page gate — pure math over the used lines' STORED generation-time CERs.

    No recognizer runs here: every bank line was TrOCR-read ONCE at bank build and its CER travelled with
    the row. Re-reading the COMPOSED page would re-measure the compositor (seating, aging, the scan pass),
    not the ink — the augmentation-audit harness owns that question. Verdict shape mirrors the render-path
    gate: a line is bad above ``line_cer_max``; the page is rejected when the bad FRACTION exceeds
    ``max_bad_frac``. Returns ``(keep, page_cer)`` where ``page_cer`` is the mean stored CER."""
    if not cers:
        return False, 1.0  # nothing seated — nothing worth keeping
    bad_frac = sum(c > line_cer_max for c in cers) / len(cers)
    return bad_frac <= max_bad_frac, float(sum(cers) / len(cers))


def empirical_measure(lines: Sequence[tuple[str, float, Image.Image]]) -> Callable[[str, int], int]:
    """A layout ``Measure`` callback built from fetched bank lines — the bank ink IS the truth.

    The render path wraps with the model's per-string width predictor; a bank line's pixels already exist,
    so its exact width at any height is ``ink.width * height / ink.height``. The wrapper, though, measures
    arbitrary CANDIDATE strings (prefixes of joined paragraphs), so we reduce the run to its mean char
    width and scale: aggregate-exact, per-string approximate (handwriting is not monospace). That is
    enough — the wrapper only needs statistically-right fits, and the compositor's slot-fitting absorbs
    the per-line residual when the ink is finally seated."""
    chars = sum(len(text) for text, _, _ in lines)
    width_per_unit_h = sum(pil.width / pil.height for _, _, pil in lines)  # widths normalised to height=1
    px_per_char = width_per_unit_h / max(1, chars)

    def measure(text: str, height: int) -> int:
        return max(1, round(len(text) * px_per_char * height))

    return measure


class LineBank:
    """Random access into a pre-generated, CER-gated Lance line bank (see the module docstring).

    The constructor loads only the light index columns into RAM (~1M rows of text/float/ids fits easily);
    the heavy ``png`` column stays on disk and is read per-page by :meth:`fetch`.
    """

    def __init__(self, path: str, *, min_writer_lines: int = 12) -> None:
        # min_writer_lines: a page wants tens of lines from ONE hand — a writer with fewer rows can't
        # fill one, so tiny groups fall out of the random-bank pool instead of yielding stub pages.
        self._ds = lance.dataset(path)
        names = set(self._ds.schema.names)
        missing = {"text", "cer", "png"} - names
        if missing:
            raise ValueError(f"line bank at {path} lacks required columns {sorted(missing)}")
        index_cols = ["text", "cer"] + [c for c in ("style_path", "doc_id", "line_idx") if c in names]
        tbl = self._ds.to_table(columns=index_cols)  # the COLUMN INDEX — never the ~44 GB png column
        self.texts: list[str] = tbl.column("text").to_pylist()
        self.cers: list[float] = tbl.column("cer").to_pylist()
        self.doc_ids: list[Hashable] | None = tbl.column("doc_id").to_pylist() if "doc_id" in names else None
        self.line_idxs: list[int] | None = tbl.column("line_idx").to_pylist() if "line_idx" in names else None
        self._runs: list[list[int]] | None = None
        self._writers: list[list[int]] | None = None
        if self.doc_ids is not None and self.line_idxs is not None:
            self._runs = contiguous_runs(self.doc_ids, self.line_idxs)
            # short scraps (isolated survivors between gate rejects) would be CYCLED across a whole
            # page — the same sentence and pixels stamped dozens of times; keep them out of the pool
            usable = [r for r in self._runs if len(r) >= 4]
            self._runs = usable or self._runs  # a degenerate bank still composes, just repetitively
            logger.info(
                "line bank %s: coherent, %d runs (>=4 lines) over %d rows",
                path,
                len(self._runs),
                len(self.texts),
            )
        else:
            styles: list[str] = (
                tbl.column("style_path").to_pylist() if "style_path" in names else [""] * len(self.texts)
            )
            by_writer: dict[str, list[int]] = {}
            for row, style in enumerate(styles):
                by_writer.setdefault(style, []).append(row)
            pools = [rows for rows in by_writer.values() if len(rows) >= min_writer_lines]
            # no writer big enough (or no style_path column at all): one whole-bank pool — hand
            # coherence is unrecoverable then, but pages must still compose
            self._writers = pools or [list(range(len(self.texts)))]
            if pools:
                logger.info("line bank %s: %d writer pools over %d rows", path, len(pools), len(self.texts))
            else:
                logger.warning(
                    "line bank %s: NO doc keys and no writer pool >= %d lines — one whole-bank pool, "
                    "pages will MIX HANDS (rebuild the bank with --coherent for hand+prose coherence)",
                    path, min_writer_lines,
                )  # fmt: skip

    @property
    def coherent(self) -> bool:
        """True when the bank carries ``doc_id``/``line_idx`` — pages can be coherent in PROSE, not
        just in hand."""
        return self._runs is not None

    def __len__(self) -> int:
        return len(self.texts)

    def sample_run(self, rng: random.Random, n_lines: int) -> list[int]:
        """Dataset row indices for one page's lines (``n_lines``, or fewer if the drawn run/pool is
        shorter).

        Coherent bank: a contiguous run picked WEIGHTED BY LENGTH (uniform over lines, so long clean
        stretches aren't under-sampled), then a random ``n_lines`` window of it — consecutive
        ``line_idx`` of one document. Random bank: one writer's pool (also length-weighted), then a
        random ``n_lines`` subset of that hand's rows."""
        if n_lines < 1:
            raise ValueError(f"n_lines must be >= 1, got {n_lines}")
        if self._runs is not None:
            run = rng.choices(self._runs, weights=[len(r) for r in self._runs], k=1)[0]
            if len(run) <= n_lines:
                return list(run)
            start = rng.randrange(len(run) - n_lines + 1)
            return run[start : start + n_lines]
        if self._writers is None:  # unreachable: __init__ always builds exactly one of runs/writers
            raise RuntimeError("LineBank index not built")
        pool = rng.choices(self._writers, weights=[len(p) for p in self._writers], k=1)[0]
        return list(pool) if len(pool) <= n_lines else rng.sample(pool, n_lines)

    def fetch(self, indices: Sequence[int]) -> list[tuple[str, float, Image.Image]]:
        """``(text, stored CER, decoded PIL line)`` per row, in ``indices`` order — the ONLY reader of
        the heavy ``png`` column (one page's worth of rows at a time, never the whole bank)."""
        tbl = self._ds.take(list(indices), columns=["text", "cer", "png"])
        return [
            (text, float(cer), Image.open(io.BytesIO(png)).convert("RGB"))
            for text, cer, png in zip(
                tbl.column("text").to_pylist(),
                tbl.column("cer").to_pylist(),
                tbl.column("png").to_pylist(),
                strict=True,
            )
        ]
