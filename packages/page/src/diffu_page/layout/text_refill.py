"""Refill a layout with new text: keep a page's line geometry, write other words in its lines.

A replayed ALTO/PAGE layout (:func:`load_xml_layout`) carries the page's own transcription. :func:`refill` streams
new paragraphs through its running-text lines in reading order, each line taking about as many characters as it
held: the page's writing density (characters per pixel of line width, the median over its fuller lines) times the
line's width. A paragraph ends on a short line and the next starts on a fresh one; a word that overruns a line is
split with ``¬``, as the Swedish GT transcriptions mark a break, when both halves keep at least ``MIN_SPLIT``
characters. Lines beyond the end of the text are dropped. Other regions (page numbers, marginalia, headings,
catchwords) keep their own text.
"""

from __future__ import annotations

import statistics
from collections import deque
from collections.abc import Iterable

from diffu_page.layout.model import RUNNING_TEXT, Layout, LineBox

SPLIT_MARK = "¬"
MIN_SPLIT = 3  # characters each half of a split word keeps
DENSITY_MIN_CHARS = 10  # lines shorter than this (paragraph ends, numbers) don't measure the writing density


def _writing_density(boxes: list[LineBox]) -> float:
    densities = [len(box.text) / box.w for box in boxes if len(box.text) >= DENSITY_MIN_CHARS and box.w > 0]
    if not densities:
        raise ValueError(f"no line of at least {DENSITY_MIN_CHARS} characters to measure the writing density")
    return statistics.median(densities)


class _TextStream:
    """The words of paragraphs, handed out a line at a time; each paragraph starts on a fresh line."""

    def __init__(self, paragraphs: Iterable[str]) -> None:
        self._paragraphs = (paragraph.split() for paragraph in paragraphs if paragraph.strip())
        self._words: deque[str] = deque()

    def line(self, budget: int) -> str | None:
        """The next line of at most ``budget`` characters, or None once the text has run out."""
        if not self._words:
            self._words = deque(next(self._paragraphs, []))
        if not self._words:
            return None
        line = ""
        while self._words:
            word = self._words[0]
            candidate = f"{line} {word}" if line else word
            if len(candidate) <= budget:
                line = candidate
                self._words.popleft()
                continue
            return self._split_into(line, budget)
        return line

    def _split_into(self, line: str, budget: int) -> str:
        """``line`` closed by the head of the next word and ``SPLIT_MARK`` if the word splits well, else as it is.

        An empty line takes the head of even a word too long to split well: every line holds something.
        """
        word = self._words[0]
        room = budget - len(line) - (1 if line else 0) - len(SPLIT_MARK)
        splits_well = room >= MIN_SPLIT and len(word) - room >= MIN_SPLIT
        if line and not splits_well:
            return line
        cut = max(room, 1)
        self._words[0] = word[cut:]
        head = f"{word[:cut]}{SPLIT_MARK}"
        return f"{line} {head}" if line else head


def running_text(layout: Layout) -> str:
    """A layout's running-text lines as one paragraph, the words its transcription broke with ``SPLIT_MARK``
    joined back, ready to refill another page with."""
    lines = " ".join(box.text.strip() for box in layout.boxes if box.region == RUNNING_TEXT)
    return lines.replace(f"{SPLIT_MARK} ", "")


def refill(layout: Layout, paragraphs: Iterable[str]) -> Layout:
    """Rewrite a layout's running-text lines with new text, in reading order.

    Args:
        layout: The page; its running-text lines (region ``body``) set the line widths and the writing
            density, its other lines keep their text.
        paragraphs: The new text, one paragraph per item; blank items are skipped.

    Returns:
        The layout with each running-text line holding the next part of ``paragraphs``, and without the
        running-text lines left over when the text runs out.

    Raises:
        ValueError: If no running-text line holds ``DENSITY_MIN_CHARS`` characters to measure the density by.
    """
    density = _writing_density([box for box in layout.boxes if box.region == RUNNING_TEXT])
    text = _TextStream(paragraphs)
    boxes: list[LineBox] = []
    for box in layout.boxes:
        if box.region != RUNNING_TEXT:
            boxes.append(box)
            continue
        line = text.line(max(1, round(density * box.w)))
        if line is not None:
            boxes.append(box.model_copy(update={"text": line}))
    return layout.model_copy(update={"boxes": boxes})
