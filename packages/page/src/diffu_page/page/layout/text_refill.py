"""Refill a layout with new text: keep a page's line geometry, write other words in its lines.

A replayed ALTO/PAGE layout (:func:`load_xml_layout`) carries the page's own transcription. :func:`refill` streams
new text through its running-text lines in reading order, each line taking about as many characters as it held: the
page's writing density (characters per pixel of line width, the median over its fuller lines) times the line's
width. A layout whose lines hold no text to measure (one given as boxes) takes ``CHARS_PER_LINE_ASPECT`` characters
per unit of each line's width over its height instead. A blank line in the text ends a paragraph on a short line and
the next starts on a fresh one; a word that overruns a line is split with ``¬``, as the Swedish GT transcriptions mark
a break, when both halves keep at least ``MIN_SPLIT`` characters. Lines beyond the end of the text are dropped. Other
regions (page numbers, marginalia, headings, catchwords) keep their own text. On a page whose running text is
handwritten, its printed lines are a form's print and keep theirs too: only the handwritten lines are rewritten, and
they alone measure the density. A page whose running text is all printed, a printed book's, has its print rewritten.
"""

from __future__ import annotations

import re
import statistics
from collections import deque
from collections.abc import Callable, Iterable

from diffu_page.page.layout.model import Layout, LineKind, TextLine

PARAGRAPH_BREAK = re.compile(r"\n\s*\n")  # a blank line, spaces and all
SPLIT_MARK = "¬"
MIN_SPLIT = 3  # characters each half of a split word keeps
DENSITY_MIN_CHARS = 10  # lines shorter than this (paragraph ends, numbers) don't measure the writing density
# Characters a line holds per unit of its box's width over its height: the median over 24,807 handwritten running-text
# lines of at least DENSITY_MIN_CHARS characters on 492 pages sampled at random from the Lejonet HTR ground truth of
# Swedish and Finnish archives (PAGE XML, 41 collections). Half the lines hold 2.8 to 4.8; a collection's median is
# 2.0 to 5.8.
CHARS_PER_LINE_ASPECT = 3.7


def _line_budget(lines: list[TextLine]) -> Callable[[TextLine], int]:
    """How many characters each of ``lines`` takes: the median writing density of those holding
    ``DENSITY_MIN_CHARS`` characters times its width; with none, ``CHARS_PER_LINE_ASPECT`` times its width over its
    height. Every line takes one character at least."""
    densities = [
        len(line.text) / line.width for line in lines if len(line.text) >= DENSITY_MIN_CHARS and line.width > 0
    ]
    if not densities:
        return lambda line: max(1, round(CHARS_PER_LINE_ASPECT * line.width / line.height))
    density = statistics.median(densities)
    return lambda line: max(1, round(density * line.width))


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
    lines = " ".join(line.text.strip() for line in layout.lines if line.kind is LineKind.BODY)
    return lines.replace(f"{SPLIT_MARK} ", "")


def rewritten(layout: Layout) -> Callable[[TextLine], bool]:
    """Which of a layout's lines refill rewrites: its running-text lines (kind ``body``), and of a page whose running
    text is handwritten only the handwritten ones, the print among them being a form's (a line its template says
    nothing of counts as handwritten); of a page whose running text is all printed, every one."""
    if all(line.printed for line in layout.lines if line.kind is LineKind.BODY):
        return lambda line: line.kind is LineKind.BODY
    return lambda line: line.kind is LineKind.BODY and not line.printed


def refill(layout: Layout, text: str) -> Layout:
    """Rewrite a layout's running-text lines with new text, in reading order.

    Args:
        layout: The page; the lines it rewrites (its running-text lines, kind ``body``, and of a page whose running
            text is handwritten only the handwritten ones, :func:`rewritten`) set the line widths and the writing
            density, its other lines keep their text.
        text: The new text; a blank line ends a paragraph, and the next starts on a fresh line.

    Returns:
        The layout with each line it rewrites holding the next part of ``text``, and without the lines left over
        when the text runs out.
    """
    rewrites = rewritten(layout)
    budget = _line_budget([line for line in layout.lines if rewrites(line)])
    stream = _TextStream(PARAGRAPH_BREAK.split(text))
    lines: list[TextLine] = []
    for line in layout.lines:
        if not rewrites(line):
            lines.append(line)
            continue
        written = stream.line(budget(line))
        if written is not None:
            lines.append(line.model_copy(update={"text": written}))
    return layout.model_copy(update={"lines": lines})
