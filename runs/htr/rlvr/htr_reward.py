"""The reward of a page reading: character accuracy and block boxes against the truth, times a completion gate.

One function, ``components``, is used by the reward environment (training), by the evaluation hook (mining, before
and after) and by the tests, so a reading has the same score wherever it is scored. The scorers are NVIDIA's
(``nemotron.recipes.parse.document_rewards``): ``text_quality`` (max(0, 1 - CER) on the visible text),
``box_quality`` (one-to-one IoU matching of block boxes), ``completion_quality`` (loop detection, natural end) and
``weighted_reward``. Only the page-specific parts are ours: what counts as the page's text and where its boxes are.

    reward = (w_text * text + w_boxes * boxes) / (w_text + w_boxes)  *  gate
    gate   = 0 if the answer looped or ran to the token cap,
             0 if it is known not to have ended with EOS,
             1 otherwise (ended with EOS, or the end is not known and nothing shows a failure)

A loop is what NVIDIA's detector finds; with the loop rule's ``min_length_ratio`` it must also make the answer's
text longer than that many times the truth's. Pilot 2's mining (25 Nov) showed why: of 404 sampled answers the
detector called loops, 129 were under 10 % CER, lists whose dot leaders or ditto entries ran a few repeats past the
truth's; at 1.15 times the truth's length 84 stay loops (mean CER 0.81) and 3 of the 129 do. A loop the reader breaks
off early (a short answer) is no longer gated; its missing text keeps its text score low.

A page whose truth has no boxed block has no box component (``boxes`` is None and the text carries all the weight).
"""

from __future__ import annotations

import re

from nemotron.recipes.parse.document_rewards import box_quality, completion_quality, text_quality, weighted_reward
from nemotron.recipes.parse.document_rewards.baselines import _visible  # the text text_quality compares, for its length

WEIGHTS = {"text": 0.8, "boxes": 0.2}
# NVIDIA's detector, with a shorter n-gram than its default (24 words, 4 times): on our test readings the default
# misses the short loops (a word ten times over), and 6 words 4 times flags them without flagging a page under
# 10 % CER (4 readings of the 200 test pages; see the report of 24 Nov 2026). More repeats than the truth has count.
LOOP = {"ngram_size": 6, "min_repeats": 4, "token_run_limit": 32}
DETECTOR = ("ngram_size", "min_repeats", "token_run_limit")


def loop_rule(rule: dict | None) -> dict[str, float]:
    """A loop rule as numbers: the detector's three sizes as whole numbers, ``min_length_ratio`` (if set) as is."""
    rule = LOOP if rule is None else rule
    unknown = set(rule) - {*DETECTOR, "min_length_ratio"}
    if unknown:
        raise ValueError(f"unknown loop settings: {sorted(unknown)}")
    return {name: float(value) if name == "min_length_ratio" else int(value) for name, value in rule.items()}

# The same table markup our dev CER leaves out (diffu_htr_train.nemotron_parse.checkpoint_cer.TABLE), so the text
# component is the transcription and not the LaTeX serialisation (docs/nemotron/parse/rewards.md asks for that).
TABLE = re.compile(r"\\begin\{tabular\}\{[^}]*\}|\\end\{tabular\}|\\linebreak\{\}|\\\\| & ")
NUMBER = r"([0-9]*\.?[0-9]+)"
BLOCK = re.compile(rf"<x_{NUMBER}><y_{NUMBER}>(.*?)<x_{NUMBER}><y_{NUMBER}><class_([^>]+)>", re.S)


def transcription(answer: str) -> str:
    """A target or an answer without table markup; ``text_quality`` takes out the box and class tokens itself."""
    return TABLE.sub(" ", answer or "").replace("#", " ")


def boxes(answer: str) -> list[list[float]]:
    """The ``[x1, y1, x2, y2]`` boxes of the answer's complete blocks, in the page's 0..1 frame, in reading order."""
    return [[float(m[1]), float(m[2]), float(m[4]), float(m[5])] for m in BLOCK.finditer(answer or "")]


def valid(box: list[float]) -> bool:
    """A box with a positive area (``box_quality`` refuses a reference without one)."""
    return box[2] > box[0] and box[3] > box[1]


def components(
    answer: str,
    target: str,
    *,
    weights: dict[str, float] | None = None,
    loop: dict[str, int] | None = None,
    token_ids: list[int] | None = None,
    eos_token_id: int | None = None,
    ended: bool | None = None,
    capped: bool | None = None,
) -> dict[str, float]:
    """Score one answer against the page's target; every value is a number so the logs can average it.

    Args:
        answer: The generated text, special tokens kept (boxes and classes are read from it).
        target: The page's full target in Parse's format.
        weights: ``{"text": .., "boxes": ..}``; ``WEIGHTS`` when None.
        loop: The loop detector's ``ngram_size``, ``min_repeats`` and ``token_run_limit``, and optionally
            ``min_length_ratio`` (a loop must make the answer's text longer than that many times the truth's);
            ``LOOP`` when None.
        token_ids: The generated token ids when the caller has them (the end and token runs are read from them).
        eos_token_id: The tokenizer's EOS id, with ``token_ids``.
        ended: Whether generation ended with EOS, when the caller knows it some other way.
        capped: Whether generation stopped at the token cap, when the caller knows it.

    Returns:
        ``reward`` and its parts: ``text`` (max(0, 1 - cer)), ``cer``, ``wer``, ``chars`` (the truth's length, so
        ``cer * chars`` are the edits and pages can be pooled), ``boxes`` (-1 when the truth has none), ``box_f1``,
        ``predicted_boxes``, ``looped`` (the rule's verdict), ``loop_detected`` (the detector's alone), ``capped``,
        ``ended`` (-1 when not known), ``gate``.
    """
    weights = WEIGHTS if weights is None else weights
    text = text_quality(transcription(answer), transcription(target))
    truth_boxes = [box for box in boxes(target) if valid(box)]
    found = boxes(answer)
    box = box_quality(found, truth_boxes) if truth_boxes else {"score": None}
    rule = loop_rule(loop)
    length_ratio = rule.pop("min_length_ratio", None)
    done = completion_quality(
        answer,
        token_ids=token_ids,
        eos_token_id=eos_token_id,
        eos_reached=ended,
        finish_reason="length" if capped else None,
        reference_text=target,
        **{name: int(value) for name, value in rule.items()},
    )
    looped = bool(done["looped"])
    if looped and length_ratio is not None:
        looped = len(_visible(transcription(answer))) > length_ratio * len(_visible(transcription(target)))
    if done["capped"] or looped:
        gate = 0.0
    else:
        gate = 1.0 if done["eos_reached"] is None else float(done["eos_reached"])
    mixed = weighted_reward({"text": text, "boxes": box}, weights)["score"]
    return {
        "reward": mixed * gate,
        "text": text["score"],
        "cer": text["cer"],
        "wer": text["wer"],
        "chars": float(len(_visible(transcription(target)))),
        "boxes": -1.0 if box["score"] is None else box["score"],
        "box_f1": box.get("f1", -1.0),
        "predicted_boxes": float(len(found)),
        "looped": float(looped),
        "loop_detected": float(done["looped"]),
        "capped": float(done["capped"]),
        "ended": -1.0 if done["eos_reached"] is None else float(done["eos_reached"]),
        "gate": gate,
    }
