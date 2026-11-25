"""The checkpoints' dev error rate: the distance, the draw, and the score of a reading."""

from __future__ import annotations

import json
from pathlib import Path

from diffu_htr_train.nemotron_parse.checkpoint_cer import (
    DevPage,
    dev_pages,
    drawn,
    edits,
    page_reads,
    page_text,
    scored,
    unscored,
)


def page(name: str, volume: str, truth: str = "abcd") -> DevPage:
    return DevPage(id=name, volume=volume, image=f"{name}.jpg", prompt="", truth=truth)


def test_edits_counts_a_swap_a_missing_and_an_added_character() -> None:
    assert edits("kitten", "sitting") == 3
    assert edits("Anno 1689", "Anno 1689") == 0
    assert edits("", "abc") == 3
    assert edits("åäö", "aao") == 3  # a character is a character, whatever its bytes


def test_page_text_leaves_out_boxes_classes_and_table_markup() -> None:
    assert page_text("<x_0.1><y_0.2>Anno\n1689<x_0.5><y_0.6><class_Text>") == "Anno 1689"


def test_drawn_keeps_a_small_volume_whole_and_caps_a_large_one() -> None:
    pages = [page(f"a{k}", "large") for k in range(50)] + [page(f"b{k}", "small") for k in range(3)]

    picked = drawn(pages, per_volume=10, seed=0)

    assert sum(p.volume == "large" for p in picked) == 10
    assert sum(p.volume == "small" for p in picked) == 3
    assert [p.id for p in picked] == [p.id for p in drawn(pages, per_volume=10, seed=0)]  # the same draw every time


def test_scored_gives_the_draw_and_each_volume_their_own_error_rate() -> None:
    pages = [page("a", "one", "abcd"), page("b", "two", "abcdefgh")]

    reads = page_reads(pages, ["abcd", "abcdxxxx"])
    score = scored(250, reads, seconds=1.0)

    assert [(read.id, read.edits, read.chars) for read in reads] == [("a", 0, 4), ("b", 4, 8)]
    assert score.volumes == {"one": 0.0, "two": 0.5}
    assert abs(score.cer - 4 / 12) < 1e-9  # the pages together, not the mean of the volumes


def test_unscored_lists_finished_checkpoints_without_a_score_oldest_first(tmp_path: Path) -> None:
    for step in (250, 500, 750):
        (tmp_path / "output" / f"checkpoint-{step}").mkdir(parents=True)
    for step in (250, 500):  # 750 is still being written: no trainer state yet
        (tmp_path / "output" / f"checkpoint-{step}" / "trainer_state.json").write_text("{}")

    assert [step for step, _ in unscored(tmp_path, done={250})] == [500]


def test_dev_pages_keeps_a_label_whose_text_holds_unicode_line_breaks(tmp_path: Path) -> None:
    label = "Anno 1689\u0085den 3 maj Per Olsson"
    rows = [
        {
            "messages": [{"role": "user", "content": "<image>P"}, {"role": "assistant", "content": label}],
            "images": ["a.jpg"],
        },
        {
            "messages": [{"role": "user", "content": "<image>P"}, {"role": "assistant", "content": "två"}],
            "images": ["b.jpg"],
        },
    ]
    metas = [{"id": "a", "volume": "v1"}, {"id": "b", "volume": "v2"}]
    samples = tmp_path / "set.jsonl"
    samples.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), "utf-8")
    samples.with_name("set.meta.jsonl").write_text("".join(json.dumps(m) + "\n" for m in metas), "utf-8")

    pages = dev_pages(samples)

    assert [(p.id, p.volume) for p in pages] == [("a", "v1"), ("b", "v2")]
    assert pages[0].truth == page_text(label)
