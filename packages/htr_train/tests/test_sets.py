"""The sets htr-train builds: a dev split of whole volumes (or another split's volumes), with each side's options; Jev's
pairs from a line set's lines of one side (the GT text, one wrong candidate drawn from the rules and a swapped line,
and readers' reads, positive where they are the GT; each rendered by our frozen renderer, in our English prompt),
refusing a line of the other side and a read of a line the set does not hold, a line whose crop cannot be read recorded
and left out; and a draft's responses from its served target, a failed prompt recorded and the rest kept, a set whose
prompts mostly fail refused. The CLI splits a set (refusing a share of no volume) and builds pairs (crops as cut at
height 0; refused without the renderer's font)."""

from __future__ import annotations

import json
import random
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from PIL import Image
from typer.testing import CliRunner

from diffu_htr_train.cli import app
from diffu_htr_train.jev.prompt import INSTRUCTION, model_prompt, prompt_digest, sample_prompt
from diffu_htr_train.jev.render import render_text, renderer_version
from diffu_htr_train.sets import SetRefused
from diffu_htr_train.sets.dev_split import split_dev, volumes_of
from diffu_htr_train.sets.dspark import ResponsesSettings, write_responses
from diffu_htr_train.sets.jev_pairs import PairSettings, Side, build_pairs
from diffu_htr_train.sets.perturb import Perturbation, perturbed

if TYPE_CHECKING:
    from conftest import MakeSet, SetLine


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


VOLUMES = [("GT-A", "V1"), ("GT-A", "V2"), ("GT-B", "V3"), ("GT-B", "V4")]


def _four_volumes(make_set: MakeSet, name: str = "lines") -> Path:
    return make_set(
        name,
        [
            (f"{volume}_0_{k}", f"line {k} of {volume}", collection, volume, "train")
            for collection, volume in VOLUMES
            for k in range(2)
        ],
    )


def test_split_dev_holds_out_whole_volumes_and_records_each_sides_options(make_set: MakeSet) -> None:
    lines = _four_volumes(make_set)
    lines.with_name("lines.options.json").write_text('{"task": "lines"}\n', encoding="utf-8")

    split = split_dev(lines, share=0.25, seed=0)

    held = random.Random(0).sample(sorted(f"{c}/{v}" for c, v in VOLUMES), 1)
    assert split.dev_volumes == held
    assert {
        f"{row['collection']}/{row['volume']}" for row in _rows(split.dev.with_name("lines.dev.meta.jsonl"))
    } == set(held)
    assert [row["id"] for row in _rows(split.train.with_name("lines.train.meta.jsonl"))] == [
        f"{volume}_0_{k}" for collection, volume in VOLUMES if f"{collection}/{volume}" not in held for k in range(2)
    ]
    assert [row["messages"][1]["content"] for row in _rows(split.dev)] == [
        f"line {k} of {held[0].split('/')[1]}" for k in range(2)
    ]
    options = json.loads(split.dev.with_name("lines.dev.options.json").read_text(encoding="utf-8"))
    assert (options["cut_from"], options["options"], options["share"], options["split"]["dev_volumes"]) == (
        str(lines.resolve()),
        {"task": "lines"},
        0.25,
        held,
    )


def test_split_dev_holds_out_another_splits_volumes_for_a_set_mixed_with_it(make_set: MakeSet) -> None:
    lines, regions = _four_volumes(make_set, "lines"), _four_volumes(make_set, "regions")
    lines_split = split_dev(lines, share=0.5, seed=3)

    split = split_dev(regions, share=0.25, seed=0, like=volumes_of(lines_split.dev))

    assert split.dev_volumes == lines_split.dev_volumes


def test_split_dev_refuses_to_cut_like_a_split_it_shares_no_volume_with_and_writes_nothing(make_set: MakeSet) -> None:
    lines = _four_volumes(make_set)

    with pytest.raises(SetRefused, match="shares no volume with the dev split"):
        split_dev(lines, share=0.25, seed=0, like={"GT-Z/elsewhere"})

    assert not lines.with_name("lines.dev.jsonl").exists()


@pytest.mark.parametrize(
    ("lines", "refusal"),
    [
        pytest.param(
            [("a", "x", "GT-A", "V1", "train"), ("b", "y", "GT-A", "V2", "train")],
            "has 2 volumes: a dev split by volume needs at least 3",
            id="too-few-volumes",
        ),
        pytest.param([("a", "x", "", "", "train")], "a names no volume", id="no-volume"),
        pytest.param([("a", "x", "GT-A", "V1", "test")], "a is a real sample of the test side", id="test-line"),
        pytest.param([("a", "x", "GT-A", "V1", None)], "a is a real sample of the unsplit side", id="unsplit"),
    ],
)
def test_split_dev_refuses_a_set_it_cannot_split_by_volume(
    make_set: MakeSet, lines: list[SetLine], refusal: str
) -> None:
    path = make_set("lines", lines)

    with pytest.raises(SetRefused, match=refusal):
        split_dev(path, share=0.5, seed=0)


def test_split_dev_cli_writes_each_side_of_a_set(make_set: MakeSet) -> None:
    lines = _four_volumes(make_set)

    result = CliRunner().invoke(app, ["split-dev", str(lines), "--share", "0.25"])

    assert result.exit_code == 0, result.output
    assert [len(_rows(lines.with_name(f"lines.{side}.jsonl"))) for side in ("train", "dev")] == [6, 2]


def test_split_dev_cli_refuses_a_share_of_no_volume(make_set: MakeSet) -> None:
    lines = _four_volumes(make_set)

    result = CliRunner().invoke(app, ["split-dev", str(lines), "--share", "0"])

    assert (result.exit_code, lines.with_name("lines.dev.jsonl").exists()) == (2, False)


def test_jev_prompt_puts_the_text_between_the_crop_and_its_rendering_then_the_english_instruction() -> None:
    assert (
        sample_prompt("Anno 1715"),
        model_prompt("x").endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    ) == (
        f"<image>\nOCR text:\nAnno 1715\n<image>\n{INSTRUCTION}",
        True,
    )


def test_perturbed_gives_each_rule_once_in_the_seeds_order_and_never_the_text() -> None:
    made = perturbed("Anno 1715 den 4 maj", 7, random.Random(0))

    assert made == [
        (Perturbation.WORD_SWAP, "Anno 1715 den maj 4"),
        (Perturbation.DROP, "Anno 1715 den 4 ma"),
        (Perturbation.INSERT, "Anno 1715o den 4 maj"),
        (Perturbation.CONFUSABLE, "Anno 1715 dcn 4 maj"),
        (Perturbation.DIGIT, "Anno 1715 den 3 maj"),
        (Perturbation.WORD_DROP, "Anno 1715 den 4"),
        (Perturbation.TRUNCATE, "1715 den 4 maj"),
    ]


def test_perturbed_gives_only_the_rules_a_text_admits() -> None:
    made = perturbed("x", 7, random.Random(0))  # one letter: no confusable, digit, word or second letter to drop

    assert [kind for kind, _ in made] == [Perturbation.INSERT]


def test_build_pairs_writes_the_gt_text_one_wrong_candidate_and_the_reads_for_each_line(
    tmp_path: Path, make_set: MakeSet, jev_font: Path
) -> None:
    lines = make_set(
        "lines",
        [("p_0_0", "Anno 1715", "GT-A", "V1", "train"), ("p_0_1", "den 4 maj", "GT-A", "V1", "train")],
        size=(400, 80),
    )
    reads = tmp_path / "reads.jsonl"
    reads.write_text(
        json.dumps({"id": "p_0_0", "reader": "trocr", "text": "Amno 1715"})  # an error no rule makes
        + "\n"
        + json.dumps({"id": "p_0_1", "reader": "trocr", "text": "den  4 maj"})  # the GT, normalised: a positive
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "pairs" / "jev.jsonl"
    settings = PairSettings(negatives=1, height=40, seed=0, font=jev_font)

    built = build_pairs(lines, out, settings=settings, reads=[reads])

    rows, metas = _rows(out), _rows(out.with_name("jev.meta.jsonl"))
    crops, renders = out.with_name("jev_crops").resolve(), out.with_name("jev_renders").resolve()
    origins = [meta["candidate"].split(":")[0] for meta in metas]
    assert [(meta["line"], origin, row["label"]) for meta, origin, row in zip(metas, origins, rows, strict=True)] == [
        ("p_0_0", "gt", 1),
        ("p_0_0", "perturb", 0),
        ("p_0_0", "reader", 0),
        ("p_0_1", "gt", 1),
        ("p_0_1", "swap", 0),
        ("p_0_1", "reader", 1),
    ]
    assert [metas[k]["candidate"] for k in (1, 4)] == ["perturb:word-drop", "swap"]
    texts = [row["messages"][0]["content"] for row in rows]
    assert [texts[k] for k in (0, 2, 3, 5)] == [
        sample_prompt(text) for text in ("Anno 1715", "Amno 1715", "den 4 maj", "den  4 maj")
    ]
    assert [row["images"] for row in rows] == [
        [str(crops / f"{line}.jpg"), str(renders / f"{line}_{n}.png")] for line in ("p_0_0", "p_0_1") for n in range(3)
    ]
    assert [(meta["source"], meta["page"], meta["volume"], meta["renderer"], meta["prompt"]) for meta in metas] == [
        ("real", "p", "V1", renderer_version(jev_font), prompt_digest())
    ] * 6
    assert [metas[k]["cer"] for k in (2, 5)] == [round(1 / 9, 4), round(1 / 9, 4)]
    assert [_size(crops / f"{line}.jpg") for line in ("p_0_0", "p_0_1")] == [(200, 40), (200, 40)]
    assert _size(renders / "p_0_0_0.png") == (200, 40)
    assert (built.positives, built.negatives, built.skipped) == (3, 3, 0)


@pytest.mark.parametrize(
    ("split", "side", "refusal"),
    [
        pytest.param(
            "test", Side.TRAIN, "a real line of the test side: a train pair set takes train lines only", id="test-line"
        ),
        pytest.param(None, Side.TRAIN, "a real line of the unsplit side", id="unsplit-line"),
        pytest.param(
            "train",
            Side.TEST,
            "a real line of the train side: a test pair set takes test lines only",
            id="train-line-in-test-pairs",
        ),
    ],
)
def test_build_pairs_refuses_a_line_of_the_other_side(
    tmp_path: Path, make_set: MakeSet, jev_font: Path, split: str | None, side: Side, refusal: str
) -> None:
    lines = make_set("lines", [("p_0_0", "Anno 1715", "GT-A", "V1", split)])

    with pytest.raises(SetRefused, match=refusal):
        build_pairs(lines, tmp_path / "jev.jsonl", settings=PairSettings(side=side, font=jev_font), reads=[])


def test_build_pairs_records_a_line_whose_crop_cannot_be_read_and_pairs_the_rest(
    tmp_path: Path, make_set: MakeSet, jev_font: Path
) -> None:
    lines = make_set(
        "lines", [("p_0_0", "Anno 1715", "GT-A", "V1", "train"), ("p_0_1", "den 4 maj", "GT-A", "V1", "train")]
    )
    lost = (tmp_path / "sets" / "lines_lines" / "p_0_1.jpg").resolve()
    lost.unlink()
    out = tmp_path / "jev.jsonl"

    built = build_pairs(lines, out, settings=PairSettings(font=jev_font), reads=[])

    assert built.skipped == 1
    assert {row["line"] for row in _rows(out.with_name("jev.meta.jsonl"))} == {"p_0_0"}
    assert _rows(out.with_name("jev.skipped.jsonl")) == [
        {"id": "p_0_1", "error": f"FileNotFoundError: [Errno 2] No such file or directory: '{lost}'"}
    ]


def test_build_pairs_refuses_a_read_of_a_line_the_set_does_not_hold(
    tmp_path: Path, make_set: MakeSet, jev_font: Path
) -> None:
    lines = make_set("lines", [("p_0_0", "Anno 1715", "GT-A", "V1", "train")])
    reads = tmp_path / "reads.jsonl"
    reads.write_text(json.dumps({"id": "q_0_0", "reader": "trocr", "text": "Anno 1715"}) + "\n", encoding="utf-8")

    with pytest.raises(SetRefused, match="reads q_0_0, a line the set does not hold"):
        build_pairs(lines, tmp_path / "jev.jsonl", settings=PairSettings(font=jev_font), reads=[reads])


def test_jev_set_keeps_the_crops_as_cut_at_height_0(tmp_path: Path, make_set: MakeSet, jev_font: Path) -> None:
    lines = make_set("lines", [("p_0_0", "Anno 1715", "GT-A", "V1", "train")], size=(400, 80))
    out = tmp_path / "pairs" / "jev.jsonl"

    result = CliRunner().invoke(
        app,
        ["jev-set", "--lines", str(lines), "--out", str(out), "--height", "0"],
        env={"HTR_TRAIN_JEV_FONT": str(jev_font)},
    )

    assert result.exit_code == 0, result.output
    assert _size(out.with_name("jev_crops") / "p_0_0.jpg") == (400, 80)


def test_jev_set_refuses_to_build_pairs_without_the_renderers_font(
    tmp_path: Path, make_set: MakeSet, monkeypatch: pytest.MonkeyPatch
) -> None:
    lines = make_set("lines", [("p_0_0", "Anno 1715", "GT-A", "V1", "train")])
    monkeypatch.chdir(tmp_path)  # short paths: rich wraps its output at 80 columns

    result = CliRunner().invoke(
        app, ["jev-set", "--lines", str(lines), "--out", "jev.jsonl"], env={"HTR_TRAIN_JEV_FONT": "missing.ttf"}
    )

    assert result.exit_code == 1
    assert "the renderer's font missing.ttf is missing: set HTR_TRAIN_JEV_FONT" in " ".join(result.output.split())
    assert not Path("jev.jsonl").exists()


def test_render_text_draws_black_ink_on_white_within_the_crops_size(jev_font: Path) -> None:
    image = render_text("Anno 1715", jev_font, (300, 48))

    inked = image.convert("L").point(lambda level: 255 if level < 128 else 0).getbbox()
    assert image.size == (300, 48)
    assert inked is not None and inked[0] >= 4 and inked[2] <= 300 and inked[3] - inked[1] > 20


class _Target(BaseHTTPRequestHandler):
    """A served target's chat API: it answers each crop with the crop's file name, and fails on ``bad.jpg``."""

    requests: list[dict[str, Any]] = []

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append(body)
        image = body["messages"][0]["content"][0]["image_url"]["url"]
        if image.endswith("bad.jpg"):
            self.send_response(500)
            self.end_headers()
            return
        reply = json.dumps({"choices": [{"message": {"content": Path(image).stem}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, format: str, *args: object) -> None:  # quiet: the test reads no server log
        return


@pytest.fixture
def served_target() -> Iterator[str]:
    """A local server standing in for vLLM serving the target: its URL."""
    _Target.requests = []
    server = HTTPServer(("127.0.0.1", 0), _Target)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def test_write_responses_answers_each_prompt_with_the_served_target_and_records_it(
    tmp_path: Path, make_set: MakeSet, served_target: str
) -> None:
    prompts = make_set("lines", [("good", "Anno", "GT-A", "V1", "train"), ("bad", "1715", "GT-A", "V1", "train")])
    target = tmp_path / "target"
    target.mkdir()
    out = tmp_path / "draft" / "responses.jsonl"

    settings = ResponsesSettings(target=target, endpoint=served_target, max_tokens=64, max_failed=0.5)

    made = write_responses(prompts, out, settings)

    crop = (tmp_path / "sets" / "lines_lines" / "good.jpg").resolve()
    assert made.model_dump() == {"answered": 1, "failed": 1}
    assert _rows(out) == [
        {
            "conversations": [
                {
                    "role": "user",
                    "content": [{"type": "image", "path": str(crop)}, {"type": "text", "text": "Text Recognition:"}],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "good"}]},
            ]
        }
    ]
    assert _Target.requests[0] == {
        "model": str(target.resolve()),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": crop.as_uri()}},
                    {"type": "text", "text": "Text Recognition:"},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert json.loads(out.with_name("responses.target.json").read_text(encoding="utf-8")) == {
        "target": str(target.resolve()),
        "samples": 1,
        "failed": 1,
        "max_failed": 0.5,
        "max_tokens": 64,
    }
    assert [row["number"] for row in _rows(out.with_name("responses.skipped.jsonl"))] == [2]


def test_write_responses_asks_a_seeded_draw_at_once_and_writes_it_in_the_sets_order(
    tmp_path: Path, make_set: MakeSet, served_target: str
) -> None:
    names = [f"line{k}" for k in range(8)]
    prompts = make_set("lines", [(name, "Anno", "GT-A", "V1", "train") for name in names])
    target = tmp_path / "target"
    target.mkdir()
    out = tmp_path / "responses.jsonl"
    settings = ResponsesSettings(target=target, endpoint=served_target, limit=3, seed=1, concurrency=3)

    made = write_responses(prompts, out, settings)

    answers = [row["conversations"][1]["content"][0]["text"] for row in _rows(out)]
    assert made.answered == 3
    assert answers == sorted(answers, key=names.index)
    assert answers != names[:3]  # a draw from the whole set, not its first lines
    assert answers == [names[k] for k in sorted(random.Random(1).sample(range(8), 3))]


def test_write_responses_refuses_a_set_whose_prompts_fail_past_its_share(
    tmp_path: Path, make_set: MakeSet, served_target: str
) -> None:
    prompts = make_set("lines", [("good", "Anno", "GT-A", "V1", "train"), ("bad", "1715", "GT-A", "V1", "train")])
    target = tmp_path / "target"
    target.mkdir()
    settings = ResponsesSettings(target=target, endpoint=served_target, max_failed=0.01)

    with pytest.raises(SetRefused, match="1 of 2 prompts failed"):
        write_responses(prompts, tmp_path / "responses.jsonl", settings)


def test_write_responses_refuses_prompts_of_a_gt_test_page_and_asks_nothing(tmp_path: Path, make_set: MakeSet) -> None:
    prompts = make_set("lines", [("seen", "Anno", "GT-A", "V1", "train"), ("held", "1715", "GT-A", "V2", "test")])
    target = tmp_path / "target"
    target.mkdir()
    settings = ResponsesSettings(target=target, endpoint="http://127.0.0.1:9")  # nothing listens: asking would fail

    with pytest.raises(SetRefused, match="holds held, a real sample not of the GT's training side"):
        write_responses(prompts, tmp_path / "responses.jsonl", settings)

    assert not (tmp_path / "responses.jsonl").exists()
