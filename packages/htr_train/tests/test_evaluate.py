"""Scoring the readers: a line set's test side as lines to score; MiniCPM-OCR's reading through its served model (each
line asked as OpenBMB evaluates it, the predictions in the lines' order, why each ended and the speed recorded), and
its readings as reads for Jev's pairs (a line cut off left out); and error rates as each side scores itself (the
Swedish Lion's per-line mean CER, whitespace normalised; OpenBMB's corpus CER and WER, NFC), with the baseline's
training normalisation on both sides, and the lines cut off counted."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

from diffu_htr_train.evaluate.cer import by_subset, cut_loop, trocr_normalize, whitespace
from diffu_htr_train.evaluate.cli import app
from diffu_htr_train.evaluate.in_training import Reading, answer, metrics, write_readings
from diffu_htr_train.evaluate.lines import EvalLine, read_lines

if TYPE_CHECKING:
    from conftest import MakeSet


class _Served(BaseHTTPRequestHandler):
    """A served MiniCPM-OCR's chat API: it reads each crop as the crop's file name."""

    requests: list[dict[str, Any]] = []

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append(body)
        image = body["messages"][0]["content"][0]["image_url"]["url"]
        finished = "length" if Path(image).stem.startswith("long") else "stop"
        reply = json.dumps(
            {"choices": [{"message": {"content": Path(image).stem}, "finish_reason": finished}]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, format: str, *args: object) -> None:  # quiet: the test reads no server log
        return


@pytest.fixture
def served_minicpm() -> Iterator[str]:
    """A local server standing in for vLLM serving MiniCPM-OCR: its URL."""
    _Served.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Served)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def test_minicpm_reads_a_sets_test_lines_as_openbmb_asks_and_records_its_speed(
    tmp_path: Path, make_set: MakeSet, served_minicpm: str
) -> None:
    test = make_set(
        "lines.test",
        [
            ("a_0_0", "Anno 1715", "GT-A", "V1", "test"),
            ("b_0_0", "den 4 maj", "GT-B", "V2", "test"),
            ("long_0_0", "och sedan", "GT-B", "V2", "test"),
        ],
    )
    lines, out, reads = tmp_path / "lines.jsonl", tmp_path / "read" / "minicpm.jsonl", tmp_path / "reads.jsonl"
    runner = CliRunner()

    made = runner.invoke(app, ["set-lines", str(test), "--out", str(lines)])
    read = runner.invoke(
        app,
        [
            *("read", "--lines", str(lines), "--reader", "minicpm", "--model", "base"),
            *("--endpoint", served_minicpm, "--out", str(out)),
        ],
    )

    as_reads = runner.invoke(app, ["as-reads", str(out), "--reader", "minicpm-all", "--out", str(reads)])

    assert (made.exit_code, read.exit_code, as_reads.exit_code) == (0, 0, 0), made.output + read.output
    assert [(line.subset, line.text) for line in read_lines(lines)][:2] == [
        ("GT-A", "Anno 1715"),
        ("GT-B", "den 4 maj"),
    ]
    rows = [json.loads(row) for row in out.read_text(encoding="utf-8").splitlines()]
    assert [(row["id"], row["text"], row["prediction"], row["finished"]) for row in rows] == [
        ("a_0_0", "Anno 1715", "a_0_0", "stop"),
        ("b_0_0", "den 4 maj", "b_0_0", "stop"),
        ("long_0_0", "och sedan", "long_0_0", "length"),
    ]
    assert [json.loads(row) for row in reads.read_text(encoding="utf-8").splitlines()] == [
        {"id": "a_0_0", "reader": "minicpm-all", "text": "a_0_0"},
        {"id": "b_0_0", "reader": "minicpm-all", "text": "b_0_0"},
    ]
    asked = _Served.requests[0]
    assert (asked["model"], asked["temperature"], asked["top_p"], asked["top_k"], asked["max_tokens"]) == (
        "base",
        0.01,
        0.001,
        1,
        512,
    )
    assert asked["messages"][0]["content"][1] == {"type": "text", "text": "Text Recognition:"}
    assert (asked["chat_template_kwargs"], asked["mm_processor_kwargs"]) == (
        {"enable_thinking": False},
        {"downsample_mode": "4x", "max_slice_nums": 9},
    )
    speed = json.loads(out.with_name("minicpm.speed.json").read_text(encoding="utf-8"))
    assert (speed["reader"], speed["lines"], speed["settings"]["concurrency"]) == ("minicpm", 3, 32)
    assert speed["lines_per_second"] > 0


def test_cer_is_the_per_line_mean_the_baseline_reports_beside_the_corpus_rate() -> None:
    rows = [
        ("V1", "Anno 1715 den 4", "Anno 1715 den 4"),  # 0 edits of 15
        ("V1", "abcd", "abcf"),  # 1 of 4
        ("V2", "34.", ""),  # 3 of 3: a short line weighs as much as a long one in the mean
        ("V2", "  den  4  maj ", "den 4 maj"),  # whitespace only: 0
    ]

    scored = by_subset(rows, whitespace)

    assert scored["all"].lines == 4
    assert scored["all"].mean_cer == pytest.approx((0 + 0.25 + 1 + 0) / 4)
    assert scored["all"].corpus_cer == pytest.approx(4 / (15 + 4 + 3 + 9))
    assert scored["V2"].mean_cer == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("text", "normalised"),
    [
        pytest.param("2½ skeppund", "2 1/2 skeppund", id="fraction"),
        pytest.param("„Hwad ”sade” han´", 'Hwad "sade" han\'', id="quotes"),
        pytest.param("Den .... 13 .. Januarii", "Den ... 13 .. Januarii", id="dots"),
        pytest.param("Carl _ _ _ Wikman. 16.", "Carl — Wikman. 16.", id="dashes"),
        pytest.param("Com ¬¬", "Com¬", id="hyphen"),
    ],
)
def test_the_baselines_normalisation_reads_as_its_training_writes_it(text: str, normalised: str) -> None:
    assert trocr_normalize(text) == normalised


def test_the_cer_command_scores_each_predictions_file_as_each_side_scores_itself(tmp_path: Path) -> None:
    predictions = tmp_path / "read.jsonl"
    rows = [
        {"id": "a", "subset": "V1", "text": "„Anno 1715", "prediction": "Anno 1715", "finished": "stop"},
        {"id": "b", "subset": "V2", "text": "abcd", "prediction": "abcf", "finished": "length"},
    ]
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    result = CliRunner().invoke(app, ["cer", str(predictions), "--subsets"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    all_row = next(line for line in result.output.splitlines() if "│ all" in line)
    # mean CER (1/10 + 1/4) / 2; OpenBMB's CER 2 edits of 14 characters, WER 2 of 3 words; normalised „ drops: 1/4 / 2
    assert [cell.strip() for cell in all_row.split("│")[2:-1]] == [
        "2", "17.50%", "14.29%", "66.67%", "12.50%", "1", "17.50%", "14.29%", "0",
    ]  # fmt: skip


@pytest.mark.parametrize(
    ("reading", "cut_off", "kept"),
    [  # the loops the in-training readings of steps 14,000-21,000 held, and what a loop is not
        pytest.param("Hl.r Chronander, damn. damn. damn.", False, "Hl.r Chronander, damn.", id="words"),
        pytest.param("lag at förfaras.\nP.\nP.\nP.\nP.\nP.", True, "lag at förfaras.\nP.", id="lines"),
        pytest.param("Summa" + " ." * 40, True, "Summa .", id="dots-to-the-limit"),
        pytest.param("Herr ...........", False, "Herr ...........", id="dots"),
        pytest.param("utsatt af 450,000", False, "utsatt af 450,000", id="zeros"),
        pytest.param("Item 2 tunnor 2 tunnor", False, "Item 2 tunnor 2 tunnor", id="twice"),
    ],
)
def test_a_looping_tail_is_cut_to_one_copy_and_nothing_else_is(reading: str, cut_off: bool, kept: str) -> None:
    assert cut_loop(reading, cut_off=cut_off) == kept


def test_the_cer_command_scores_the_readings_with_their_loops_cut_too(tmp_path: Path) -> None:
    predictions = tmp_path / "read.jsonl"
    rows = [
        {"id": "a", "subset": "V1", "text": "damn.", "prediction": "damn. damn. damn. damn.", "finished": "stop"},
        {"id": "b", "subset": "V1", "text": "abcd", "prediction": "abcf", "finished": "stop"},
    ]
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    result = CliRunner().invoke(app, ["cer", str(predictions)], env={"COLUMNS": "250"})

    assert result.exit_code == 0, result.output
    all_row = next(line for line in result.output.splitlines() if "│ all" in line)
    # as read: (18/5 + 1/4) / 2 and 19 edits of 9; loops cut: (0 + 1/4) / 2 and 1 of 9, one reading cut
    assert [cell.strip() for cell in all_row.split("│")[3:5] + all_row.split("│")[8:-1]] == [
        "192.50%", "211.11%", "12.50%", "11.11%", "1",
    ]  # fmt: skip


def test_readings_in_training_score_as_the_cer_command_scores_them_written(tmp_path: Path) -> None:
    lines = [
        EvalLine(id="a", subset="V1", image="a.png", text="„Anno 1715"),
        EvalLine(id="b", subset="V2", image="b.png", text="abcd"),
    ]
    readings = [
        Reading(prediction=answer("<think>\n\n</think>\n\nAnno 1715"), finished="stop"),
        Reading(prediction=answer("abcf"), finished="length"),
    ]
    written = tmp_path / "cer" / "step-10.dev.jsonl"

    logged = metrics("dev", lines, readings)
    write_readings(written, lines, readings)
    result = CliRunner().invoke(app, ["cer", str(written)], env={"COLUMNS": "200"})

    assert readings[0].prediction == "Anno 1715"
    assert logged == pytest.approx(
        {"cer_dev": (1 / 10 + 1 / 4) / 2, "cer_corpus_dev": 2 / 14, "cer_norm_dev": 1 / 8, "cut_dev": 1.0}
    )
    all_row = next(line for line in result.output.splitlines() if "│ all" in line)
    assert [cell.strip() for cell in all_row.split("│")[2:4]] == ["2", f"{logged['cer_dev']:.2%}"]
