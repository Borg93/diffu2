"""Run a real HTR pipeline end-to-end on a generated page — proof the synthetic data is usable.

This wraps **HTRFlow** (AI-Riksarkivet) with the production Riksarkivet pipeline and runs it
**directly — no Ray, no Serve, no FastAPI** (just ``Pipeline.from_config(...).run(Document(img))``):

    YOLO region seg  ->  YOLO line-in-region seg  ->  TrOCR (Swedish hist)  ->  reading order

Because we *know* the text we composited onto the page, running this yields the real end-to-end
number: **can a production Swedish HTR model segment and read our generated pages?** — the only
honest "is the synthetic data good enough" test (a clean page that a real recogniser can't read
is useless as HTR training data; one it *can* read is the goal).

Heavy deps (htrflow, torch, ultralytics, transformers) are imported lazily and live in a dedicated
env so the training venv is untouched::

    uv venv /tmp/htr_env --python 3.12
    uv pip install --python /tmp/htr_env/bin/python \\
        opencv-python-headless 'git+https://github.com/AI-Riksarkivet/htrflow.git'
    PYTHONPATH=packages/diffu-page/src /tmp/htr_env/bin/python -m diffu_page.htr \\
        page_samples/page_00.png --gt page_samples/page_00.page.xml
"""

from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import typer
from pydantic import BaseModel

from .validate import char_error_rate

# Recogniser = the SAME one diffu uses for gen_CER: Config().aux.htr_recognizer_eval, i.e.
# `trocr-base-handwritten-hist-swe-3` (see src/diffu/config.py, app.py, eval_cer.py). The repo is
# GATED to the Riksarkivet org (404 over the network here) but is ALREADY CACHED locally — diffu
# loads it with local_files_only=True — so run htrflow with HF_HUB_OFFLINE=1 to read it from
# ~/.cache/huggingface. Override via --recognizer.
DEFAULT_RECOGNIZER = "Riksarkivet/trocr-base-handwritten-hist-swe-3"

# The production Riksarkivet pipeline, verbatim from the Ray deployment minus the serving layer.
# The TrOCR generation_settings carry hard-won fixes (do not drop them):
#   interpolate_pos_encoding  base-3 emits 192x1024 crops but its ViT grid is 384x384; without
#                             this generate() raises ValueError and htrflow's worker thread dies.
#   use_cache                 shipped generation_config has use_cache=false -> O(n^2) decode.
#   eos_token_id: 2           base-3 is subword (vocab 50265); eos=2 terminates cleanly.
PIPELINE_YAML = """
steps:
  - step: Segmentation
    settings:
      model: yolo
      model_settings:
        model: Riksarkivet/yolov9-regions-1
  - step: Segmentation
    settings:
      model: yolo
      model_settings:
        model: Riksarkivet/yolov9-lines-within-regions-1
  - step: TextRecognition
    settings:
      model: trocr
      model_settings:
        model: __RECOGNIZER__
      generation_settings:
        batch_size: 8
        interpolate_pos_encoding: True
        max_new_tokens: 128
        use_cache: True
        eos_token_id: 2
  - step: OrderLines
"""


class HtrResult(BaseModel):
    """End-to-end HTRFlow output for one page."""

    image: str
    lines: list[str]  # recognised line transcriptions, in reading order
    alto: str  # the ALTO XML htrflow produced

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


class HtrFlowPipeline:
    """The Riksarkivet HTRFlow pipeline, built once and reused. ``htrflow`` imported lazily."""

    def __init__(self, recognizer: str = DEFAULT_RECOGNIZER, pipeline_yaml: str | None = None) -> None:
        self.pipeline_yaml = (pipeline_yaml or PIPELINE_YAML).replace("__RECOGNIZER__", recognizer)
        self._pipeline: Any = None  # htrflow Pipeline (built lazily in the dedicated env)
        self._serializer: Any = None

    def _ensure(self) -> None:
        if self._pipeline is not None:
            return
        from htrflow.pipeline.pipeline import Pipeline
        from htrflow.serialization.serialization import get_serializer

        # A unique temp file per instance — a fixed path would let concurrent runs clobber each other.
        fd, cfg_path = tempfile.mkstemp(prefix="diffu_htrflow_pipeline_", suffix=".yaml")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(self.pipeline_yaml)
            self._pipeline = Pipeline.from_config(cfg_path)
        finally:
            os.unlink(cfg_path)
        self._serializer = get_serializer("alto")

    def run(self, image_path: str) -> HtrResult:
        """Segment + recognise one page image end-to-end. Returns lines (reading order) + ALTO."""
        from htrflow.document import Document

        self._ensure()
        if self._pipeline is None or self._serializer is None:
            raise RuntimeError("HtrFlowPipeline not initialised")  # _ensure guarantees these
        doc = self._pipeline.run(Document(image_path))  # type: ignore[attr-defined]
        alto = _as_alto_str(self._serializer.serialize(doc))  # type: ignore[attr-defined]
        return HtrResult(image=image_path, lines=_alto_line_texts(alto), alto=alto)


def _as_alto_str(serialized: object) -> str:
    """htrflow's serialize() returns either an ALTO string or a list of ``(name, content)``."""
    if isinstance(serialized, str):
        return serialized
    if isinstance(serialized, (list, tuple)) and serialized:
        first = serialized[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            return str(first[1])
        return str(first)
    return str(serialized)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _alto_line_texts(alto_xml: str) -> list[str]:
    """Pull one transcription per ALTO ``TextLine`` (joining its ``String`` tokens), reading order."""
    root = ET.fromstring(alto_xml)
    lines: list[str] = []
    for tl in root.iter():
        if _local(tl.tag) != "TextLine":
            continue
        words = [s.get("CONTENT", "") for s in tl.iter() if _local(s.tag) == "String" and s.get("CONTENT")]
        if words:
            lines.append(" ".join(words))
    return lines


def ground_truth_lines(xml_path: str) -> list[str]:
    """Read the GT line texts from our exported PAGE-XML or ALTO label file."""
    text = Path(xml_path).read_text(encoding="utf-8")
    root = ET.fromstring(text)
    if "alto" in root.tag.lower():
        return _alto_line_texts(text)
    out: list[str] = []  # PAGE-XML: the LINE-level TextEquiv -> Unicode. It's a DIRECT child of TextLine;
    for tl in root.iter():  # iterating all descendants would grab a <Word>'s nested Unicode (the first word).
        if _local(tl.tag) != "TextLine":
            continue
        equiv = next((c for c in tl if _local(c.tag) == "TextEquiv"), None)
        uni = (
            next((u.text for u in equiv.iter() if _local(u.tag) == "Unicode" and u.text), None)
            if equiv is not None
            else None
        )
        if uni:
            out.append(uni.strip())
    return out


def _norm(s: str) -> str:
    return " ".join(s.split())


def _gt_for(image: Path) -> str | None:
    """Auto-match a page image to its ground-truth label file.

    Tries the CURRENT export names first — ``EXTENSIONS['page']`` is plain ``.xml`` (what the CLI and
    the runner's page pipeline write, and the GT-Lejonet ``page/`` layout uses), also checked in a
    sibling ``page/`` dir — then the legacy ``.page.xml`` suffix from older sample sets."""
    candidates = [
        image.parent / (image.stem + ".xml"),
        image.parent.parent / "page" / (image.stem + ".xml"),  # img/ + page/ corpus layout
        image.parent / (image.stem + ".page.xml"),
        image.parent / (image.stem + ".alto.xml"),
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand)
    return None


def _line_with_box(tl: ET.Element) -> tuple[str, tuple[int, int, int, int]] | None:
    """One ``TextLine``'s GT ``(text, bbox)`` — PAGE ``TextEquiv``+``Coords`` first, else ALTO
    ``String``+``HPOS``; ``None`` when the element lacks text or geometry."""
    equiv = next((c for c in tl if _local(c.tag) == "TextEquiv"), None)  # PAGE: line-level TextEquiv
    uni = (
        next((u.text for u in equiv.iter() if _local(u.tag) == "Unicode" and u.text), None)
        if equiv is not None
        else None
    )
    coords = next((c for c in tl if _local(c.tag) == "Coords"), None)
    if uni and coords is not None and coords.get("points"):
        pts = [p.split(",") for p in coords.get("points", "").split()]
        xs, ys = [int(p[0]) for p in pts], [int(p[1]) for p in pts]
        return uni.strip(), (min(xs), min(ys), max(xs), max(ys))
    if tl.get("HPOS") is None:  # ALTO: String CONTENT + HPOS/VPOS/WIDTH/HEIGHT
        return None
    content = " ".join(s.get("CONTENT", "") for s in tl.iter() if _local(s.tag) == "String").strip()
    if not content:
        return None
    x, y, w, h = (int(tl.get(k, 0)) for k in ("HPOS", "VPOS", "WIDTH", "HEIGHT"))
    return content, (x, y, x + w, y + h)


def ground_truth_lines_with_boxes(xml_path: str) -> list[tuple[str, tuple[int, int, int, int]]]:
    """GT ``(line text, bbox)`` pairs from our PAGE-XML/ALTO — the input for segmentation-free per-line
    recognition (we take the boxes as given instead of letting HTRFlow re-detect them)."""
    root = ET.fromstring(Path(xml_path).read_text(encoding="utf-8"))
    out: list[tuple[str, tuple[int, int, int, int]]] = []
    for tl in root.iter():
        if _local(tl.tag) != "TextLine":
            continue
        line = _line_with_box(tl)
        if line is not None:
            out.append(line)
    return out


class LineRecognizer:
    """The TrOCR recogniser alone — no YOLO. Reads a PRE-cropped line, so the score reflects legibility, not
    how HTRFlow re-segments the page (the confound that made the end-to-end page-CER meaningless). Same
    recogniser diffu's ``gen_CER`` uses, so the numbers are comparable. Heavy deps imported lazily."""

    def __init__(self, model_id: str = DEFAULT_RECOGNIZER) -> None:
        self.model_id = model_id
        self._proc: Any = None
        self._model: Any = None
        self._device = "cpu"

    def _ensure(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self._proc = TrOCRProcessor.from_pretrained(self.model_id)
        self._model = VisionEncoderDecoderModel.from_pretrained(self.model_id)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device).eval()

    def read(self, crops: list[Any], batch_size: int = 8) -> list[str]:
        """Recognise BGR line crops -> transcriptions (same order)."""
        import cv2
        import torch
        from PIL import Image

        self._ensure()
        preds: list[str] = []
        for i in range(0, len(crops), batch_size):
            imgs = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops[i : i + batch_size]]
            px = self._proc(images=imgs, return_tensors="pt").pixel_values.to(self._device)
            with torch.no_grad():
                ids = self._model.generate(
                    px, max_new_tokens=128, interpolate_pos_encoding=True, use_cache=True, eos_token_id=2
                )
            preds.extend(self._proc.batch_decode(ids, skip_special_tokens=True))
        return preds


def line_crop_cer(
    page_image: str, gt_xml: str, recognizer: LineRecognizer, *, pad: int = 4
) -> tuple[float, list[tuple[str, str, float]]]:
    """Segmentation-free page readability: crop each GT line from the page, recognise it directly, average
    the per-line CER. Returns ``(mean_cer, [(gt, pred, cer), ...])`` — the trustworthy legibility gate."""
    import cv2

    page = cv2.imread(page_image)
    h, w = page.shape[:2]
    crops: list[Any] = []
    texts: list[str] = []
    for gt_text, (x0, y0, x1, y1) in ground_truth_lines_with_boxes(gt_xml):
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
        if x1 > x0 and y1 > y0:
            crops.append(page[y0:y1, x0:x1])
            texts.append(gt_text)
    preds = recognizer.read(crops) if crops else []
    rows = [(t, p, char_error_rate(_norm(p), _norm(t))) for t, p in zip(texts, preds, strict=True)]
    mean = sum(r[2] for r in rows) / len(rows) if rows else 0.0
    return mean, rows


def evaluate(pipe: HtrFlowPipeline, image: str, gt: str | None) -> tuple[HtrResult, float | None]:
    """Run one page; return its result and page-level CER vs ground truth (None if no GT)."""
    result = pipe.run(image)
    cer = None
    if gt:
        cer = char_error_rate(_norm(result.text), _norm("\n".join(ground_truth_lines(gt))))
    return result, cer


def _run_line_cer(image: Path, gt: str | None, recognizer: str) -> None:
    """Segmentation-free per-GT-line CER (the trustworthy legibility number)."""
    gt_path = gt or _gt_for(image)
    if not gt_path:
        raise SystemExit("--line-cer needs --gt (the page's PAGE/ALTO label file)")
    mean, rows = line_crop_cer(str(image), gt_path, LineRecognizer(recognizer))
    print(f"\n=== per-GT-line recognition on {image.name} ({len(rows)} lines, no segmentation) ===")
    for gt_text, pred, cer in rows:
        print(f"  cer {cer:.2f} | GT {gt_text[:42]!r} -> READ {pred[:42]!r}")
    print(f"\nLINE-CER (mean, segmentation-free) = {mean:.3f}   (0=perfect; real scans ~0.02)")


def _run_batch(pipe: HtrFlowPipeline, image: Path, gt: str | None) -> None:
    """Batch mode: every page in the folder, systematic CER table."""
    pages = sorted(p for p in image.glob("*.png") if ".htrflow" not in p.name)
    cers: list[float] = []
    print(f"=== HTRFlow over {len(pages)} pages in {image} ===", flush=True)
    for p in pages:
        _, cer = evaluate(pipe, str(p), gt or _gt_for(p))
        tag = f"{cer:.3f}" if cer is not None else "  -  (no GT)"
        if cer is not None:
            cers.append(cer)
        print(f"  {p.name:28s}  CER={tag}", flush=True)
    if cers:
        s = sorted(cers)
        mean = sum(cers) / len(cers)
        print(
            f"\n=== {len(cers)} scored | mean {mean:.3f} | median {s[len(s) // 2]:.3f} "
            f"| min {s[0]:.3f} | max {s[-1]:.3f}  (real scans ~0.016) ==="
        )


def _run_single(pipe: HtrFlowPipeline, image: Path, gt: str | None, out: Path | None) -> None:
    """Single page: full transcription dump (+ optional ALTO write + page-level CER)."""
    result, cer = evaluate(pipe, str(image), gt or _gt_for(image))
    print(f"\n=== HTRFlow read {len(result.lines)} lines from {image} ===", flush=True)
    for i, ln in enumerate(result.lines):
        print(f"{i:3d} | {ln}")
    if out:
        out.write_text(result.alto, encoding="utf-8")
        print(f"\nALTO -> {out}")
    if cer is not None:
        print(f"\nPAGE-LEVEL CER = {cer:.3f}   (0=perfect, ~1=unreadable, real scans ~0.016)")


def run_pages_cli(
    image: Path,
    gt: str | None = None,
    out: Path | None = None,
    recognizer: str = DEFAULT_RECOGNIZER,
    line_cer: bool = False,
) -> None:
    """Run HTRFlow on a page image or a folder of pages — the logic behind the ``run`` command.

    Args:
        image: A page image, or a directory of pages (batch mode).
        gt: GT xml for a single image; in batch mode it's auto-matched per ``<stem>.page.xml``.
        out: Write htrflow's ALTO here (single image only).
        recognizer: TrOCR repo id (override the gated ``-swe-3``).
        line_cer: Recognise each GT-line crop directly instead (no YOLO re-segmentation).
    """
    if line_cer:
        _run_line_cer(image, gt, recognizer)
        return
    pipe = HtrFlowPipeline(recognizer=recognizer)  # built once, reused across pages
    if image.is_dir():
        _run_batch(pipe, image, gt)
        return
    _run_single(pipe, image, gt, out)


def run(
    image: Path = typer.Argument(..., help="a page image, OR a directory of pages (batch mode)"),
    gt: str | None = typer.Option(
        None, "--gt", help="GT xml (single image); in batch mode it's auto-matched per <stem>.page.xml"
    ),
    out: Path | None = typer.Option(None, "--out", help="write htrflow's ALTO here (single image only)"),
    recognizer: str = typer.Option(
        DEFAULT_RECOGNIZER, "--recognizer", help="TrOCR repo id (override the gated -swe-3)"
    ),
    line_cer: bool = typer.Option(
        False,
        "--line-cer",
        help="trustworthy gate: recognise each GT-line crop directly (no YOLO re-segmentation)",
    ),
) -> None:
    """Run HTRFlow end-to-end on a page — or a folder of pages."""
    run_pages_cli(image, gt=gt, out=out, recognizer=recognizer, line_cer=line_cer)


def main() -> None:
    typer.run(run)


if __name__ == "__main__":
    main()
