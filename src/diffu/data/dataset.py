"""Build training triples (line_image, transcription, writer_id) from PAGE/ALTO XML, plus a
torch Dataset, writer-disjoint split, and a character-coverage report.

Parsing/manifest building is pure stdlib + Pillow (no torch). The torch Dataset imports torch
lazily so the prep tools run on a CPU box without torch installed.
"""

from __future__ import annotations

import glob
import json
import os
import random
import unicodedata
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageDraw

from .parse import LineRecord, parse_file

if TYPE_CHECKING:
    import torch

    from .latent_cache import LatentCache

# DINOv2/v3 image normalization (ImageNet stats) for the style reference crop.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_text(s: str) -> str:
    """NFC so å ä ö are single precomposed codepoints (U+00E5/E4/F6); drop control chars."""
    s = unicodedata.normalize("NFC", s)
    return "".join(c for c in s if c in "\n\t" or unicodedata.category(c)[0] != "C").strip()


def crop_line(
    page_img: Image.Image,
    bbox: tuple[float, float, float, float],
    height: int = 64,
    pad: int = 4,
    polygon: list[tuple[float, float]] | None = None,
) -> Image.Image | None:
    """Crop a line from a page image and resize to a fixed height (aspect preserved).

    When ``polygon`` (the PAGE ``<Coords>`` line polygon) is given, everything outside it is masked
    to white — same as HTRflow's ``polygon_mask`` (``Image.composite(image, white, mask)``) — so ink
    from neighbouring lines that falls inside the bbox is removed. Without it, the plain bbox is used.

    Returns None for a degenerate or inverted bbox (some polygons have x1<x0 / y1<y0, which PIL's
    ``crop`` rejects) — coordinates are sorted and zero-area boxes are skipped.
    """
    x0, y0, x1, y1 = bbox
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    left, upper, right, lower = max(0, int(x0) - pad), max(0, int(y0) - pad), int(x1) + pad, int(y1) + pad
    if right <= left or lower <= upper:
        return None
    crop = page_img.crop((left, upper, right, lower)).convert("RGB")
    if polygon:
        mask = Image.new("L", crop.size, 0)
        ImageDraw.Draw(mask).polygon([(px - left, py - upper) for px, py in polygon], fill=255)
        crop = Image.composite(crop, Image.new("RGB", crop.size, (255, 255, 255)), mask)
    w, h = crop.size
    if h <= 0 or w <= 0:
        return None
    return crop.resize((max(8, int(w * height / h)), height), Image.LANCZOS)


_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")


def _index_images(image_dir: str) -> dict[str, str]:
    """Map every image's filename AND stem under image_dir (recursively) -> its full path.

    Archive exports nest page images in volume folders and the XML's ``imageFilename`` is often a
    short name that doesn't match the actual file, so we resolve by the (globally unique) stem via a
    recursive index rather than a flat ``image_dir/basename`` join.
    """
    index: dict[str, str] = {}
    for root, _dirs, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith(_IMAGE_EXTS):
                path = os.path.join(root, f)
                index.setdefault(f, path)
                index.setdefault(os.path.splitext(f)[0], path)
    return index


def _resolve_image(image_field: str, xml_path: str, index: dict[str, str]) -> str | None:
    """Resolve a page image from the recursive index, trying the XML file's own stem first (the most
    reliable key for archive exports), then the ``imageFilename`` and its stem."""
    keys = [os.path.splitext(os.path.basename(xml_path))[0]]
    if image_field:
        keys += [os.path.basename(image_field), os.path.splitext(os.path.basename(image_field))[0]]
    return next((index[k] for k in keys if k in index), None)


def build_manifest(xml_dir: str, image_dir: str, out_dir: str, height: int = 64) -> int:
    """Parse every ``*.xml`` under xml_dir, crop each line from its page image, and write
    ``out_dir/lines/*.png`` + ``out_dir/manifest.jsonl`` with ``{image, text, writer_id}``.

    Each page image is opened once inside a context manager (its records are grouped first), so no
    file handles are leaked even on large corpora.

    Returns:
        Number of line crops written.
    """
    os.makedirs(os.path.join(out_dir, "lines"), exist_ok=True)
    image_index = _index_images(image_dir)
    n = 0
    with open(os.path.join(out_dir, "manifest.jsonl"), "w", encoding="utf-8") as mf:
        for xml_path in sorted(glob.glob(os.path.join(xml_dir, "**", "*.xml"), recursive=True)):
            volume = os.path.basename(os.path.dirname(xml_path))  # parent folder = volume (coarser split key)
            by_image: defaultdict[str, list[LineRecord]] = defaultdict(list)
            for rec in parse_file(xml_path):
                img_path = _resolve_image(rec.image, xml_path, image_index)
                if img_path is not None:
                    by_image[img_path].append(rec)
            for img_path, recs in by_image.items():
                try:
                    with Image.open(img_path) as page:
                        page.load()
                        for rec in recs:
                            crop = crop_line(page, rec.bbox, height, polygon=rec.polygon)
                            text = normalize_text(rec.text)
                            if crop is None or not text:
                                continue
                            out_png = os.path.join(out_dir, "lines", f"{n:07d}.png")
                            crop.save(out_png)
                            mf.write(
                                json.dumps(
                                    {
                                        "image": out_png,
                                        "text": text,
                                        "writer_id": rec.source_id,
                                        "volume": volume,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            n += 1
                except OSError:
                    continue
    return n


def load_manifest(manifest_path: str) -> list[dict[str, str]]:
    with open(manifest_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def char_coverage(manifest_path: str) -> Counter[str]:
    """Character histogram — check å ä ö Å Ä Ö (and historical glyphs) are well-represented."""
    counter: Counter[str] = Counter()
    for row in load_manifest(manifest_path):
        counter.update(row["text"])
    return counter


def _group_key(row: dict[str, str], group_by: str) -> str:
    """Grouping id for the writer-disjoint split.

    ``writer_id`` = page (finest, leakiest); ``volume`` = the page's document volume; ``collection``
    = the whole archive series (coarsest, most conservative). ``volume`` is qualified with the
    ``collection`` when present so identically-named folders in different collections (e.g.
    ``Uppland_1``) can never merge into one group. Falls back to the ``<vol>_<page>`` page-id prefix
    for manifests without an explicit ``volume`` field (e.g. the goteborgs export).
    """
    if group_by == "volume" and "volume" in row:
        collection = row.get("collection", "")
        return f"{collection}/{row['volume']}" if collection else row["volume"]
    if group_by in row:
        return row[group_by]
    if group_by == "volume":
        return row["writer_id"].rsplit("_", 1)[0]
    return row["writer_id"]


def writer_disjoint_split(
    manifest_path: str,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    group_by: str = "writer_id",
) -> dict[str, list[dict[str, str]]]:
    """Split so no GROUP appears in two splits — the correct protocol for HTR eval.

    ``group_by`` selects the granularity: ``writer_id`` (page, the default) or ``volume`` (coarser —
    a more conservative writer-disjoint estimate when one scribe spans many pages of a volume).
    """
    rows = load_manifest(manifest_path)
    groups = sorted({_group_key(r, group_by) for r in rows})
    random.Random(seed).shuffle(groups)
    n = len(groups)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    test_g = set(groups[:n_test])
    val_g = set(groups[n_test : n_test + n_val])
    split: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    for r in rows:
        g = _group_key(r, group_by)
        bucket = "test" if g in test_g else "val" if g in val_g else "train"
        split[bucket].append(r)
    return split


def write_splits(
    manifest_path: str,
    out_dir: str,
    *,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    group_by: str = "writer_id",
) -> dict[str, dict[str, int]]:
    """Group-disjoint split of a manifest into ``out_dir/{train,val,test}.jsonl``.

    The shared "finalize" step for both preprocessing front-ends (``prepare.py`` for PAGE/ALTO XML,
    ``ingest.py`` for pre-cropped lines). ``group_by`` is the split granularity (``writer_id`` page,
    or ``volume``). Returns per-split ``{"lines": N, "groups": M}`` counts.
    """
    split = writer_disjoint_split(manifest_path, val_frac, test_frac, seed, group_by)
    summary: dict[str, dict[str, int]] = {}
    for name, rows in split.items():
        with open(os.path.join(out_dir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[name] = {"lines": len(rows), "groups": len({_group_key(r, group_by) for r in rows})}
    return summary


class HandwritingLineDataset:
    """torch Dataset over manifest rows.

    Each item is ``{'image': [3,H,W] in [-1,1], 'text': str, 'writer_id': str,
    'style_pixel_values': [3,S,S]}`` where the style reference is another line from the SAME writer
    (ImageNet-normalized for DINO). With ``ref_lines=True`` (``style_mode="canvas"``) the same reference
    line is also returned at line height as ``style_image`` (``[3,H,W_ref]`` in ``[-1,1]``, plus
    ``style_latent`` when a latent cache is attached) for the in-canvas path. torch/torchvision are
    imported lazily so the prep tools above need neither. Pair with the width-bucketing ``collate_lines``.
    """

    def __init__(
        self,
        rows: list[dict[str, str]],
        height: int = 64,
        max_width: int = 1024,
        style_size: int = 224,
        latent_cache: LatentCache | None = None,
        ref_lines: bool = False,
    ) -> None:
        self.rows = rows
        self.ref_lines = ref_lines
        self.height = height
        self.max_width = max_width
        self.style_size = style_size
        # Optional precomputed VAE latents (--cache-latents): index-aligned with `rows` (both come from the
        # same split manifest, in order), so latent_cache[i] is the encoded x0 for rows[i]. See data/latent_cache.py.
        self.latent_cache = latent_cache
        self.by_writer: defaultdict[str, list[int]] = defaultdict(list)
        for idx, row in enumerate(rows):
            self.by_writer[str(row["writer_id"])].append(idx)

    def __len__(self) -> int:
        return len(self.rows)

    def _style_index(self, writer_id: str, target: int) -> int:
        """Index of a *different* line by the same writer, for the style reference.

        Never the target line itself: at inference the style ref is always a foreign line, so
        training on the target's own image leaks its content (the model could copy the answer off
        the style tokens). Falls back to the target only when the writer has a single line.
        """
        peers = [p for p in self.by_writer.get(writer_id, ()) if p != target]
        return random.choice(peers) if peers else target

    def _load_style(self, path: str) -> torch.Tensor:
        from torchvision.transforms import functional as TF

        with Image.open(path) as im:
            img = im.convert("RGB").resize((self.style_size, self.style_size), Image.BILINEAR)
        return TF.normalize(TF.to_tensor(img), _IMAGENET_MEAN, _IMAGENET_STD)

    def _load_line(self, path: str) -> torch.Tensor:
        from torchvision.transforms import functional as TF

        with Image.open(path) as im:
            img = im.convert("RGB")
        if self.max_width and img.width > self.max_width:
            img = img.resize((self.max_width, self.height), Image.LANCZOS)
        return TF.normalize(TF.to_tensor(img), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def __getitem__(self, i: int) -> dict[str, Any]:
        r = self.rows[i]
        image = self._load_line(r["image"])
        si = self._style_index(str(r["writer_id"]), i)
        style = self._load_style(self.rows[si]["image"])
        item: dict[str, Any] = {
            "image": image,
            "text": r["text"],
            "writer_id": r["writer_id"],
            "style_pixel_values": style,
            # The style ref's OWN text (a different line, normally != r["text"]) — so the live eval can
            # show "style wrote B, we asked A": a copy-the-style model renders B and gen_CER-vs-A spikes.
            "style_text": self.rows[si]["text"],
        }
        if self.ref_lines:  # the same reference line at line height, for style_mode="canvas"
            item["style_image"] = image if si == i else self._load_line(self.rows[si]["image"])
        if self.latent_cache is not None:  # precomputed x0 [C, h, w] for rows[i]; collate pads + stacks it
            item["latent"] = self.latent_cache[i]
            if self.ref_lines:
                item["style_latent"] = self.latent_cache[si]
        return item
