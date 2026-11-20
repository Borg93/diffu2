"""The language of every pool scan's text as columns of scans.lance: `language`, `language_p` (CURATION E64).

    uv run --with lingua-language-detector python runs/htr/curate/prototypes/language_column.py detect OUT_DIR [WORKERS]
    flock runs/htr/curate/scans.lock uv run python runs/htr/curate/prototypes/language_column.py join OUT_DIR

The text is the old label's (every scan has one; the reader has read only a part of the pools), its first 2,000
characters, named by lingua among the languages the archives hold. A scan with under 100 characters of text gets no
language. Parquet parts of 50,000 scans in OUT_DIR, resumable; `join` adds the columns (run alone on the table).
"""
import os

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0"); os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
import random, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lance, pyarrow as pa, pyarrow.parquet as pq

HOME = Path("runs/htr/curate"); PART = 50_000; MIN_CHARS = 100
ROWS = pa.schema([("image", pa.utf8()), ("language", pa.utf8()), ("language_p", pa.float32())])
DETECTOR = None


def detector():
    global DETECTOR
    if DETECTOR is None:
        from lingua import Language, LanguageDetectorBuilder
        DETECTOR = LanguageDetectorBuilder.from_languages(Language.SWEDISH, Language.GERMAN, Language.LATIN, Language.DUTCH, Language.FRENCH,
                                                          Language.FINNISH, Language.DANISH, Language.ENGLISH).build()
    return DETECTOR


def named(text: str) -> tuple[str | None, float | None]:
    """The language of ``text`` and lingua's confidence in it; (None, None) for too little text."""
    if len(text) < MIN_CHARS:
        return None, None
    best = detector().compute_language_confidence_values(text[:2000])[0]
    return best.language.name.lower(), float(best.value)


def one(pair: tuple[str, str]) -> tuple[str, str | None, float | None]:
    from diffu_curate.alto import read_alto
    image, labels = pair
    try:
        text = " ".join(line.text for line in read_alto(Path(labels)).lines)
    except (OSError, ValueError, SyntaxError):
        return image, None, None
    return (image, *named(text))


if __name__ == "__main__":
    what, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    if what == "detect":
        from diffu_curate.settings import CurateSettings
        settings = CurateSettings(); workers = int(sys.argv[3]) if len(sys.argv) > 3 else 24
        rows = lance.dataset(str(HOME / "scans.lance")).to_table(columns=["image", "pool", "scan"], filter="pool != 'gt' AND lines > 0").to_pylist()
        random.Random(0).shuffle(rows)
        done = {image for part in out.glob("part-*.parquet") for image in pq.read_table(part, columns=["image"]).column("image").to_pylist()}
        todo = [(row["image"], str(settings.pool(row["pool"]).label_file(row["scan"]))) for row in rows if row["image"] not in done]
        print(len(rows), "scans;", len(done), "done before;", len(todo), "to do", flush=True); start = time.time()
        with ProcessPoolExecutor(workers) as pool:
            for at in range(0, len(todo), PART):
                got = list(pool.map(one, todo[at:at + PART], chunksize=200))
                part = out / f"part-{time.time_ns()}.parquet"
                pq.write_table(pa.table({"image": [g[0] for g in got], "language": [g[1] for g in got], "language_p": [g[2] for g in got]}, schema=ROWS), part.with_suffix(".unfinished"))
                part.with_suffix(".unfinished").replace(part)
                print(f"{at + len(got)} of {len(todo)}, {(at + len(got)) / (time.time() - start):.0f} scans a second", flush=True)
        print("done", flush=True)
    else:
        from diffu_curate.table import add_features
        features = pa.concat_tables([pq.read_table(part) for part in sorted(out.glob("part-*.parquet"))])
        print(f"{add_features(HOME / 'scans.lance', features):,} of {len(features):,} matched in scans.lance")
        held = lance.dataset(str(HOME / "scans.lance")).to_table(columns=["language"], filter="pool != 'gt' AND language IS NOT NULL").to_pandas()
        print(len(held), "pool scans with a language:", (held.language.value_counts() / len(held)).round(4).to_dict())
