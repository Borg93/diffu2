"""A vision-language model describing scans by a fixed form (CURATION task 9, pilot).

Each scan is sent whole, its longer side SIDE px, to the model served by runs/htr/curate/serve_vl.sh, which answers
the form below as JSON, asked for in the prompt as NVIDIA's cookbook does (a schema forced by the server gave broken
JSON and one answer for every opening under speculative decoding), thinking off. Appends a row a scan to OUT,
skipping scans already there.

    uv run python runs/htr/curate/prototypes/vl_describe.py SCANS.jsonl OUT.jsonl [PORT]

SCANS.jsonl holds a row a scan with at least "image" (other keys are carried to the output).
"""
import base64, io, json, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
SIDE = 1536  # the cookbook: 1536x1024 for documents
WORKERS = 64
THINK = bool(os.environ.get("THINK"))  # reasoning before the answer
FORM = {
    "type": "object",
    "properties": {
        "caption": {"type": "string", "description": "One sentence, at most 25 words: what the scan shows."},
        "kind": {"type": "string", "enum": ["handwritten_text", "handwritten_table_or_ledger", "printed_or_typed_text",
                                           "sign_or_title_card", "blank", "cover_or_spine", "drawing_or_map", "other"]},
        "layout": {"type": "string", "enum": ["single_page", "opening_both_pages_written", "opening_one_page_empty",
                                             "opening_both_pages_empty", "several_loose_sheets", "not_a_page"]},
        "written_side": {"type": "string", "enum": ["left", "right", "both", "none", "not_an_opening"]},
        "text_columns": {"type": "integer", "minimum": 0, "maximum": 8, "description": "Columns of running text on the fullest page; 0 if none."},
        "table_across_fold": {"type": "boolean", "description": "One table or ledger continues over both pages of the opening."},
        "writing_at_fold": {"type": "boolean", "description": "Writing reaches within a finger's width of the fold or runs over it."},
        "turned_writing": {"type": "boolean", "description": "Most writing is turned 90 or 180 degrees."},
        "fold_x": {"type": "number", "minimum": 0, "maximum": 1, "description": "Where the fold between the two pages is, as a share of the image's width from the left; 0 if there is no fold."},
    },
    "required": ["caption", "kind", "layout", "written_side", "text_columns", "table_across_fold", "writing_at_fold", "turned_writing", "fold_x"],
    "additionalProperties": False,
}
ASK = ("This is a scan from a Swedish archive (1600-1900s), photographed or microfilmed. It may show one page, an "
       "opening of a bound volume (two facing pages with a fold between them), loose sheets, a sign or card laid in by "
       "the archive, a cover, or nothing. Judge what you see; do not transcribe the text. Return only JSON, no prose, with exactly these keys:\n")
KEYS = "\n".join(f'"{key}": ' + (" | ".join(spec["enum"]) if "enum" in spec else spec["type"]) + (f'  ({spec["description"]})' if "description" in spec else "")
                 for key, spec in FORM["properties"].items())


def describe(row: dict, url: str) -> dict:
    started = time.perf_counter()
    try:
        with Image.open(row["image"]) as scan:
            scan.draft("RGB", (SIDE * 2, SIDE * 2)); scan = scan.convert("RGB"); scan.thumbnail((SIDE, SIDE))
            held = io.BytesIO(); scan.save(held, "JPEG", quality=85)
        body = {"model": "nemotron-3.5-super-vl", "temperature": 0.6, "top_p": 0.95, "max_tokens": 4096 if THINK else 600,
                "chat_template_kwargs": {"enable_thinking": THINK},
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": ASK + KEYS},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(held.getvalue()).decode()}}]}]}
        request = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=600) as answer:
            reply = json.load(answer)
        text = re.sub(r"^```(?:json)?|```$", "", (reply["choices"][0]["message"]["content"] or "").strip(), flags=re.M)
        return {**row, **json.loads(text.strip()), "tokens_in": reply["usage"]["prompt_tokens"],
                "tokens_out": reply["usage"]["completion_tokens"], "seconds": round(time.perf_counter() - started, 2)}
    except (OSError, ValueError, KeyError) as error:
        return {**row, "error": f"{type(error).__name__}: {error}"[:300]}


if __name__ == "__main__":
    scans, out = Path(sys.argv[1]), Path(sys.argv[2]); port = sys.argv[3] if len(sys.argv) > 3 else "8011"
    done = {json.loads(line)["image"] for line in out.open()} if out.exists() else set()
    rows = [row for row in map(json.loads, scans.open()) if row["image"] not in done]
    url = f"http://127.0.0.1:{port}/v1/chat/completions"; started = time.time()
    with ThreadPoolExecutor(WORKERS) as pool, out.open("a", encoding="utf-8") as held:
        for count, row in enumerate(pool.map(lambda r: describe(r, url), rows), 1):
            held.write(json.dumps(row, ensure_ascii=False) + "\n"); held.flush()
            if count % 50 == 0 or count == len(rows):
                print(f"{count} of {len(rows)} scans, {count / (time.time() - started):.2f} a second", flush=True)
