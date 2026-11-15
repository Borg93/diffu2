"""The second pass over the described scans: the two questions the first form asked badly, each option defined
(CURATION E19: with definitions the model tells a table from running text in 33-35 of 35 scans judged by eye, against
about half with the first form's bare option names).

    uv run python runs/htr/curate/prototypes/vl_ask2.py SCANS.jsonl OUT.jsonl [PORT]
"""
import base64, io, json, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
SIDE, WORKERS = 1536, 64
ASK = ("This is a scan from a Swedish archive. Answer two questions about the handwriting on it.\n"
       "1. writing_layout: is it laid out as a table or ledger, or as running text?\n"
       "   - table: ruled columns or rows, or short entries with numbers or amounts lined up in columns down the page.\n"
       "   - running_text: sentences in lines across the page, as in a letter, a court record or minutes, even if dense, with a margin, headings, dates or a few numbers.\n"
       "   - none: no handwriting (a sign, a cover, a blank).\n"
       "2. turned_writing: is any handwriting turned on its side or upside down against the rest?\n"
       "   - none: all writing runs the same way, upright.\n"
       "   - some: a note, an address or a margin entry is turned, the rest is upright.\n"
       "   - most: most of the writing is on its side or upside down as the scan lies.\n"
       'Return only JSON: {"layout": "table" | "running_text" | "none", "turned_writing": "none" | "some" | "most"}')


def ask(row: dict, url: str) -> dict:
    try:
        with Image.open(row["image"]) as scan:
            scan.draft("RGB", (SIDE * 2, SIDE * 2)); scan = scan.convert("RGB"); scan.thumbnail((SIDE, SIDE)); held = io.BytesIO(); scan.save(held, "JPEG", quality=85)
        body = {"model": "nemotron-3.5-super-vl", "temperature": 0.6, "top_p": 0.95, "max_tokens": 120, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [{"type": "text", "text": ASK}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(held.getvalue()).decode()}}]}]}
        with urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600) as answer:
            text = json.load(answer)["choices"][0]["message"]["content"] or ""
        said = json.loads(re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip())
        return {"image": row["image"], "writing_layout": said["layout"], "turned": said["turned_writing"]}
    except (OSError, ValueError, KeyError) as error:
        return {"image": row["image"], "error": f"{type(error).__name__}: {error}"[:200]}


if __name__ == "__main__":
    scans, out = Path(sys.argv[1]), Path(sys.argv[2]); port = sys.argv[3] if len(sys.argv) > 3 else "8011"
    done = {json.loads(line)["image"] for line in out.open()} if out.exists() else set()
    rows = [row for row in map(json.loads, scans.open()) if row["image"] not in done]; started = time.time()
    with ThreadPoolExecutor(WORKERS) as pool, out.open("a", encoding="utf-8") as held:
        for count, row in enumerate(pool.map(lambda r: ask(r, f"http://127.0.0.1:{port}/v1/chat/completions"), rows), 1):
            held.write(json.dumps(row, ensure_ascii=False) + "\n"); held.flush()
            if count % 500 == 0 or count == len(rows): print(f"{count} of {len(rows)} scans, {count / (time.time() - started):.2f} a second", flush=True)
