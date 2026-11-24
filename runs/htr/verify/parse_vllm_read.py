"""The 200 test pages read through a vLLM server of Nemotron 3.5 Parse, many pages in flight (CURATION E50).

    uv run --with httpx python runs/htr/verify/parse_vllm_read.py OUT.jsonl [CONCURRENCY [LIMIT]]

The request is the model card's: the task prompt as the text, one page image, temperature 0, top-k 1, repetition
penalty 1.1, at most 8,744 tokens, special tokens kept. Rows as nemotron_parse_test_read.py writes them (id, subset,
text, prediction, raw, tokens, seconds, finish), so the same scorers read them; the last line printed is the rate.
"""
import base64, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
import httpx

PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"
LAYOUT = re.compile(r"<(?:x|y)_[0-9.]+>|<class_[A-Za-z-]+>")
TABLE = re.compile(r"\\begin\{tabular\}\{[^}]*\}|\\end\{tabular\}|\\linebreak\{\}|\\\\| & ")
URL = os.environ.get("URL", "http://127.0.0.1:8011/v1"); MODEL = os.environ.get("MODEL", "nemotron-parse")
def text(answer: str) -> str: return " ".join(TABLE.sub(" ", LAYOUT.sub("\n", answer)).replace("#", " ").split())

out = sys.argv[1]; concurrency = int(sys.argv[2]) if len(sys.argv) > 2 else 32; limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
wanted = sorted(json.loads(l)["id"] for s in (0, 1) for l in open(f"runs/htr/uocr2/test_read/shard{s}.jsonl", encoding="utf-8"))[:limit]
rows = open("data/htr/nemotron_parse_pages.test.jsonl", encoding="utf-8").read().splitlines()
metas = [json.loads(l) for l in open("data/htr/nemotron_parse_pages.test.meta.jsonl", encoding="utf-8")]
index = {m["id"]: i for i, m in enumerate(metas)}
pages = [(p, json.loads(rows[index[p]]), metas[index[p]]) for p in wanted]
SCALE = float(os.environ.get("SCALE", "1"))  # 24 Nov: the page sent smaller (each side times SCALE), to see what the image stage's cost buys
def sent(path: str) -> bytes:
    if SCALE == 1: return open(path, "rb").read()
    import io; from PIL import Image
    image = Image.open(path).convert("RGB"); image = image.resize((round(image.width * SCALE), round(image.height * SCALE)), Image.Resampling.BILINEAR)
    buffer = io.BytesIO(); image.save(buffer, "JPEG", quality=92); return buffer.getvalue()
images = {p: base64.b64encode(sent(row["images"][0])).decode() for p, row, _ in pages}  # read before the clock
client = httpx.Client(timeout=1800, limits=httpx.Limits(max_connections=max(64, concurrency)))

def read(item):
    page, row, meta = item; started = time.perf_counter()
    body = {"model": MODEL, "temperature": 0, "max_tokens": 8744, "top_k": 1, "repetition_penalty": 1.1, "skip_special_tokens": False,
            "repetition_detection": {"min_pattern_size": 4, "max_pattern_size": 120, "min_count": 6},  # vLLM stops a looping page (E51)
            "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + images[page]}}]}]}
    answer = client.post(f"{URL}/chat/completions", json=body).raise_for_status().json(); choice = answer["choices"][0]; raw = choice["message"]["content"] or ""
    raw = raw.replace("</s>", "").replace("<s>", "").replace("<pad>", "")
    return {"id": page, "subset": meta["collection"] or meta["volume"], "text": text(row["messages"][1]["content"]), "prediction": text(raw), "raw": raw,
            "tokens": answer["usage"]["completion_tokens"], "seconds": round(time.perf_counter() - started, 2), "finish": choice["finish_reason"]}

start = time.perf_counter()
with ThreadPoolExecutor(concurrency) as pool: got = list(pool.map(read, pages))
took = time.perf_counter() - start
with open(out, "w", encoding="utf-8") as f:
    for row in got: f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"{len(got)} pages in {took:.1f} s: {len(got) / took:.2f} pages a second, {sum(r['tokens'] for r in got) / took:.0f} tokens a second, concurrency {concurrency}; cut at the token limit: {sum(r['finish'] == 'length' for r in got)}")
