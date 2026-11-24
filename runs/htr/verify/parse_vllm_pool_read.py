"""Every page of a reader set read through vLLM servers of Nemotron 3.5 Parse, appending, resumable (CURATION E50).

    SET=<set stem> URLS=http://127.0.0.1:8011/v1,http://127.0.0.1:8012/v1 uv run --with httpx python runs/htr/verify/parse_vllm_pool_read.py OUT.jsonl [CONCURRENCY [LIMIT]]

As pool_read.py (same order, same rows: id, subset, text, prediction, raw, tokens, seconds) but many pages in flight,
shared over the servers named. The request is the model card's (temperature 0, top-k 1, repetition penalty 1.1,
special tokens kept), with at most MAX_NEW tokens: vLLM has no repetition guard yet, a page is about 450 tokens, and a
page that reaches the cap is written with ``finish`` "length" so it can be read again with the guarded reader.
"""
import base64, json, os, random, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import httpx

PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"; MAX_NEW = int(os.environ.get("MAX_NEW", "3000"))
LAYOUT = re.compile(r"<(?:x|y)_[0-9.]+>|<class_[A-Za-z-]+>")
TABLE = re.compile(r"\\begin\{tabular\}\{[^}]*\}|\\end\{tabular\}|\\linebreak\{\}|\\\\| & ")
def text(answer: str) -> str: return " ".join(TABLE.sub(" ", LAYOUT.sub("\n", answer)).replace("#", " ").split())

out = sys.argv[1]; concurrency = int(sys.argv[2]) if len(sys.argv) > 2 else 64; limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
SET = os.environ["SET"]; urls = os.environ.get("URLS", "http://127.0.0.1:8011/v1").split(",")
rows = open(f"{SET}.jsonl", encoding="utf-8").read().split("\n")  # not splitlines: a label may hold U+0085, which it splits at (24 Nov: pool_v3 read under the next page's id)
rows = rows[:-1] if rows and rows[-1] == "" else rows
metas = [json.loads(l) for l in open(f"{SET}.meta.jsonl", encoding="utf-8")]
assert len(rows) == len(metas), f"{len(rows)} pages, {len(metas)} lines of meta"
done = {json.loads(l)["id"] for l in open(out, encoding="utf-8")} if os.path.exists(out) else set()
order = list(range(len(metas))); random.Random(0).shuffle(order)
picked = [i for i in order if metas[i]["id"] not in done][:limit]
client = httpx.Client(timeout=1800, limits=httpx.Limits(max_connections=max(64, concurrency))); lock = threading.Lock(); count = [0]; start = time.perf_counter()

def read(n_i):
    n, i = n_i; row = json.loads(rows[i]); meta = metas[i]; started = time.perf_counter()
    image = base64.b64encode(open(row["images"][0], "rb").read()).decode()
    body = {"model": "nemotron-parse", "temperature": 0, "max_tokens": MAX_NEW, "top_k": 1, "repetition_penalty": 1.1, "skip_special_tokens": False,
            "repetition_detection": {"min_pattern_size": 4, "max_pattern_size": 120, "min_count": 6},  # vLLM stops a looping page (E51)
            "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image}}]}]}
    try:
        answer = client.post(f"{urls[n % len(urls)]}/chat/completions", json=body).raise_for_status().json()
    except httpx.HTTPError as error:  # the page stays unread and a later run takes it
        print("failed", meta["id"], type(error).__name__, flush=True); return
    choice = answer["choices"][0]; raw = (choice["message"]["content"] or "").replace("</s>", "").replace("<s>", "").replace("<pad>", "")
    line = json.dumps({"id": meta["id"], "subset": meta["collection"] or meta["volume"], "text": text(row["messages"][1]["content"]), "prediction": text(raw), "raw": raw,
                       "tokens": answer["usage"]["completion_tokens"], "seconds": round(time.perf_counter() - started, 2), "finish": choice["finish_reason"]}, ensure_ascii=False)
    with lock:
        f.write(line + "\n"); f.flush(); count[0] += 1
        if count[0] % 1000 == 0: print(f"{count[0]} of {len(picked)} pages, {count[0] / (time.perf_counter() - start):.2f} a second", flush=True)

print(len(metas), "pages;", len(done), "read before;", len(picked), "to read", flush=True)
with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(concurrency) as pool:
    list(pool.map(read, enumerate(picked)))
print(f"done {count[0]} of {len(picked)} in {time.perf_counter() - start:.0f} s", flush=True)
