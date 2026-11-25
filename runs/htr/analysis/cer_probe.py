import sys, time
from pathlib import Path
import torch
from swift.infer_engine import TransformersEngine
from diffu_htr_train.evaluate.in_training import CerSet, metrics
from diffu_htr_train.evaluate.lines import read_lines
from diffu_htr_train.evaluate.swift_plugin import swift_read
ckpt, mode = sys.argv[1], sys.argv[2]
engine = TransformersEngine(ckpt, model_type="minicpm_ocr", template_type="minicpm_ocr", torch_dtype=torch.bfloat16, attn_impl="sdpa")
if mode == "as_training":  # what the training model looks like: checkpointing on, config.use_cache off
    engine.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    engine.model.config.use_cache = False
    if hasattr(engine.model.config, "text_config"): engine.model.config.text_config.use_cache = False
engine.model.eval()
d = Path("data/htr/eval/in_training")
for name, s, n in [("dev_lines", CerSet(lines=d/"dev_lines.jsonl", max_tokens=512, batch=16), 64), ("dev_regions", CerSet(lines=d/"dev_regions.jsonl", max_tokens=4096, batch=8), 16), ("ood", CerSet(lines=d/"ood.jsonl", max_tokens=512, batch=16), 64)]:
    lines = read_lines(s.lines)[:n]
    t = time.perf_counter(); r = swift_read(engine, lines, s); dt = time.perf_counter() - t
    print(name, f"{n} in {dt:.1f}s", {k: round(v, 4) for k, v in metrics(name, lines, r).items()})
    print("  ", repr(lines[0].text[:80]), "->", repr(r[0].prediction[:80]), r[0].finished)
print("peak GB", torch.cuda.max_memory_allocated() / 1e9)
