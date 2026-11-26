"""The captured pages (capture_generate.py) drawn by SGLang Diffusion with diffu_sglang_moss, one engine, one page at a
time: 28 steps, guidance 2 (SGLang guides every step), the plugin's own negative prompt and skipped block. Prints one
JSON line a page (wall seconds of the generate call) and saves each page. Run in the SGLang environment with
SGLANG_EXTERNAL_MODEL_PACKAGE=diffu_sglang_moss.

    python sglang_bench.py MODEL CAPTURE_DIR OUT_DIR '<server arguments as JSON>' ['<extra sampling parameters as JSON>']
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sglang.multimodal_gen import DiffGenerator


def main() -> None:
    model, captured, out, server = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), json.loads(sys.argv[4])
    extra = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
    out.mkdir(parents=True, exist_ok=True)
    if server.get("enable_breakable_cuda_graph") and "warmup_resolutions" not in server:  # a graph is captured per size
        sizes = [json.loads(path.read_text()) for path in sorted(captured.glob("page_*.json"))]
        server["warmup_resolutions"] = sorted({f"{size['width']}x{size['height']}" for size in sizes})
    started = time.time()
    generator = DiffGenerator.from_pretrained(model_path=model, num_gpus=1, **server)
    print(json.dumps({"server": server, "extra": extra, "load_seconds": round(time.time() - started, 1)}), flush=True)
    try:
        for k in range(len(list(captured.glob("page_*.txt")))):
            size = json.loads((captured / f"page_{k}.json").read_text())
            params = {
                "prompt": (captured / f"page_{k}.txt").read_text(encoding="utf-8"),
                "image_path": str(captured / f"page_{k}.png"),
                "width": size["width"], "height": size["height"],
                "num_inference_steps": 28, "guidance_scale": 2.0, "seed": 7, "save_output": False,
                **extra,
            }
            began = time.time()
            result = generator.generate(sampling_params_kwargs=params)
            seconds = round(time.time() - began, 2)
            if result is None:
                print(json.dumps({"page": k, "error": "SGLang drew nothing"}), flush=True)
                continue
            first = result[0] if isinstance(result, list) else result  # several images a call come back as a list
            samples = torch.as_tensor(first.samples).float().cpu()
            pixels = (samples[0] if samples.ndim == 4 else samples).numpy().transpose(1, 2, 0)
            Image.fromarray((pixels * 255).round().astype(np.uint8)).save(out / f"page_{k}.jpg", quality=95)
            print(json.dumps({"page": k, "seconds": seconds, "images": int(params.get("num_outputs_per_prompt", 1)), **size}), flush=True)
    finally:
        generator.shutdown()


if __name__ == "__main__":  # SGLang starts its worker by spawn, which imports this file again
    main()
