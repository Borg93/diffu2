"""Draw one image with SGLang Diffusion and save its pixels; run by the equivalence test in SGLang's environment.

    python sglang_draw.py <request.json> <out.npy>

The request holds the model folder (``model``), SGLang's sampling parameters (``params``) and its server arguments
(``server``). The pixels are SGLang's output, ``[3, H, W]`` in [0, 1], saved as float32.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sglang.multimodal_gen import DiffGenerator

if __name__ == "__main__":
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    generator = DiffGenerator.from_pretrained(model_path=request["model"], **request["server"])
    try:
        result = generator.generate(sampling_params_kwargs={**request["params"], "save_output": False})
        if result is None:
            raise SystemExit("SGLang drew nothing: its log above says why")
        np.save(sys.argv[2], np.asarray(result.samples, dtype=np.float32))
    finally:
        generator.shutdown()
