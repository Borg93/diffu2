"""Launch the gated PAGE pipeline. Env overrides: OUT, TARGET, GPUS, BATCH, SEED.

Runs from /tmp so Ray's captured working_dir stays small (the repo root would ship checkpoints/data
to every worker despite .rayignore pruning most of it).
"""

import os

from diffu_runner.pages import PageConfig, run_pages

D = "/home/blackwell/Desktop/diffu"
OUT = os.environ.get("OUT", f"{D}/synth_pages/pages")
TARGET = int(os.environ.get("TARGET", "10000"))
GPUS = int(os.environ.get("GPUS", "1"))
BATCH = int(os.environ.get("BATCH", "24"))
SEED = int(os.environ.get("SEED", "1"))

os.makedirs("/tmp/ray_wd_pages", exist_ok=True)
os.chdir("/tmp/ray_wd_pages")

cfg = PageConfig(
    ckpt=f"{D}/checkpoints/exp_sd35_128/step_120000/model.safetensors",
    train_jsonl=f"{D}/data_out_lejonet_128/train.jsonl",
    out=OUT,
    paper_dir=f"{D}/assets/paper_gtl",
)
path, stats = run_pages(cfg, target=TARGET, gpus=GPUS, batch=BATCH, seed=SEED)
print("PAGES DONE:", stats)
