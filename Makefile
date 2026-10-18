# Diffu task runner (uv-based). Override any var on the CLI, e.g.:
#   make data XML_DIR=/path/xml IMAGE_DIR=/path/imgs OUT_DIR=out
#   make train-2gpu STEPS=50000 LR=5e-5 BATCH=16
#
# GPU pinning: GPU 0 hosts a foreign job (rerun_plus_olof) — the free pair is 1+2, and the next
# planned run is 2-GPU, so `make train` defaults to GPUS=1,2. Override GPUS / NPROC / GPUS_1 /
# GPUS_2 to retarget.

# --- data prep ---
XML_DIR     ?= data_raw/xml          # PAGE/ALTO XML (for `make data`)
IMAGE_DIR   ?= data_raw/images       # page images (for `make data`)
DIR         ?= data_raw/crops        # folder tree of pre-cropped lines (for `make ingest`)
OUT_DIR     ?= data_out_lejonet_128
LINE_HEIGHT ?= 128   # matches Config() and the released Gabriel/diffu_sd35_128

# --- training ---
DATA_DIR ?= $(OUT_DIR)
STEPS    ?= 100000
LR       ?= 1e-4
BATCH    ?= 64   # per-GPU; ~45GB/97GB at REPA-on, 144 img/s on 3 GPUs (matches DiffBrush IMS_PER_BATCH)

# --- gpu selection ---
GPUS   ?= 1,2      # default: the free pair (GPU 0 = foreign rerun_plus_olof job — leave it alone)
NPROC  ?= 2        # processes for `make train` — one per GPU listed in GPUS
GPUS_1 ?= 1        # single-GPU targets use this device
GPUS_2 ?= 1,2      # dual-GPU targets use these devices

ACCEL = uv run accelerate launch --mixed_precision bf16

.PHONY: help install install-dev sync data ingest \
        train train-1gpu train-2gpu smoke-1gpu smoke-2gpu smoke-compile vae-gate \
        check lint fmt test clean

help:
	@echo "Diffu — make targets:"
	@echo "  install      uv sync (runtime deps; torch from the cu130 index)"
	@echo "  install-dev  uv sync --extra dev (adds ruff / pyrefly / pytest)"
	@echo "  data         PAGE/ALTO XML -> line crops + manifest + writer-disjoint split"
	@echo "  ingest       pre-cropped line folders (DIR=) -> manifest + writer-disjoint split"
	@echo "  train        Stage-1 training on GPUs $(GPUS) (default: the free pair)"
	@echo "  train-1gpu   Stage-1 training on GPU $(GPUS_1)"
	@echo "  train-2gpu   Stage-1 training on GPUs $(GPUS_2) (accelerate --multi_gpu)"
	@echo "  smoke-1gpu   synthetic forward/backward on GPU $(GPUS_1) (no dataset needed)"
	@echo "  smoke-2gpu   synthetic multi-GPU step on GPUs $(GPUS_2) (no dataset needed)"
	@echo "  smoke-compile  regional torch.compile the backbone + a few steps (compile-path smoke)"
	@echo "  vae-gate     Stage-0: does the VAE preserve å ä ö? (recon-CER on $(OUT_DIR)/val.jsonl)"
	@echo "  check        pyrefly type-check"
	@echo "  lint / fmt   ruff check / ruff format + autofix"
	@echo "  test         run the pytest suite"
	@echo "  clean        remove generated data_out artifacts"
	@echo ""
	@echo "override: XML_DIR IMAGE_DIR OUT_DIR LINE_HEIGHT DATA_DIR STEPS LR BATCH GPUS NPROC GPUS_1 GPUS_2"

install:
	uv sync

install-dev:
	uv sync --extra dev

sync: install

data:
	uv run diffu-data \
		--xml-dir $(XML_DIR) --image-dir $(IMAGE_DIR) \
		--out-dir $(OUT_DIR) --height $(LINE_HEIGHT)

ingest:
	uv run diffu-ingest --dir $(DIR) --out-dir $(OUT_DIR)

train:
	CUDA_VISIBLE_DEVICES=$(GPUS) $(ACCEL) --multi_gpu --num_processes $(NPROC) \
		-m diffu.train --data-dir $(DATA_DIR) --out-dir $(OUT_DIR)/checkpoints \
		--steps $(STEPS) --lr $(LR) --batch-size $(BATCH)

train-1gpu:
	CUDA_VISIBLE_DEVICES=$(GPUS_1) $(ACCEL) --num_processes 1 \
		-m diffu.train --data-dir $(DATA_DIR) --out-dir $(OUT_DIR)/checkpoints \
		--steps $(STEPS) --lr $(LR) --batch-size $(BATCH)

train-2gpu:
	CUDA_VISIBLE_DEVICES=$(GPUS_2) $(ACCEL) --multi_gpu --num_processes 2 \
		-m diffu.train --data-dir $(DATA_DIR) --out-dir $(OUT_DIR)/checkpoints \
		--steps $(STEPS) --lr $(LR) --batch-size $(BATCH)

smoke-1gpu:
	CUDA_VISIBLE_DEVICES=$(GPUS_1) uv run diffu-smoke --steps 3 --batch-size 2

smoke-compile:   # compile-path smoke: regional torch.compile the backbone + a few real steps
	CUDA_VISIBLE_DEVICES=$(GPUS_1) uv run diffu-smoke --steps 4 --batch-size 2 --compile

vae-gate:
	CUDA_VISIBLE_DEVICES=$(GPUS_1) uv run diffu-vae-gate \
		--manifest $(OUT_DIR)/val.jsonl --n 500 --save-dir $(OUT_DIR)/vae_check

smoke-2gpu:
	CUDA_VISIBLE_DEVICES=$(GPUS_2) $(ACCEL) --multi_gpu --num_processes 2 \
		-m diffu.smoketest --steps 3 --batch-size 2

check:
	uvx pyrefly check          # isolated/torch-free: checks our logic, not torch's stubs

lint:
	uv run ruff check src tests

fmt:
	uv run ruff format src tests
	uv run ruff check --fix src tests

test:
	uv run pytest

clean:
	rm -rf $(OUT_DIR)/lines $(OUT_DIR)/*.jsonl $(OUT_DIR)/manifest.jsonl
