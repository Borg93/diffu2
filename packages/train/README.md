# diffu-train

NeMo AutoModel glue.

- **Training configs:**
  - `configs/flux3/`: the FLUX.3 LoRA runs, v2 for style by name and v3 for style by example.
  - `configs/diffu/`: diffu trained from our own random weights.
- **Dataloader configs** those YAMLs name: `diffu_train.nemo`, with the aspect buckets interleaved.
- **Caching:**
  - `python -m diffu_train.cache_diffu`: diffu's training cache, made with diffu's own encoder steps (`diffu_train.cache`).
  - `python -m diffu_train.cache_pairs`: the v3 pair caching.

The commands and `diffu_train.nemo` run only in NeMo AutoModel's environment. `diffu_train.cache`, `diffu_train.dropout`
and `diffu_train.sampler` also run and are tested here.
