# diffu-train

NeMo AutoModel glue. It holds:

- the training configs: `configs/flux3/` has the FLUX.3 LoRA runs, v2 for style by name and v3 for style by example;
- the dataloader configs those YAMLs name: `diffu_train.nemo`, with aspect buckets interleaved;
- the v3 pair caching: `python -m diffu_train.cache_pairs`.

It is importable only in NeMo AutoModel's environment.
