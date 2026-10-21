# diffu

Our own page model: FLUX.3's architecture (diffusers' `Flux3Transformer2DModel`, FLUX.3's denoise blocks) with our own
weights, a Qwen3.5 text encoder and the Qwen-Image VAE, laid out like a diffusers modular pipeline
(`diffu.modular_pipelines.diffu`). `diffu-init` writes a freshly initialised model repo for NeMo AutoModel to train
and `DiffuModularPipeline.from_pretrained` to load.
