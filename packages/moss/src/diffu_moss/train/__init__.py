"""Our own training of MOSS-Image to take a reference image (a sample page of the same volume): a LoRA on the DiT, or
every weight of it (:mod:`diffu_moss.train.full`, MOSS-Image's own Muon and AdamW recipe), trained on style pairs
(:mod:`diffu_moss.train.pairs`) from offline caches of the prompts' embeddings and the images' latents
(:mod:`diffu_moss.train.cache`), with MOSS-Image's flow-matching objective and noise levels (:mod:`diffu_moss.train.loop`,
:mod:`diffu_moss.train.sigmas`); the LoRA merged into a DiT folder that ``diffu-moss-init`` draws with
(:mod:`diffu_moss.train.export`), the full fine-tune's checkpoints such folders themselves."""
