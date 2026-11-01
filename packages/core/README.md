# diffu-core

What every member shares about a model directory (a Modular Diffusers repo, MOSS-Image's or FLUX.3 Klein's):

- `repo.py`: its `modular_model_index.json` (`ModelIndex`: the blocks class it names, where each component loads from;
  `diffu-moss-init` writes it, the page pipeline reads which model family a directory is).
- `loading.py`: its components loaded in the dtypes drawing and training share (`COMPONENT_DTYPES`: the VAE in
  float32), without letting one that failed to load through (`require_loaded`).
