# diffu-gen

Generation with Modular Diffusers: FLUX.3's own blocks with ours around them (crop prompt, reference grid, HTR
read-back, label export), split into a text pipeline and a crop pipeline over one ComponentsManager; `CropGenerator`
draws one crop plan into one image with its labels. CLIs: `diffu-generate`, `diffu-evaluate`, `diffu-sheets`.
