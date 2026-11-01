# diffu-flux3

FLUX.3 Klein LoRA training with NeMo AutoModel ([docs/FLUX3.md](../../docs/FLUX3.md)).

- **Training configs** (`configs/`): `flux3_layout_lora_v2.yaml`, the text-to-image LoRA (style by name; the
  per-hand LoRAs' recipe), and `flux3_style_ref_lora_v3d.yaml`, the style-reference LoRA (style by example), run by
  `jobs/flux3/run_v3d.sh`. NeMo's own `examples/diffusion/finetune/finetune.py` trains with them.
- **Dataloader configs** those YAMLs name: `diffu_flux3.nemo`, NeMo's with the aspect buckets interleaved
  (`diffu_flux3.sampler`) and caption dropout on the real empty prompt (`diffu_flux3.dropout`).
- **Caching**: `python -m diffu_flux3.cache_pairs`, the style pairs (`diffu_page.data.style_pairs`) through NeMo's
  FLUX.3 image-edit encoder, with the empty prompt beside them.

`diffu_flux3.cache_pairs` and `diffu_flux3.nemo` run only in NeMo AutoModel's environment; `diffu_flux3.dropout` and
`diffu_flux3.sampler` also run and are tested here.
