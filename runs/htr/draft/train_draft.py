"""NeMo AutoModel's DFlash2 recipe on a cache of a Nemotron Parse reader (``make_cache.py``), without the reader.

AutoModel (commit 0f6814b) has the cached route only in ``TrainDFlashRecipe``: ``_validate_cached_inputs`` refuses
any subclass, and ``TrainDFlash2Recipe._build_trainer_module`` takes the head and the embedding from a live target.
``ParseDFlash2Recipe`` below changes those two things and nothing else, so the AutoModel clone stays as it is:

- the cached route is let through for this class (the checks themselves are the recipe's own);
- the DFlash2 trainer module is given the cached target tables (``recipe_args.target_embedding_key`` is the
  embedding times sqrt(d_model), ``target_lm_head_key`` the tied table itself: see ``make_cache.write_tables``).

Three more things a draft for vLLM needs, also here:

- ``recipe_args.init_draft_path``: a draft's ``model.safetensors`` (NVIDIA's ``dflash2/``) to start from;
- ``--eval-only``: the recipe's own validation over ``val_cache_path``, printed as one JSON line, nothing trained;
- ``--export DIR``: the draft written as NVIDIA's ``dflash2/`` is (``config.json`` + ``model.safetensors``, bf16,
  the same keys), after training or straight after loading.

Run with ``/mnt/scratch/gabriel/envs/automodel/bin/python`` and ``CUDA_VISIBLE_DEVICES=2``::

    python runs/htr/draft/train_draft.py -c runs/htr/draft/dflash2_parse.yaml [--eval-only] [--export DIR] \\
        [--recipe_args.train_cache_path ...]
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.speculative.dflash.dflash2_core import DFlash2TrainerModule
from nemo_automodel.recipes.speculative_decoding.train_dflash2 import TrainDFlash2Recipe
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)

# NVIDIA's dflash2/config.json, key for key: what vLLM's DFlash2 draft is built from
_TOP_KEYS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "intermediate_size",
    "hidden_act",
    "rms_norm_eps",
    "vocab_size",
    "max_position_embeddings",
    "initializer_range",
    "attention_bias",
    "attention_dropout",
)
_DFLASH_KEYS = ("conv_kernel_size", "conv_group_size", "selector_rank", "selector_top_k")


class ParseDFlash2Recipe(TrainDFlash2Recipe):
    """``TrainDFlash2Recipe`` reading an offline cache and the cached target tables."""

    def _validate_cached_inputs(self) -> tuple[str | None, str | None]:
        """``TrainDFlashRecipe._validate_cached_inputs`` without its "this class only" line."""
        args = self.cfg.recipe_args
        train_cache, val_cache = args.get("train_cache_path"), args.get("val_cache_path")
        for name, path in (("train_cache_path", train_cache), ("val_cache_path", val_cache)):
            if not isinstance(path, str) or not path.strip():
                raise ValueError(f"recipe_args.{name} must be a nonempty path: this recipe trains from a cache only")
        if args.get("train_data_path") or args.get("val_data_path"):
            raise ValueError("Use cache paths, not raw data paths")
        if args.get("packed_sequence_size") or args.get("mask_reasoning_content") or args.get("mask_generation_prompt"):
            raise ValueError("Packing and the loss-mask options do not apply to a cache")
        if any(int(self.cfg.get(f"distributed.{axis}_size", 1) or 1) > 1 for axis in ("cp", "tp", "pp", "ep")):
            raise ValueError("Cached DFlash supports only unsharded or data-parallel target tables")
        if float(args.get("input_embedding_scale", 1.0)) != 1.0:
            # vLLM multiplies by it on top of Parse's own scaled embedding module: the scale is in the table instead
            raise ValueError("input_embedding_scale must stay 1.0: the cached embedding table is already scaled")
        return train_cache, val_cache

    def _build_trainer_module(self, attention_backend: str, recipe_cfg) -> DFlash2TrainerModule:
        """``TrainDFlash2Recipe._build_trainer_module`` with the cached tables in the live target's place."""
        tables = self._cached_target_tables
        return DFlash2TrainerModule(
            draft_model=self.draft_model,
            target_lm_head=tables.lm_head,
            target_embed_tokens=tables.embed_tokens,
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
            attention_backend=attention_backend,
            num_anchors=int(recipe_cfg.get("num_anchors", 512)),
            loss_decay_gamma=recipe_cfg.get("loss_decay_gamma", 7.0),
            loss_type=str(recipe_cfg.get("loss_type", None) or "dflash"),
            dpace_alpha=float(recipe_cfg.get("dpace_alpha", 0.5)),
            selector_loss_weight=float(recipe_cfg.get("selector_loss_weight", 1.0)),
            sliding_window=self.draft_sliding_window,
        )

    def setup(self) -> None:
        """Build as the recipe does, then start from ``init_draft_path`` unless a checkpoint was restored."""
        super().setup()
        init = self.cfg.recipe_args.get("init_draft_path", None)
        if init and self.runtime.global_step == 0:
            load_draft(self.draft_model, Path(init))


def load_draft(draft: torch.nn.Module, weights: Path) -> None:
    """Load a published draft's weights into the trainer's draft: the names are the same, nothing is renamed."""
    state = load_file(str(weights))
    own = draft.state_dict()
    missing = sorted(set(own) - set(state))
    unexpected = sorted(set(state) - set(own))
    wrong = sorted(key for key in set(own) & set(state) if own[key].shape != state[key].shape)
    if missing or unexpected or wrong:
        raise ValueError(f"{weights}: missing {missing}, unexpected {unexpected}, other shapes {wrong}")
    draft.load_state_dict(state, strict=True)
    logger.info("draft started from %s: %d tensors, none missing, none left over", weights, len(state))


def export_draft(draft: torch.nn.Module, folder: Path) -> None:
    """Write the draft as NVIDIA's ``dflash2/`` folder is laid out, for ``--speculative-config`` in vLLM."""
    folder.mkdir(parents=True, exist_ok=True)
    config = draft.config.to_dict()
    dflash = config["dflash_config"]
    if float(dflash.get("input_embedding_scale", 1.0)) != 1.0 or float(dflash.get("output_multiplier", 1.0)) != 1.0:
        raise ValueError("the draft was built with a scale vLLM would apply a second time")
    layers = config["num_hidden_layers"]
    written = {
        "architectures": ["DFlash2DraftModel"],
        "model_type": "qwen3",
        "block_size": config["block_size"],
        "dflash_config": {
            "mask_token_id": dflash["mask_token_id"],
            "target_layer_ids": list(dflash["target_layer_ids"]),
            "causal": False,
            "projector_type": "dflash2",
            **{key: dflash[key] for key in _DFLASH_KEYS},
            "block_size": dflash["block_size"],
        },
        **{key: config[key] for key in _TOP_KEYS},
        "rope_theta": (config.get("rope_parameters") or {}).get("rope_theta", config.get("rope_theta")),
        "rope_scaling": None,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "num_target_layers": config["num_target_layers"],
        "layer_types": ["full_attention"] * layers,
        "is_causal": False,
    }
    (folder / "config.json").write_text(json.dumps(written, indent=2) + "\n", "utf-8")
    state = {key: value.detach().to("cpu", torch.bfloat16).contiguous() for key, value in draft.state_dict().items()}
    save_file(state, str(folder / "model.safetensors"))
    logger.info("draft exported to %s (%d tensors)", folder, len(state))


def main() -> None:
    """Train (or only validate) a DFlash2 draft from a Parse cache."""
    argv = sys.argv[1:]
    eval_only = "--eval-only" in argv
    export = None
    if "--export" in argv:
        export = Path(argv[argv.index("--export") + 1])
        del argv[argv.index("--export") : argv.index("--export") + 2]
    argv = [arg for arg in argv if arg != "--eval-only"]
    recipe = ParseDFlash2Recipe(parse_args_and_load_config(argv=argv))
    recipe.setup()
    if eval_only:
        torch.manual_seed(int(recipe.cfg.recipe_args.get("shuffle_seed", 42)))  # the anchors are drawn at random
        result = recipe._run_eval()  # noqa: SLF001  the recipe's own validation, as run after each epoch
        result["val_pages"] = len(recipe.val_dataloader.dataset)
        result["block_size"] = recipe.block_size
        print("VALIDATION " + json.dumps(result), flush=True)
    else:
        recipe.run_train_validation_loop()
    if export is not None:
        export_draft(recipe.draft_model, export)


if __name__ == "__main__":
    main()
