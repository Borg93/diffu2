"""diffu's training entry point: NeMo AutoModel's diffusion recipe, a glyph model trained with its own adapter.

Run as NeMo's own ``examples/diffusion/finetune/finetune.py`` is run, from its checkout and in its environment:

    cd ~/Desktop/repos/Automodel && .venv-managed/bin/torchrun --nproc-per-node=1 -m diffu_train.finetune -c <config>

NeMo picks the flow-matching adapter by name (``flow_matching.adapter_type``) from a table of its own, which a config
cannot extend. :class:`DiffuDiffusionRecipe` is NeMo's recipe with that adapter replaced, once NeMo has set the run
up, by the one :func:`diffu_train.nemo.adapter_for` gives the diffu repo it trains (``model.pretrained_model_name_or_path``):
a glyph model (``diffu-init --glyph``) trains with ``GlyphConcatAdapter``; any other model trains as with NeMo's own
entry point.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import override

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.diffusion.train import TrainDiffusionRecipe

from diffu.repo import ModelIndex
from diffu_train.nemo import adapter_for

logger = logging.getLogger(__name__)


class DiffuDiffusionRecipe(TrainDiffusionRecipe):
    """NeMo's diffusion recipe, with a glyph model's flow-matching adapter in place of NeMo's.

    Raises:
        ValueError: From ``setup``, if the model is no diffu repo, or a glyph model to train with another adapter than
            FLUX.3's.
    """

    @override
    def setup(self) -> None:
        super().setup()
        model = Path(self.model_id)
        adapter = adapter_for(ModelIndex.read(model), self.adapter_type)
        if adapter is not None:
            self.flow_matching_pipeline.model_adapter = adapter
            logger.info("glyph concat: %s takes its glyph map beside its latents (%s)", model, type(adapter).__name__)


def main() -> None:
    """Train with the config given by ``-c``, NeMo's command-line overrides applied."""
    recipe = DiffuDiffusionRecipe(parse_args_and_load_config())
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()
