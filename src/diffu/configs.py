"""What fixes a diffu model: its transformer size, its text encoder with the layers it stacks, and its VAE.

diffu is FLUX.3's architecture with our own weights: diffusers' ``Flux3Transformer2DModel`` built from a config with
random weights, FLUX.3's own text encoder (Qwen3-VL-4B-Instruct, the same layers stacked) and the Qwen-Image VAE (both
Apache-2.0). No FLUX.3 transformer weights, outputs or autoencoder: FLUX.3's autoencoder is under its non-commercial
licence.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# The Qwen3-VL hidden layers FLUX.3 stacks into its prompt embeddings (diffusers' Flux3TextEncoderStep)
FLUX3_TEXT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)


class DiffuSize(BaseModel):
    """A transformer size: attention heads, then the blocks each token mode runs alone and the joint blocks.

    Attributes:
        num_attention_heads: Heads per attention layer; the hidden width is ``num_attention_heads * attention_head_dim``.
        mode_layers: Blocks each token mode (image, reference, text) runs on its own before the joint blocks.
        joint_layers: Blocks over the joint sequence of all modes.
        attention_head_dim: The width of one head, split equally over FLUX.3's four RoPE axes (t, h, w, l).
    """

    model_config = {"frozen": True}

    num_attention_heads: int
    mode_layers: int
    joint_layers: int
    attention_head_dim: int = 128


DIFFU_S = DiffuSize(num_attention_heads=12, mode_layers=2, joint_layers=12)  # hidden 1536, 618M parameters
DIFFU_M = DiffuSize(num_attention_heads=16, mode_layers=4, joint_layers=16)  # hidden 2048, 1.64B parameters
SizeName = Literal["S", "M"]  # an assignment, not a type statement: typer reads it at runtime
SIZES: dict[SizeName, DiffuSize] = {"S": DIFFU_S, "M": DIFFU_M}


class TextEncoder(BaseModel):
    """The text encoder and the hidden layers whose states are stacked into the prompt embeddings: FLUX.3's.

    Attributes:
        repo: A Qwen3-VL model: Hub repo id or local directory.
        revision: Its pinned Hub revision; ``None`` for a local directory.
        encoder_subfolder: Where the model sits in the repo.
        processor_subfolder: Where its processor (tokenizer and chat template) sits in the repo.
        layers: The ``hidden_states`` indices stacked on channels.
        hidden_size: The encoder's hidden width.
    """

    model_config = {"frozen": True}

    # byte-identical to the text_encoder/ of black-forest-labs/flux-3-image-klein-base, from Qwen's own repo
    repo: str = "Qwen/Qwen3-VL-4B-Instruct"
    revision: str | None = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
    encoder_subfolder: str = ""
    processor_subfolder: str = ""
    layers: tuple[int, ...] = FLUX3_TEXT_LAYERS
    hidden_size: int = 2560


class Vae(BaseModel):
    """The image VAE: Qwen-Image's, 16 latent channels at 8x downsampling.

    Attributes:
        repo: Hub repo id or local directory.
        revision: Its pinned Hub revision; ``None`` for a local directory.
        subfolder: Where the VAE sits in the repo.
        latent_channels: The VAE's latent channels (its ``z_dim``).
    """

    model_config = {"frozen": True}

    repo: str = "Qwen/Qwen-Image"
    revision: str | None = "75e0b4be04f60ec59a75f475837eced720f823b6"
    subfolder: str = "vae"
    latent_channels: int = 16


class DiffuConfig(BaseModel):
    """A diffu model: its transformer size, its text encoder and its VAE."""

    model_config = {"frozen": True}

    size: DiffuSize
    text_encoder: TextEncoder = TextEncoder()
    vae: Vae = Vae()

    def transformer_config(self) -> dict[str, object]:
        """The ``Flux3Transformer2DModel`` config: this size, the widths the VAE and the text encoder give, and FLUX.3's
        own defaults for the rest (timestep channels, MLP ratio, RoPE theta, norm eps, no flow map)."""
        packed_channels = 4 * self.vae.latent_channels  # each 2x2 latent patch packed on channels, as FLUX.3 packs
        return {
            "in_channels": packed_channels,
            "cond_in_channels": packed_channels,
            "num_layers": self.size.mode_layers,
            "num_single_layers": self.size.joint_layers,
            "attention_head_dim": self.size.attention_head_dim,
            "num_attention_heads": self.size.num_attention_heads,
            "joint_attention_dim": len(self.text_encoder.layers) * self.text_encoder.hidden_size,
            "axes_dims_rope": (self.size.attention_head_dim // 4,) * 4,
        }
