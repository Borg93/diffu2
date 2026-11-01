"""The pixel grid FLUX.3 latents need: image sides in multiples of 16, under a pixel budget.

NeMo's FLUX.3 cache encoder resizes every image it caches onto this grid (squashing, never padding), so an image
already on it reaches training and generation unchanged.
"""

from __future__ import annotations

from PIL import Image

from diffu_page.crops.geometry import BUCKETS

LATENT_GRID = 16  # px: 8x VAE downsampling, then 2x2 patches
LARGEST_BUCKET_PIXELS = max(width * height for width, height in BUCKETS)  # NeMo caches images under this many
REF_PIXELS = 512 * 1024  # a reference image's size: the hand smaller than in the ~1 MP target, at half its tokens


def as_reference(image: Image.Image) -> Image.Image:
    """``image`` as a reference image is drawn from, in training and generation alike: scaled under ``REF_PIXELS``,
    each side floored to a multiple of ``LATENT_GRID`` as NeMo's cache encoder resizes it (squashed onto the grid,
    never padded), so an image already there is unchanged."""
    scale = min(1.0, (REF_PIXELS / (image.width * image.height)) ** 0.5)
    width = max(LATENT_GRID, int(image.width * scale) // LATENT_GRID * LATENT_GRID)
    height = max(LATENT_GRID, int(image.height * scale) // LATENT_GRID * LATENT_GRID)
    return image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
