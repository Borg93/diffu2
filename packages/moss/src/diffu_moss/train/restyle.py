"""Paired look changes ("restyles"): one photometric transform, its parameters drawn once, applied alike to a pair's
target and its sample image before both are encoded (``diffu-moss-cache-pairs --restyle-share``,
:mod:`diffu_moss.train.cache`), and before ``diffu_page.cli.draw_pairs`` draws such a pair back.

In the eight hands' page-pair set each volume's pages and samples share one look, and half the volumes are microfilm,
so a model can take a page's look from its hand and leave its sample unread. A restyled pair's page and sample take one
new look together; the sample form names no look (``Variant.SAMPLE``), so its prompt stays true and the look can only
come from the sample. Three kinds, a third of the restyled pairs each (:func:`draw_restyles`):

- :class:`Microfilm`: a greyscale microfilm scan: the ink black, the paper a drawn grey, the darker part of the range
  between them crushed to black and the rest through a gamma;
- :class:`Colourise`: a colour scan: the range from the ink to the paper laid from a drawn ink colour to a drawn paper
  colour (cream, yellowed, off-white or white paper; black, brown or grey ink), so a microfilm volume's hand comes in
  colour too;
- :class:`Tint`: a scanner's white balance shifted warm or cool, its contrast, brightness and gamma jittered.

Microfilm and colourise read each image's own ink and paper (the median luminance of its darkest 2 % and of its
lightest 40 %, as a caption measures a scan's look: ``diffu_page.captions.look``), so a page and its sample, whatever
their scans' exposure, take the drawn grey or colours alike. Every transform is photometric: a pixel's new colour
depends on its own colour (and its image's ink and paper) alone, never on its place, so an image keeps its size and
every line its box, and the Layout prompt's boxes stay true. The ranges keep a restyled page a plausible archive scan,
as looked at beside the real scans of the eight hands' volumes.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Annotated, Literal

from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

type Rgb = tuple[int, int, int]  # 0-255

LUMA = (0.299, 0.587, 0.114)  # ITU-R 601-2, as PIL converts to greyscale
INK_SHARE = 0.02  # an image's ink: the median luminance of its darkest 2 %
PAPER_SHARE = 0.4  # its paper: the median of its lightest 40 %
LEAST_RANGE = 0.1  # the least luminance from ink to paper read, so a blank image keeps its paper

MICROFILM_CRUSH = (0.0, 0.22)  # the share of the range from ink to paper that comes out black
MICROFILM_PAPER = (0.70, 0.94)  # the paper's grey: the real microfilm volumes' measures 0.67-0.96
MICROFILM_GAMMA = (0.8, 1.25)
COLOUR_CRUSH = (0.0, 0.1)
COLOUR_GAMMA = (0.85, 1.2)
PAPERS: tuple[Rgb, ...] = (  # as the colour volumes' paper measures
    (224, 210, 189),  # cream (Jämtland)
    (229, 208, 168),  # yellowed (Överståthållarämbetet)
    (223, 212, 200),  # off-white (Påhlman)
    (236, 232, 222),  # white
)
INKS: tuple[Rgb, ...] = (  # and their ink
    (28, 22, 21),  # black (Jämtland, Överståthållarämbetet)
    (60, 44, 34),  # dark brown
    (89, 73, 61),  # brown (Göteborg)
    (72, 71, 70),  # grey
)
PAPER_BRIGHTNESS = (0.96, 1.03)  # a paper tone scaled by it, its hue kept
INK_BRIGHTNESS = (0.85, 1.15)
TINT_WARMTH = (-0.03, 0.03)  # red raised and blue lowered by it (cooler below 0); 0.04 turned a beige page orange
TINT_CONTRAST = (0.88, 1.12)
TINT_BRIGHTNESS = (-0.04, 0.04)
TINT_GAMMA = (0.87, 1.15)


def _luma(rgb: Tensor) -> Tensor:
    """The luminance ``[B, 1, H, W]`` of images ``[B, 3, H, W]`` in [0, 1]."""
    return (rgb * rgb.new_tensor(LUMA).view(1, 3, 1, 1)).sum(1, keepdim=True)


def _levels(luma: Tensor, crush: float, gamma: float) -> Tensor:
    """``luma`` ``[B, 1, H, W]`` as levels of each image's own range from its ink (0) to its paper (1), lighter going
    past 1: the share ``crush`` of that range from the ink up held at 0, the rest stretched over it, through
    ``gamma``."""
    ordered = luma.flatten(1).sort(dim=1).values
    count = ordered.shape[1]
    ink = ordered[:, : max(1, round(count * INK_SHARE))].median(dim=1).values.view(-1, 1, 1, 1)
    paper = ordered[:, count - max(1, round(count * PAPER_SHARE)) :].median(dim=1).values.view(-1, 1, 1, 1)
    level = 1 + (luma - paper) / (paper - ink).clamp(min=LEAST_RANGE)
    return ((level - crush) / (1 - crush)).clamp(min=0) ** gamma


def _colour(rgb: Rgb, like: Tensor) -> Tensor:
    """A 0-255 colour as a ``[1, 3, 1, 1]`` tensor in [0, 1], to broadcast over images like ``like``."""
    return like.new_tensor(rgb).view(1, 3, 1, 1) / 255


class Microfilm(BaseModel):
    """A greyscale microfilm scan: each image's ink black and its paper the ``paper`` grey, the share ``crush`` of the
    range between them from the ink up black, the rest through ``gamma``; the same grey in every channel."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["microfilm"] = "microfilm"
    crush: float = Field(ge=0, lt=1)
    paper: float = Field(gt=0, le=1)
    gamma: float = Field(gt=0)

    def apply(self, rgb: Tensor) -> Tensor:
        """Images ``[B, 3, H, W]`` in [0, 1] in this look."""
        grey = self.paper * _levels(_luma(rgb), self.crush, self.gamma)
        return grey.clamp(0, 1).repeat(1, 3, 1, 1)


class Colourise(BaseModel):
    """A colour scan: each image's range from its ink to its paper laid from the ``ink`` colour to the ``paper``
    colour, the share ``crush`` of it from the ink up the ink's colour, the rest through ``gamma``."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["colourise"] = "colourise"
    crush: float = Field(ge=0, lt=1)
    gamma: float = Field(gt=0)
    paper: Rgb
    ink: Rgb

    def apply(self, rgb: Tensor) -> Tensor:
        """Images ``[B, 3, H, W]`` in [0, 1] in this look."""
        level = _levels(_luma(rgb), self.crush, self.gamma)
        ink = _colour(self.ink, rgb)
        return (ink + (_colour(self.paper, rgb) - ink) * level).clamp(0, 1)


class Tint(BaseModel):
    """A scanner's white balance and exposure: red scaled by 1 + ``warmth`` and blue by 1 - ``warmth`` (cooler below
    0), the contrast scaled about mid-grey, the ``brightness`` added, then ``gamma``."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["tint"] = "tint"
    warmth: float
    contrast: float = Field(gt=0)
    brightness: float
    gamma: float = Field(gt=0)

    def apply(self, rgb: Tensor) -> Tensor:
        """Images ``[B, 3, H, W]`` in [0, 1] in this look."""
        balanced = rgb * rgb.new_tensor((1 + self.warmth, 1.0, 1 - self.warmth)).view(1, 3, 1, 1)
        return ((balanced - 0.5) * self.contrast + 0.5 + self.brightness).clamp(0, 1) ** self.gamma


type Restyle = Annotated[Microfilm | Colourise | Tint, Field(discriminator="kind")]


def restyled(image: Image.Image, restyle: Restyle) -> Image.Image:
    """``image`` in ``restyle``'s look: an RGB image of its size, 8 bits a channel, as a drawing is given a sample, so
    the cache encodes a restyled image as drawing would encode it."""
    rgb = VaeImageProcessor.numpy_to_pt(VaeImageProcessor.pil_to_numpy(image.convert("RGB")))
    return VaeImageProcessor.numpy_to_pil(VaeImageProcessor.pt_to_numpy(restyle.apply(rgb)))[0]


def _uniform(rng: random.Random, bounds: tuple[float, float]) -> float:
    """A value drawn evenly between ``bounds``, to 3 decimals (as a cache records it)."""
    return round(rng.uniform(*bounds), 3)


def _tone(rng: random.Random, tones: tuple[Rgb, ...], brightness: tuple[float, float]) -> Rgb:
    """One of ``tones`` scaled by a factor drawn from ``brightness``, its hue kept."""
    scale = rng.uniform(*brightness)
    red, green, blue = (min(255, round(channel * scale)) for channel in rng.choice(tones))
    return red, green, blue


def _microfilm(rng: random.Random) -> Microfilm:
    return Microfilm(
        crush=_uniform(rng, MICROFILM_CRUSH),
        paper=_uniform(rng, MICROFILM_PAPER),
        gamma=_uniform(rng, MICROFILM_GAMMA),
    )


def _colourise(rng: random.Random) -> Colourise:
    return Colourise(
        crush=_uniform(rng, COLOUR_CRUSH),
        gamma=_uniform(rng, COLOUR_GAMMA),
        paper=_tone(rng, PAPERS, PAPER_BRIGHTNESS),
        ink=_tone(rng, INKS, INK_BRIGHTNESS),
    )


def _tint(rng: random.Random) -> Tint:
    return Tint(
        warmth=_uniform(rng, TINT_WARMTH),
        contrast=_uniform(rng, TINT_CONTRAST),
        brightness=_uniform(rng, TINT_BRIGHTNESS),
        gamma=_uniform(rng, TINT_GAMMA),
    )


DRAWS: tuple[Callable[[random.Random], Restyle], ...] = (_microfilm, _colourise, _tint)


def draw_restyles(count: int, rng: random.Random) -> list[Restyle]:
    """``count`` restyles drawn from ``rng``: of each kind a third (the first kinds one more when ``count`` is not a
    multiple of three), placed by a shuffle, then each one's parameters within the module's ranges."""
    kinds = [DRAWS[k % len(DRAWS)] for k in range(count)]
    rng.shuffle(kinds)
    return [draw(rng) for draw in kinds]
