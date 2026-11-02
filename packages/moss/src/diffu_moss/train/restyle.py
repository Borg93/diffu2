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

A hand change (:class:`HandRestyle`, ``--hand-restyle-share``) is the same for the hand: one geometric change, its
parameters drawn once, applied alike to the target and its sample, so each volume's hand comes slanted, weighted and
spaced otherwise and the hand too can only come from the sample: the ink's strokes thinned by up to half a pixel or
thickened by up to one on each side (each image's ink coverage, read from its ink and paper as above, its paper's grain
left out, dilated or eroded by that fraction of a pixel and laid on in its strokes' own colour), the page sheared
across about its middle row by up to ten degrees, and scaled across by 0.85 to 1.15 and fitted back onto the latent
grid at its pixel count. The target's Layout prompt moves with it
(:meth:`HandRestyle.moved_prompt`): each box the bounding box of where its ink goes, on the changed page's canvas; a
change that would move a line off its page or onto another line is refused (:class:`HandRefused`).
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Sequence
from itertools import combinations
from typing import Annotated, Literal

import torch
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor
from torch.nn import functional as F

from diffu_moss.latents import PIXELS_PER_LATENT

type Rgb = tuple[int, int, int]  # 0-255
type CentredBox = list[float]  # y0, x0, y1, x1: px from a canvas's centre, as a Layout prompt's items give them

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

SLANTS = (-10.0, 10.0)  # degrees the letters lean further right (left below 0): the page sheared by its tangent
# px the ink's strokes grow by on each side (shrink, below 0). A 1 MP page's pen strokes are 1-2 px wide: thinned by
# a whole px or two they broke up or were gone, thickened by two its letters filled in
STROKES = (-0.5, 1.0)
WIDTHS = (0.85, 1.15)  # the page scaled across by it, then fitted back to its pixel count: the letters' width
INK_DEAD_ZONE = 0.25  # of the range from paper to ink: a pixel covered less is the paper's grain, never stroke
INK_CORE = 0.6  # of the range above the dead zone: a pixel covered more is a stroke's core, its colour the ink's
OFF_PAGE = "a line off its page"
OVERLAP = "lines onto each other"


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


class HandRefused(ValueError):
    """A hand change refused for a page: it would move a line off the page, or onto another line (``reason``)."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"the hand change would move {reason}")


def _median_colour(rgb: Tensor, pixels: Tensor) -> Tensor:
    """The median of each channel ``[3, 1, 1]`` over the pixels ``pixels`` (a ``[H * W]`` mask) of one image
    ``[3, H, W]``."""
    return rgb.flatten(1)[:, pixels].median(dim=1).values.view(3, 1, 1)


def _dilated(values: Tensor) -> Tensor:
    """``values`` ``[B, 1, H, W]``, each the largest of its 3 x 3 neighbourhood, the edges repeated."""
    height, width = values.shape[-2:]
    padded = F.pad(values, (1, 1, 1, 1), mode="replicate")
    largest = values
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            largest = largest.maximum(padded[..., dy : dy + height, dx : dx + width])
    return largest


def _stroked(rgb: Tensor, stroke: float) -> Tensor:
    """Images ``[B, 3, H, W]`` in [0, 1] with their ink's strokes ``stroke`` px wider on each side (narrower below 0,
    at most one px either way). Each image's ink coverage (by :func:`_levels`, its part above
    :data:`INK_DEAD_ZONE`, so the paper's own grain is no ink) takes ``stroke`` of the step to its 3 x 3 dilation (or
    erosion), and each pixel takes the coverage it gains in the colour of the strokes' cores (the median of the pixels
    covered beyond :data:`INK_CORE`), or gives up what it loses towards the paper's colour (the median of the pixels
    in the dead zone) and no further: the paper beyond the strokes keeps every pixel and its grain, and a stroke its
    colour."""
    if stroke == 0:
        return rgb
    changed = []
    for image in rgb:  # each image's own ink and paper
        coverage = (1 - _levels(_luma(image[None]), 0.0, 1.0)).clamp(0, 1)[0]
        ink_cover = ((coverage - INK_DEAD_ZONE) / (1 - INK_DEAD_ZONE)).clamp(0, 1)
        cores, grain = (ink_cover >= INK_CORE).flatten(), (coverage < INK_DEAD_ZONE).flatten()
        if not (cores.any() and grain.any()):  # no stroke, or no paper, to change between
            changed.append(image)
            continue
        ink, paper = _median_colour(image, cores), _median_colour(image, grain)
        if stroke > 0:
            gained = stroke * (_dilated(ink_cover[None])[0] - ink_cover)
            changed.append(image + gained * (1 - INK_DEAD_ZONE) * (ink - paper))
        else:
            lost = -stroke * (ink_cover + _dilated(-ink_cover[None])[0])  # its coverage minus its 3 x 3 erosion's
            changed.append((image + lost * (1 - INK_DEAD_ZONE) * (paper - ink)).minimum(image.maximum(paper)))
    return torch.stack(changed).clamp(0, 1)


def _sheared(rgb: Tensor, shear: float) -> Tensor:
    """Images ``[B, 3, H, W]`` in [0, 1] sheared across about their middle row: a row ``d`` px above it moved
    ``shear * d`` px right (below it, left), sampled bicubically; the edges it uncovers take the image mirrored there."""
    batch, _, height, width = rgb.shape
    # each drawn pixel's place in the image read, both in [-1, 1] of the image (affine_grid's coordinates)
    theta = rgb.new_tensor([[1.0, shear * height / width, 0.0], [0.0, 1.0, 0.0]]).expand(batch, 2, 3)
    grid = F.affine_grid(theta, [batch, 3, height, width], align_corners=False)
    return F.grid_sample(rgb, grid, mode="bicubic", padding_mode="reflection", align_corners=False).clamp(0, 1)


def _on_grid(side: float) -> int:
    """A side, rounded onto the latent grid (at least one cell)."""
    return max(PIXELS_PER_LATENT, round(side / PIXELS_PER_LATENT) * PIXELS_PER_LATENT)


def _edge(value: float, rounding: Callable[[float], int]) -> int:
    """A box's edge as whole pixels, floored or ceiled as the Layout prompt's compiler rounds them (a float's last
    digits dropped first, so 3.0000000004 is 3)."""
    return rounding(round(value, 9))


def _overlap(a: Sequence[float], b: Sequence[float]) -> bool:
    """Whether two boxes (``y0, x0, y1, x1``) share some area."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


class HandRestyle(BaseModel):
    """A change of hand: the ink's strokes ``stroke`` px wider on each side (narrower below 0; :func:`_stroked`), the
    page sheared across about its middle row so that its letters lean ``slant`` degrees further right (left below 0),
    then scaled across by ``width`` and fitted back onto the latent grid at its pixel count (:meth:`size`): wider or
    narrower letters at the page's own writing scale."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["hand"] = "hand"
    slant: float = Field(gt=-45, lt=45)
    stroke: float = Field(ge=-1, le=1)
    width: float = Field(gt=0)

    @property
    def shear(self) -> float:
        """How many px across a row moves per px above the page's middle row: the slant's tangent."""
        return math.tan(math.radians(self.slant))

    def size(self, size: tuple[int, int]) -> tuple[int, int]:
        """The size (width, height) px an image of ``size`` comes out at: scaled across by ``width``, then by the same
        factor on both sides back to its pixel count, each side rounded onto the latent grid."""
        width, height = size
        root = math.sqrt(self.width)
        return _on_grid(width * root), _on_grid(height / root)

    def apply(self, image: Image.Image) -> Image.Image:
        """``image`` in this hand: an RGB image of :meth:`size`, 8 bits a channel, as a drawing is given a sample."""
        rgb = VaeImageProcessor.numpy_to_pt(VaeImageProcessor.pil_to_numpy(image.convert("RGB")))
        changed = _stroked(rgb, self.stroke)
        if self.slant:  # a slant of 0 resamples nothing
            changed = _sheared(changed, self.shear)
        sheared = VaeImageProcessor.numpy_to_pil(VaeImageProcessor.pt_to_numpy(changed))[0]
        return sheared.resize(self.size(sheared.size), Image.Resampling.LANCZOS)

    def sheared_box(self, box: Sequence[float]) -> CentredBox:
        """A box (``y0, x0, y1, x1``, px from its canvas's centre) sheared: the bounding box of where the ink inside it
        goes, its rows kept, its sides moved by the shift of its top and of its bottom row."""
        y0, x0, y1, x1 = box
        shifts = (-self.shear * y0, -self.shear * y1)  # a row above the middle (y < 0) moves right as the slant grows
        return [y0, x0 + min(shifts), y1, x1 + max(shifts)]

    def moved_prompt(self, prompt: str, size: tuple[int, int]) -> tuple[str, tuple[int, int]]:
        """``prompt``, a compiled Layout prompt of a page of ``size`` (width, height) px, for the page in this hand,
        and the page's size then (:meth:`size`): each item's box sheared (:meth:`sheared_box`) and scaled with the
        page, its top and left edges floored and its bottom and right ceiled to whole px as the compiler rounds them;
        the paper, which covers the whole canvas, the whole new canvas. Every other word is kept.

        Raises:
            HandRefused: If the shear would move an item's box off the page, or make two lines' boxes overlap that did
                not.
        """
        width, height = size
        changed = self.size(size)
        canvas = [-height / 2, -width / 2, height / 2, width / 2]
        layout = json.loads(prompt)
        items = layout["items"]
        sheared = [self.sheared_box(item["bbox"]) for item in items]
        paper = [item["bbox"] == canvas for item in items]
        if any(
            box[1] < canvas[1] or box[3] > canvas[3] for box, whole in zip(sheared, paper, strict=True) if not whole
        ):
            raise HandRefused(OFF_PAGE)
        lines = [index for index, item in enumerate(items) if item["label"] == "text"]
        for a, b in combinations(lines, 2):
            if _overlap(sheared[a], sheared[b]) and not _overlap(items[a]["bbox"], items[b]["bbox"]):
                raise HandRefused(OVERLAP)
        across, down = changed[0] / width, changed[1] / height
        for item, box, whole in zip(items, sheared, paper, strict=True):
            if whole:
                item["bbox"] = [-changed[1] // 2, -changed[0] // 2, changed[1] // 2, changed[0] // 2]
                continue
            y0, x0, y1, x1 = box
            item["bbox"] = [
                _edge(y0 * down, math.floor),
                _edge(x0 * across, math.floor),
                _edge(y1 * down, math.ceil),
                _edge(x1 * across, math.ceil),
            ]
        return json.dumps(layout, ensure_ascii=False, separators=(",", ":"), allow_nan=False), changed


def draw_hand_restyles(count: int, rng: random.Random) -> list[HandRestyle]:
    """``count`` hand changes drawn from ``rng``, each its slant, its stroke and its width within the module's
    ranges."""
    return [
        HandRestyle(slant=_uniform(rng, SLANTS), stroke=_uniform(rng, STROKES), width=_uniform(rng, WIDTHS))
        for _ in range(count)
    ]
