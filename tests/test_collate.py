"""Width-bucketing collate."""

from __future__ import annotations

import torch

from diffu.data.collate import collate_lines


def _item(width: int, text: str = "x", writer: str = "w0") -> dict[str, object]:
    return {
        "image": torch.zeros(3, 64, width) - 1.0,  # all -1 (black) so white pad (1.0) is detectable
        "text": text,
        "writer_id": writer,
        "style_pixel_values": torch.zeros(3, 224, 224),
    }


def test_collate_pads_to_rounded_max_width():
    batch = [_item(100), _item(130)]
    out = collate_lines(batch, width_multiple=16, max_width=1024)
    # max width 130 rounded up to a multiple of 16 -> 144
    assert out["images"].shape == (2, 3, 64, 144)
    assert out["style_pixel_values"].shape == (2, 3, 224, 224)
    assert out["texts"] == ["x", "x"]


def test_collate_right_pads_with_white():
    out = collate_lines([_item(100), _item(130)], width_multiple=16, max_width=1024)
    img0 = out["images"][0]
    assert torch.allclose(img0[:, :, :100], torch.full((3, 64, 100), -1.0))  # original content
    assert torch.allclose(img0[:, :, 100:], torch.ones(3, 64, 44))  # white padding


def test_collate_caps_at_max_width():
    out = collate_lines([_item(2000)], width_multiple=16, max_width=512)
    assert out["images"].shape[-1] == 512  # cropped to the cap
