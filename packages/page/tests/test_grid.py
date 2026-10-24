"""The reference grid: a reference image is drawn from as training saw it, on the 16 px grid and at most the size the
references were trained at."""

from __future__ import annotations

from PIL import Image

from diffu_page.crops.grid import REF_PIXELS, as_reference


def test_as_reference_squashes_a_small_reference_onto_the_16_px_grid() -> None:
    reference = Image.linear_gradient("L").resize((50, 34)).convert("RGB")  # off the 16 px grid

    gridded = as_reference(reference)

    assert gridded.size == (48, 32)
    assert gridded.tobytes() == reference.resize((48, 32), Image.Resampling.LANCZOS).tobytes()


def test_as_reference_shrinks_a_full_crop_to_the_size_references_were_trained_at() -> None:
    crop = Image.linear_gradient("L").resize((1472, 640)).convert("RGB")  # a 0.94 MP training-bucket crop

    reference = as_reference(crop)

    assert reference.size == (1088, 464)
    assert reference.width * reference.height <= REF_PIXELS


def test_as_reference_leaves_a_reference_sized_image_unchanged() -> None:
    reference = as_reference(Image.linear_gradient("L").resize((1472, 640)).convert("RGB"))

    again = as_reference(reference)

    assert (again.size, again.tobytes()) == (reference.size, reference.tobytes())
