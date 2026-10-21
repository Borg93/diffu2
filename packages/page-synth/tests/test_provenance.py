"""Provenance: the archive name a caption gives a GT collection folder."""

from __future__ import annotations

from page_synth.captions.provenance import collection_title


def test_collection_title_names_a_folder_whose_trailing_space_is_sfm_encoded() -> None:
    assert collection_title("GT-RA_arjeplogs_tingslag_20260220") == "the court minutes of Arjeplogs tingslag"
