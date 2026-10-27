"""The print/hand classifier's making: its lines cut from ground truths with their kind and collection, its validation
holding out related collections together, and its weights saved and loaded."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from diffu_page.data.print_hand_set import GroundTruth, Sample, build, read_line_set
from diffu_page.marking import PrintHandClassifier, PrintHandNet
from diffu_page.marking.train import collection_groups, deal_folds


def _ground_truth(root: Path, collection: str, texts: list[str]) -> None:
    """A collection of one 1000 x 400 px page under ``root``, a 40 px line per text, its scan beside its PAGE file."""
    volume = root / collection / "volume"
    (volume / "page").mkdir(parents=True)
    scan = np.full((400, 1000), 225, np.uint8)
    lines = []
    for k, text in enumerate(texts):
        top = 50 + 80 * k
        scan[top + 10 : top + 30, 100:900:6] = 40
        lines.append(
            f'<TextLine id="l{k}"><Coords points="100,{top} 900,{top} 900,{top + 40} 100,{top + 40}"/>'
            f"<TextEquiv><Unicode>{text}</Unicode></TextEquiv></TextLine>"
        )
    cv2.imwrite(str(volume / "p1.png"), scan)
    (volume / "page" / "p1.xml").write_text(
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">'
        f'<Page imageFilename="p1.png" imageWidth="1000" imageHeight="400"><TextRegion id="r">{"".join(lines)}'
        "</TextRegion></Page></PcGts>",
        encoding="utf-8",
    )


def test_print_hand_set_cuts_each_collections_lines_of_three_characters_or_more_with_their_kind(tmp_path: Path) -> None:
    _ground_truth(tmp_path / "ocr", "kubhist2", ["Stockholms Dagblad", "No"])  # "No" is too short to show its type
    _ground_truth(tmp_path / "htr", "GT-RA_court_20260220", ["Anno 1720", "den 5 maj", "ibid"])
    truths = [
        GroundTruth(root=tmp_path / "ocr", printed=True, sample=Sample(pages=1, lines=10)),
        GroundTruth(root=tmp_path / "htr", printed=False, sample=Sample(pages=1, lines=2)),
    ]

    build(truths, tmp_path / "lines.npz", seed=0)

    lines = read_line_set(tmp_path / "lines.npz")
    assert lines.collections == ["kubhist2", "GT-RA_court_20260220"]
    assert lines.collection.tolist() == [0, 1, 1]  # the court records cut to their two lines of a sample
    assert lines.printed.tolist() == [True, False, False]
    assert lines.pages == ["kubhist2/volume/page/p1.xml"] + ["GT-RA_court_20260220/volume/page/p1.xml"] * 2
    assert [image.shape for image in lines.images] == [(64, 768)] * 3  # 1280 px at 64 px high, kept to its first 768


def test_collection_groups_hold_a_series_and_the_exports_of_one_together() -> None:
    collections = [
        "OCRmaskinskrift_Lappfogden",
        "OCRmaskinskrift_TRAINING_VALIDATION_SET",  # pages of the Lappfogden volumes again
        "kubhist2",
        "GT-RA_landskapshandlingar_20260220",
        "GT-RA_landskapshandlingar_20260227",  # a later export of the same series
        "GT-SSA_politikollegiet_20260218",
    ]

    groups = collection_groups(collections)

    assert groups == [
        "OCRmaskinskrift_",
        "OCRmaskinskrift_",
        "kubhist2",
        "GT-RA_landskapshandlingar",
        "GT-RA_landskapshandlingar",
        "GT-SSA_politikollegiet",
    ]


def test_deal_folds_deal_each_kinds_groups_round_the_folds() -> None:
    folds = deal_folds(["a", "b", "c", "d"], [True, True, False, False], folds=2, seed=0)

    assert (sorted(folds[:2]), sorted(folds[2:])) == ([0, 1], [0, 1])


def test_a_print_hand_classifier_loaded_from_saved_weights_reads_lines_as_the_net_it_was_saved_from(
    tmp_path: Path,
) -> None:
    torch.manual_seed(0)
    net = PrintHandNet().eval()
    torch.save(net.state_dict(), tmp_path / "print_hand.pt")
    lines = [[np.random.default_rng(k).integers(0, 256, (64, 300), dtype=np.uint8)] for k in range(3)]

    loaded = PrintHandClassifier.load(tmp_path / "print_hand.pt")

    assert loaded(lines) == PrintHandClassifier(net)(lines)
