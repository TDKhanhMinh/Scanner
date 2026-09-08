"""AS-24 template-aware page classification and ordering tests."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from attendance_scanner.contracts import PageIdentity, PageType, SourcePage
from attendance_scanner.page_classification import (
    PageClassificationConfig,
    classify_page,
    order_source_pages,
)
from attendance_scanner.pipeline.load import load_image


def _write_day_grid(path: Path, columns: int, *, skew: bool = False) -> None:
    image = np.full((600, 800, 3), 220, dtype=np.uint8)
    paper = np.array([[100, 70], [700, 70], [700, 540], [100, 540]], dtype=np.int32)
    cv2.fillConvexPoly(image, paper, (245, 245, 245))
    cv2.polylines(image, [paper], True, (35, 35, 35), 4)
    table_left, table_right = 145, 655
    table_top, table_bottom = 160, 500
    cv2.rectangle(image, (table_left, table_top), (table_right, table_bottom), (40, 40, 40), 3)
    for index in range(1, columns):
        x = round(table_left + (table_right - table_left) * index / columns)
        cv2.line(image, (x, table_top), (x, table_bottom), (40, 40, 40), 2)
    for index in range(1, 5):
        y = round(table_top + (table_bottom - table_top) * index / 5)
        cv2.line(image, (table_left, y), (table_right, y), (40, 40, 40), 2)

    if skew:
        source = np.float32([[0, 0], [799, 0], [799, 599], [0, 599]])
        target = np.float32([[35, 10], [770, 35], [790, 570], [10, 590]])
        matrix = cv2.getPerspectiveTransform(source, target)
        image = cv2.warpPerspective(image, matrix, (800, 600), borderValue=(220, 220, 220))

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(path)


def _write_day_grid_rows(path: Path, rows: int) -> None:
    """Create a grid whose page identity is carried by horizontal day rows."""
    image = np.full((800, 600, 3), 220, dtype=np.uint8)
    paper = np.array([[50, 40], [550, 40], [550, 760], [50, 760]], dtype=np.int32)
    cv2.fillConvexPoly(image, paper, (245, 245, 245))
    table_left, table_right = 100, 500
    table_top, table_bottom = 100, 700
    cv2.rectangle(image, (table_left, table_top), (table_right, table_bottom), (40, 40, 40), 3)
    for index in range(1, 5):
        x = round(table_left + (table_right - table_left) * index / 5)
        cv2.line(image, (x, table_top), (x, table_bottom), (40, 40, 40), 2)
    for index in range(1, rows):
        y = round(table_top + (table_bottom - table_top) * index / rows)
        cv2.line(image, (table_left, y), (table_right, y), (40, 40, 40), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(path)


@pytest.mark.parametrize(
    ("columns", "expected_type", "expected_order"),
    [
        (21, PageType.FIRST_HALF, 1),
        (10, PageType.SECOND_HALF, 2),
    ],
)
def test_classifies_first_and_second_half_day_grids(
    tmp_path: Path,
    columns: int,
    expected_type: PageType,
    expected_order: int,
):
    path = tmp_path / f"random-page-{columns}.png"
    _write_day_grid(path, columns)

    identity = classify_page(load_image(path))

    assert identity.page_type == expected_type
    assert identity.page_order == expected_order
    assert identity.confidence is not None and identity.confidence >= 0.72
    assert identity.detection_method == "day_grid_layout"
    assert identity.diagnostics["detectedColumnCount"] == columns


def test_perspective_skew_is_classified_after_alignment_like_input(tmp_path: Path):
    path = tmp_path / "hash-7f3a-page.png"
    _write_day_grid(path, 10, skew=True)

    identity = classify_page(load_image(path))

    assert identity.page_type == PageType.SECOND_HALF
    assert identity.page_order == 2


@pytest.mark.parametrize(
    ("rows", "expected_type", "expected_order"),
    [
        (21, PageType.FIRST_HALF, 1),
        (10, PageType.SECOND_HALF, 2),
    ],
)
def test_classifies_page_by_horizontal_day_rows(
    tmp_path: Path,
    rows: int,
    expected_type: PageType,
    expected_order: int,
):
    path = tmp_path / f"rows-{rows}.png"
    _write_day_grid_rows(path, rows)

    identity = classify_page(load_image(path))

    assert identity.page_type == expected_type
    assert identity.page_order == expected_order
    assert identity.diagnostics["detectedAxis"] == "horizontal"


def test_unknown_template_returns_unknown_without_fake_order(tmp_path: Path):
    path = tmp_path / "random-unknown.png"
    Image.new("RGB", (800, 600), color=(128, 128, 128)).save(path)

    identity = classify_page(load_image(path), config=PageClassificationConfig())

    assert identity.page_type == PageType.UNKNOWN
    assert identity.page_order is None
    assert identity.diagnostics["reason"] == "low_confidence_or_unsupported_layout"


def test_page_two_before_page_one_is_ordered_by_identity_not_filename():
    page_two = SourcePage(
        source_relative_path="NV01/aaa-random.png",
        identity=PageIdentity(page_type=PageType.SECOND_HALF, page_order=2),
    )
    page_one = SourcePage(
        source_relative_path="NV01/zzz-random.png",
        identity=PageIdentity(page_type=PageType.FIRST_HALF, page_order=1),
    )

    decision = order_source_pages([page_two, page_one])

    assert decision.review_required is False
    assert [page.identity.page_order for page in decision.ordered] == [1, 2]
    assert [page.source_relative_path for page in decision.ordered] == [
        "NV01/zzz-random.png",
        "NV01/aaa-random.png",
    ]


def test_duplicate_or_unknown_page_identity_requires_review_without_guessing():
    duplicate_first = SourcePage(
        source_relative_path="NV01/first-random.png",
        identity=PageIdentity(page_type=PageType.FIRST_HALF, page_order=1),
    )
    duplicate_second = SourcePage(
        source_relative_path="NV01/second-random.png",
        identity=PageIdentity(page_type=PageType.FIRST_HALF, page_order=1),
    )
    unknown = SourcePage(
        source_relative_path="NV01/unknown-random.png",
        identity=PageIdentity(page_type=PageType.UNKNOWN),
    )

    duplicate_decision = order_source_pages([duplicate_first, duplicate_second])
    unknown_decision = order_source_pages([duplicate_first, unknown])

    assert duplicate_decision.review_required is True
    assert duplicate_decision.reason == "duplicate_or_missing_page_order"
    assert duplicate_decision.ordered == [duplicate_first, duplicate_second]
    assert unknown_decision.review_required is True
    assert unknown_decision.reason == "unknown_page_identity"
    assert unknown_decision.ordered == [duplicate_first, unknown]
