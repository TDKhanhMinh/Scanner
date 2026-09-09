"""Template-aware page classification and review-safe page ordering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from pydantic import Field

from attendance_scanner.contracts import (
    BaseContract,
    PageIdentity,
    PageType,
    SourcePage,
)


class PageClassificationConfig(BaseContract):
    """Thresholds for the current attendance form day-grid layout."""

    first_half_columns: int = Field(default=21, ge=2, le=64)
    second_half_columns: int = Field(default=10, ge=2, le=64)
    min_confidence: float = Field(default=0.72, ge=0.0, le=1.0)
    min_margin: float = Field(default=0.15, ge=0.0, le=1.0)
    min_table_ink_ratio: float = Field(default=0.01, ge=0.0, le=1.0)


@dataclass(frozen=True)
class PageOrderDecision:
    """Ordering result that refuses to guess when page identity is uncertain."""

    ordered: List[SourcePage]
    review_required: bool
    reason: Optional[str] = None


def _as_gray(image: Any) -> np.ndarray:
    """Normalize a processed image-like object to an 8-bit grayscale array."""
    candidate = getattr(image, "image", image)
    array = np.asarray(candidate)
    if array.dtype != np.uint8:
        raise ValueError(f"Page classifier expects uint8 pixels, got {array.dtype}")
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[2] in (3, 4):
        code = cv2.COLOR_BGR2GRAY if array.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
        return cv2.cvtColor(array, code)
    raise ValueError(f"Page classifier expects a grayscale or BGR image, got {array.shape}")


def _group_columns(columns: np.ndarray, *, merge_gap: int = 6) -> List[int]:
    """Collapse contiguous dark-column runs into stable vertical grid positions."""
    if columns.size == 0:
        return []
    groups: List[List[int]] = []
    for column in columns.tolist():
        if not groups or column > groups[-1][-1] + merge_gap:
            groups.append([column])
        else:
            groups[-1].append(column)
    return [int(round(sum(group) / len(group))) for group in groups]


def _day_grid_features(
    gray: np.ndarray,
    config: PageClassificationConfig,
) -> Tuple[int, float, float, Dict[str, Any]]:
    """Estimate the strongest day-grid axis and layout confidence."""
    height, width = gray.shape
    x_start = max(0, int(width * 0.15))
    x_end = min(width, int(width * 0.85))
    y_start = max(0, int(height * 0.18))
    y_end = min(height, int(height * 0.90))
    roi = gray[y_start:y_end, x_start:x_end]
    if roi.size == 0:
        return 0, 0.0, 0.0, {"detectedAxis": "unknown"}

    # Long vertical rules survive Canny/Hough even when paper/background contrast
    # changes. Text and shadows rarely produce similarly long vertical segments.
    edges = cv2.Canny(roi, 20, 100)
    min_line_length = max(20, int(roi.shape[0] * 0.22))
    threshold = max(18, int(roi.shape[0] * 0.06))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=12,
    )
    vertical_positions: List[int] = []
    horizontal_positions: List[int] = []
    if lines is not None:
        for line in np.asarray(lines).reshape(-1, 4):
            x1, y1, x2, y2 = [int(value) for value in line]
            if abs(y2 - y1) >= abs(x2 - x1) * 2:
                vertical_positions.append(int(round((x1 + x2) / 2)))
            elif abs(x2 - x1) >= abs(y2 - y1) * 2:
                horizontal_positions.append(int(round((y1 + y2) / 2)))

    vertical_count = max(
        0,
        len(_group_columns(np.array(sorted(vertical_positions), dtype=np.int32))) - 1,
    )
    horizontal_count = max(
        0,
        len(_group_columns(np.array(sorted(horizontal_positions), dtype=np.int32))) - 1,
    )
    vertical_score = max(
        _score_column_count(vertical_count, config.first_half_columns),
        _score_column_count(vertical_count, config.second_half_columns),
    )
    horizontal_score = max(
        _score_column_count(horizontal_count, config.first_half_columns),
        _score_column_count(horizontal_count, config.second_half_columns),
    )
    detected_axis = "vertical" if vertical_score >= horizontal_score else "horizontal"
    detected_count = vertical_count if detected_axis == "vertical" else horizontal_count
    return (
        detected_count,
        float((edges > 0).mean()),
        float(threshold),
        {
            "detectedAxis": detected_axis,
            "verticalGridCount": vertical_count,
            "horizontalGridCount": horizontal_count,
        },
    )


def _score_column_count(column_count: int, expected: int) -> float:
    """Return a bounded similarity score for a detected day-column count."""
    if column_count <= 0:
        return 0.0
    distance = abs(column_count - expected)
    return max(0.0, 1.0 - (distance / max(expected, 1)) * 1.5)


def classify_page(
    image: Any,
    *,
    config: Optional[PageClassificationConfig] = None,
) -> PageIdentity:
    """Classify a processed/aligned attendance page without using its filename.

    The current template exposes two stable layout signatures: a 21-row
    first-half day grid and a 10-row second-half day grid. Any weak, low-margin,
    or unsupported layout returns UNKNOWN so grouped export can require review.
    """
    cfg = config or PageClassificationConfig()
    gray = _as_gray(image)
    detected_columns, ink_ratio, score_threshold, axis_diagnostics = _day_grid_features(gray, cfg)
    first_score = _score_column_count(detected_columns, cfg.first_half_columns)
    second_score = _score_column_count(detected_columns, cfg.second_half_columns)
    scores = {
        PageType.FIRST_HALF: first_score,
        PageType.SECOND_HALF: second_score,
    }
    page_type = max(scores, key=lambda candidate: scores[candidate])
    confidence = scores[page_type]
    margin = abs(first_score - second_score)
    diagnostics: Dict[str, Any] = {
        **axis_diagnostics,
        "detectedColumnCount": detected_columns,
        "firstHalfScore": round(first_score, 4),
        "secondHalfScore": round(second_score, 4),
        "scoreMargin": round(margin, 4),
        "tableInkRatio": round(ink_ratio, 4),
        "columnThreshold": round(score_threshold, 4),
    }

    if (
        ink_ratio < cfg.min_table_ink_ratio
        or confidence < cfg.min_confidence
        or margin < cfg.min_margin
    ):
        diagnostics["reason"] = "low_confidence_or_unsupported_layout"
        return PageIdentity(
            page_type=PageType.UNKNOWN,
            page_order=None,
            confidence=confidence,
            detection_method="day_grid_layout",
            diagnostics=diagnostics,
        )

    return PageIdentity(
        page_type=page_type,
        page_order=1 if page_type == PageType.FIRST_HALF else 2,
        confidence=confidence,
        detection_method="day_grid_layout",
        diagnostics=diagnostics,
    )


def order_source_pages(pages: Sequence[SourcePage]) -> PageOrderDecision:
    """Order pages by explicit identity or preserve input for human review.

    This function never falls back to filename, mtime, or EXIF ordering. Duplicate
    or unknown page identities remain in their observed order and require review.
    """
    source_pages = list(pages)
    if not source_pages:
        return PageOrderDecision(ordered=[], review_required=False)

    page_orders = [
        page.identity.page_order
        for page in source_pages
        if page.identity.page_type != PageType.UNKNOWN
    ]
    if len(page_orders) != len(source_pages):
        return PageOrderDecision(
            ordered=source_pages,
            review_required=True,
            reason="unknown_page_identity",
        )
    if any(order is None for order in page_orders) or len(set(page_orders)) != len(page_orders):
        return PageOrderDecision(
            ordered=source_pages,
            review_required=True,
            reason="duplicate_or_missing_page_order",
        )

    return PageOrderDecision(
        ordered=sorted(source_pages, key=lambda page: page.identity.page_order or 0),
        review_required=False,
    )


def classify_document_page(
    image: Any,
    *,
    config: Optional[PageClassificationConfig] = None,
) -> PageIdentity:
    """Backward-compatible descriptive alias for ``classify_page``."""
    return classify_page(image, config=config)
