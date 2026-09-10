"""Model-independent mask and quadrilateral geometry validation gates."""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, List, Literal, Optional, Sequence, Tuple

import cv2
import numpy as np
from pydantic import Field, model_validator

from attendance_scanner.contracts import BaseContract
from attendance_scanner.detector import CanonicalCorners, DetectorPoint

GEOMETRY_VALIDATOR_VERSION: Literal["1.0"] = "1.0"


class GeometryReasonCode(str, Enum):
    """Stable reason codes for mask/quad validation outcomes."""

    MASK_MISSING = "mask_missing"
    MASK_NONFINITE = "mask_nonfinite"
    MASK_DTYPE_UNSUPPORTED = "mask_dtype_unsupported"
    MASK_VALUE_RANGE = "mask_value_range"
    MASK_SHAPE_MISMATCH = "mask_shape_mismatch"
    MASK_EMPTY = "mask_empty"
    MASK_AREA_TOO_SMALL = "mask_area_too_small"
    MASK_AREA_TOO_LARGE = "mask_area_too_large"
    MASK_FRAGMENTED = "mask_fragmented"
    MASK_AMBIGUOUS = "mask_ambiguous"
    MASK_INSUFFICIENT_CONTOUR = "mask_insufficient_contour"
    QUAD_MISSING = "quad_missing"
    QUAD_WRONG_POINT_COUNT = "quad_wrong_point_count"
    QUAD_NONFINITE = "quad_nonfinite"
    QUAD_OUT_OF_BOUNDS = "quad_out_of_bounds"
    QUAD_SELF_INTERSECTION = "quad_self_intersection"
    QUAD_NON_CANONICAL = "quad_non_canonical"
    QUAD_DEGENERATE = "quad_degenerate"
    QUAD_SIDE_TOO_SHORT = "quad_side_too_short"
    QUAD_AREA_OUT_OF_RANGE = "quad_area_out_of_range"
    QUAD_EXTREME_ANGLE = "quad_extreme_angle"
    BORDER_TOUCHING = "border_touching"


class GeometryValidationConfig(BaseContract):
    """Tunable geometry thresholds; no quality threshold is hidden in code."""

    mask_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    mask_min_area_ratio: float = Field(default=0.002, ge=0.0, le=0.5)
    mask_max_area_ratio: float = Field(default=0.995, gt=0.0, le=1.0)
    mask_tiny_area_ratio: float = Field(default=0.01, ge=0.0, le=0.5)
    mask_min_dominance_ratio: float = Field(default=0.6, ge=0.0, le=1.0)
    mask_min_contour_points: int = Field(default=4, ge=3, le=100)
    mask_min_contour_area_ratio: float = Field(default=0.002, ge=0.0, le=0.5)
    quad_bounds_tolerance_px: float = Field(default=2.0, ge=0.0, le=50.0)
    quad_min_side_px: float = Field(default=10.0, ge=0.0)
    quad_min_perimeter_px: float = Field(default=40.0, ge=0.0)
    quad_min_area_ratio: float = Field(default=0.01, ge=0.0, lt=1.0)
    quad_max_area_ratio: float = Field(default=0.995, gt=0.0, le=1.0)
    quad_min_angle_deg: float = Field(default=15.0, ge=0.0, lt=90.0)
    quad_max_angle_deg: float = Field(default=165.0, gt=90.0, le=180.0)
    border_margin_ratio: float = Field(default=0.015, ge=0.0, le=0.2)
    aspect_hint: Optional[float] = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def validate_relationships(self) -> "GeometryValidationConfig":
        if self.mask_min_area_ratio >= self.mask_max_area_ratio:
            raise ValueError("mask_min_area_ratio must be less than mask_max_area_ratio")
        if self.quad_min_area_ratio >= self.quad_max_area_ratio:
            raise ValueError("quad_min_area_ratio must be less than quad_max_area_ratio")
        return self


class MaskValidationEvidence(BaseContract):
    """Typed mask evidence without serializing mask pixels."""

    valid: bool
    reason_codes: List[GeometryReasonCode] = Field(default_factory=list)
    area_ratio: Optional[float] = None
    component_count: int = 0
    dominant_component_ratio: Optional[float] = None
    contour_points: int = 0
    contour_area_ratio: Optional[float] = None
    border_contact: List[str] = Field(default_factory=list)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)


class QuadValidationEvidence(BaseContract):
    """Typed quadrilateral evidence and normalized candidate result."""

    valid: bool
    reason_codes: List[GeometryReasonCode] = Field(default_factory=list)
    normalized_corners: Optional[CanonicalCorners] = None
    area_ratio: Optional[float] = None
    perimeter_px: Optional[float] = None
    minimum_side_px: Optional[float] = None
    angles_deg: List[float] = Field(default_factory=list)
    convex: bool = False
    aspect_ratio: Optional[float] = None
    aspect_hint_score: Optional[float] = None
    border_contact: List[str] = Field(default_factory=list)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)


class CandidateValidationResult(BaseContract):
    """Combined mask/quad validation result used by future fusion stages."""

    validator_version: Literal["1.0"] = GEOMETRY_VALIDATOR_VERSION
    valid: bool
    reason_codes: List[GeometryReasonCode] = Field(default_factory=list)
    mask: MaskValidationEvidence
    quadrilateral: QuadValidationEvidence
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)


def _point_list(value: Any) -> Optional[List[DetectorPoint]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    points: List[DetectorPoint] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        try:
            normalized = (float(point[0]), float(point[1]))
        except (TypeError, ValueError):
            return None
        points.append(normalized)
    return points


def _cross(a: DetectorPoint, b: DetectorPoint, c: DetectorPoint) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _signed_area(points: Sequence[DetectorPoint]) -> float:
    return (
        sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
        / 2.0
    )


def _segment_intersects(
    first_start: DetectorPoint,
    first_end: DetectorPoint,
    second_start: DetectorPoint,
    second_end: DetectorPoint,
) -> bool:
    def orientation(a: DetectorPoint, b: DetectorPoint, c: DetectorPoint) -> int:
        cross = _cross(a, b, c)
        if abs(cross) < 1e-7:
            return 0
        return 1 if cross > 0 else -1

    first = (
        orientation(first_start, first_end, second_start),
        orientation(first_start, first_end, second_end),
    )
    second = (
        orientation(second_start, second_end, first_start),
        orientation(second_start, second_end, first_end),
    )
    return first[0] != first[1] and second[0] != second[1]


def _raw_self_intersects(points: Sequence[DetectorPoint]) -> bool:
    return _segment_intersects(points[0], points[1], points[2], points[3]) or _segment_intersects(
        points[1], points[2], points[3], points[0]
    )


def _order_corners(points: Sequence[DetectorPoint]) -> List[DetectorPoint]:
    sums = [point[0] + point[1] for point in points]
    differences = [point[1] - point[0] for point in points]
    indices = [
        min(range(4), key=lambda index: (sums[index], index)),
        min(range(4), key=lambda index: (differences[index], index)),
        max(range(4), key=lambda index: (sums[index], -index)),
        max(range(4), key=lambda index: (differences[index], -index)),
    ]
    if len(set(indices)) != 4:
        return []
    return [points[index] for index in indices]


def _border_edges(
    points: Sequence[DetectorPoint], width: int, height: int, margin: float
) -> List[str]:
    edges: List[str] = []
    if any(point[0] <= margin for point in points):
        edges.append("left")
    if any(point[1] <= margin for point in points):
        edges.append("top")
    if any(point[0] >= width - 1 - margin for point in points):
        edges.append("right")
    if any(point[1] >= height - 1 - margin for point in points):
        edges.append("bottom")
    return edges


def validate_mask(
    mask: Optional[np.ndarray],
    *,
    expected_shape: Optional[Tuple[int, int]] = None,
    config: Optional[GeometryValidationConfig] = None,
) -> MaskValidationEvidence:
    """Validate a probability/binary mask without modifying or selecting pixels."""
    policy = config or GeometryValidationConfig()
    reasons: List[GeometryReasonCode] = []
    if mask is None:
        return MaskValidationEvidence(valid=False, reason_codes=[GeometryReasonCode.MASK_MISSING])
    array = np.asarray(mask)
    if expected_shape is not None and array.shape != expected_shape:
        reasons.append(GeometryReasonCode.MASK_SHAPE_MISMATCH)
    if array.ndim != 2 or array.size == 0:
        return MaskValidationEvidence(
            valid=False,
            reason_codes=[*reasons, GeometryReasonCode.MASK_EMPTY],
        )
    if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
        return MaskValidationEvidence(
            valid=False,
            reason_codes=[*reasons, GeometryReasonCode.MASK_DTYPE_UNSUPPORTED],
        )
    if not np.isfinite(array).all():
        reasons.append(GeometryReasonCode.MASK_NONFINITE)
    if np.issubdtype(array.dtype, np.floating):
        if float(array.min()) < 0.0 or float(array.max()) > 1.0:
            reasons.append(GeometryReasonCode.MASK_VALUE_RANGE)
        binary = array >= policy.mask_threshold
    else:
        binary = array > 0
    binary = np.asarray(binary, dtype=np.uint8)
    foreground = int(binary.sum())
    if foreground == 0:
        reasons.append(GeometryReasonCode.MASK_EMPTY)
        return MaskValidationEvidence(valid=False, reason_codes=list(dict.fromkeys(reasons)))

    height, width = binary.shape
    area_ratio = foreground / float(binary.size)
    if area_ratio < policy.mask_min_area_ratio:
        reasons.append(GeometryReasonCode.MASK_AREA_TOO_SMALL)
    if area_ratio > policy.mask_max_area_ratio:
        reasons.append(GeometryReasonCode.MASK_AREA_TOO_LARGE)
    binary_for_cv: Any = binary
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_for_cv, connectivity=8
    )
    areas = sorted(
        (int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, component_count)), reverse=True
    )
    dominant_ratio = areas[0] / float(foreground) if areas else 0.0
    if component_count - 1 > 1 and dominant_ratio < policy.mask_min_dominance_ratio:
        reasons.extend((GeometryReasonCode.MASK_FRAGMENTED, GeometryReasonCode.MASK_AMBIGUOUS))

    largest_label = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1) if areas else 0
    largest_mask = np.asarray(labels == largest_label, dtype=np.uint8) if largest_label else binary
    largest_mask_for_cv: Any = largest_mask
    contours, _ = cv2.findContours(largest_mask_for_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea) if contours else None
    contour_points = len(contour) if contour is not None else 0
    contour_area_ratio = (
        float(cv2.contourArea(contour) / binary.size) if contour is not None else 0.0
    )
    if (
        contour is None
        or contour_points < policy.mask_min_contour_points
        or contour_area_ratio < policy.mask_min_contour_area_ratio
    ):
        reasons.append(GeometryReasonCode.MASK_INSUFFICIENT_CONTOUR)
    border_contact: List[str] = []
    if binary[0, :].any():
        border_contact.append("top")
    if binary[-1, :].any():
        border_contact.append("bottom")
    if binary[:, 0].any():
        border_contact.append("left")
    if binary[:, -1].any():
        border_contact.append("right")
    if border_contact:
        reasons.append(GeometryReasonCode.BORDER_TOUCHING)
    area_score = min(1.0, area_ratio / 0.25)
    if area_ratio < policy.mask_tiny_area_ratio:
        area_score *= 0.25
    contour_score = min(1.0, contour_area_ratio / 0.25)
    quality_score = 0.5 * area_score + 0.3 * dominant_ratio + 0.2 * contour_score
    if area_ratio < policy.mask_tiny_area_ratio:
        quality_score *= 0.25
    quality_score = max(0.0, min(1.0, quality_score))
    hard_reasons = {reason for reason in reasons if reason != GeometryReasonCode.BORDER_TOUCHING}
    return MaskValidationEvidence(
        valid=not hard_reasons,
        reason_codes=list(dict.fromkeys(reasons)),
        area_ratio=area_ratio,
        component_count=max(0, component_count - 1),
        dominant_component_ratio=dominant_ratio,
        contour_points=contour_points,
        contour_area_ratio=contour_area_ratio,
        border_contact=border_contact,
        quality_score=quality_score,
    )


def validate_quadrilateral(
    points: Any,
    *,
    image_size: Tuple[int, int],
    config: Optional[GeometryValidationConfig] = None,
) -> QuadValidationEvidence:
    """Validate and canonically normalize a candidate quad without warping it."""
    policy = config or GeometryValidationConfig()
    width, height = image_size
    reasons: List[GeometryReasonCode] = []
    raw = _point_list(points)
    if raw is None:
        return QuadValidationEvidence(
            valid=False,
            reason_codes=[GeometryReasonCode.QUAD_WRONG_POINT_COUNT],
        )
    if any(not math.isfinite(value) for point in raw for value in point):
        reasons.append(GeometryReasonCode.QUAD_NONFINITE)
    if _raw_self_intersects(raw):
        reasons.append(GeometryReasonCode.QUAD_SELF_INTERSECTION)
    if any(
        point[0] < -policy.quad_bounds_tolerance_px
        or point[0] > width - 1 + policy.quad_bounds_tolerance_px
        or point[1] < -policy.quad_bounds_tolerance_px
        or point[1] > height - 1 + policy.quad_bounds_tolerance_px
        for point in raw
    ):
        reasons.append(GeometryReasonCode.QUAD_OUT_OF_BOUNDS)
    normalized = _order_corners(raw)
    canonical: Optional[CanonicalCorners] = None
    if not normalized:
        reasons.append(GeometryReasonCode.QUAD_NON_CANONICAL)
    else:
        try:
            canonical = CanonicalCorners.from_sequence(normalized)
        except ValueError:
            reasons.append(GeometryReasonCode.QUAD_NON_CANONICAL)
    if canonical is None:
        return QuadValidationEvidence(valid=False, reason_codes=list(dict.fromkeys(reasons)))

    canonical_points = canonical.as_list()
    side_lengths = [
        math.hypot(
            canonical_points[(index + 1) % 4][0] - canonical_points[index][0],
            canonical_points[(index + 1) % 4][1] - canonical_points[index][1],
        )
        for index in range(4)
    ]
    perimeter = sum(side_lengths)
    area_ratio = _polygon_area(canonical_points) / float(width * height)
    if min(side_lengths) < policy.quad_min_side_px:
        reasons.append(GeometryReasonCode.QUAD_SIDE_TOO_SHORT)
    if perimeter < policy.quad_min_perimeter_px:
        reasons.append(GeometryReasonCode.QUAD_DEGENERATE)
    if area_ratio < policy.quad_min_area_ratio or area_ratio > policy.quad_max_area_ratio:
        reasons.append(GeometryReasonCode.QUAD_AREA_OUT_OF_RANGE)
    angles: List[float] = []
    for index in range(4):
        previous = np.asarray(canonical_points[(index - 1) % 4]) - np.asarray(
            canonical_points[index]
        )
        following = np.asarray(canonical_points[(index + 1) % 4]) - np.asarray(
            canonical_points[index]
        )
        denominator = float(np.linalg.norm(previous) * np.linalg.norm(following))
        angle = (
            math.degrees(
                math.acos(max(-1.0, min(1.0, float(np.dot(previous, following) / denominator))))
            )
            if denominator > 1e-9
            else 0.0
        )
        angles.append(angle)
    if any(
        angle < policy.quad_min_angle_deg or angle > policy.quad_max_angle_deg for angle in angles
    ):
        reasons.append(GeometryReasonCode.QUAD_EXTREME_ANGLE)
    border_contact = _border_edges(
        canonical_points,
        width,
        height,
        max(width, height) * policy.border_margin_ratio,
    )
    if border_contact:
        reasons.append(GeometryReasonCode.BORDER_TOUCHING)
    aspect_ratio = (side_lengths[0] + side_lengths[2]) / max(
        side_lengths[1] + side_lengths[3], 1e-9
    )
    aspect_hint_score = None
    if policy.aspect_hint is not None:
        aspect_hint_score = max(
            0.0, min(1.0, 1.0 - abs(aspect_ratio - policy.aspect_hint) / policy.aspect_hint)
        )
    hard_reasons = {reason for reason in reasons if reason != GeometryReasonCode.BORDER_TOUCHING}
    quality_score = min(1.0, area_ratio / 0.5)
    if aspect_hint_score is not None:
        quality_score = 0.8 * quality_score + 0.2 * aspect_hint_score
    return QuadValidationEvidence(
        valid=not hard_reasons,
        reason_codes=list(dict.fromkeys(reasons)),
        normalized_corners=canonical,
        area_ratio=area_ratio,
        perimeter_px=perimeter,
        minimum_side_px=min(side_lengths),
        angles_deg=angles,
        convex=True,
        aspect_ratio=aspect_ratio,
        aspect_hint_score=aspect_hint_score,
        border_contact=border_contact,
        quality_score=max(0.0, min(1.0, quality_score)),
    )


def validate_candidate(
    *,
    mask: Optional[np.ndarray],
    corners: Any,
    image_size: Tuple[int, int],
    config: Optional[GeometryValidationConfig] = None,
) -> CandidateValidationResult:
    """Validate a generic segmentation/CV/reference candidate as one gate."""
    width, height = image_size
    mask_result = validate_mask(mask, expected_shape=(height, width), config=config)
    quad_result = validate_quadrilateral(corners, image_size=image_size, config=config)
    reasons = list(dict.fromkeys([*mask_result.reason_codes, *quad_result.reason_codes]))
    valid = mask_result.valid and quad_result.valid
    return CandidateValidationResult(
        valid=valid,
        reason_codes=reasons,
        mask=mask_result,
        quadrilateral=quad_result,
        quality_score=(mask_result.quality_score + quad_result.quality_score) / 2.0,
    )


def _polygon_area(points: Sequence[DetectorPoint]) -> float:
    return abs(_signed_area(points))


__all__ = [
    "CandidateValidationResult",
    "GEOMETRY_VALIDATOR_VERSION",
    "GeometryReasonCode",
    "GeometryValidationConfig",
    "MaskValidationEvidence",
    "QuadValidationEvidence",
    "validate_candidate",
    "validate_mask",
    "validate_quadrilateral",
]
