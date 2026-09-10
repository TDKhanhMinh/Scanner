"""Safe corner intersections and quality-gated refinement decisions."""

from __future__ import annotations

import math
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from pydantic import Field

from attendance_scanner.contracts import BaseContract
from attendance_scanner.detector import CanonicalCorners, DetectorPoint
from attendance_scanner.geometry_validator import GeometryValidationConfig, validate_quadrilateral
from attendance_scanner.line_fitting import FittedDocumentLine, FittedLineSet

CORNER_REFINEMENT_VERSION = "1.0"


class RefinementReasonCode(str, Enum):
    """Stable refinement decision reasons."""

    ACCEPTED = "accepted"
    PARTIAL = "partial"
    NO_USABLE_LINES = "no_usable_lines"
    NEAR_PARALLEL_LINES = "near_parallel_lines"
    NONFINITE_INTERSECTION = "nonfinite_intersection"
    DISPLACEMENT_TOO_LARGE = "displacement_too_large"
    GEOMETRY_INVALID = "geometry_invalid"
    EDGE_SUPPORT_DROP = "edge_support_drop"
    PARTIAL_NOT_ALLOWED = "partial_not_allowed"


class CornerRefinementConfig(BaseContract):
    """Bounded displacement, line confidence, and post-refinement quality policy."""

    max_corner_displacement_px: float = Field(default=80.0, gt=0.0, le=1000.0)
    max_corner_displacement_ratio: float = Field(default=0.08, gt=0.0, le=0.5)
    minimum_line_confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    minimum_refined_edges: int = Field(default=2, ge=1, le=4)
    allow_partial_refinement: bool = True
    edge_support_drop_tolerance: float = Field(default=0.05, ge=0.0, le=1.0)
    geometry: GeometryValidationConfig = Field(default_factory=GeometryValidationConfig)


class CornerRefinementResult(BaseContract):
    """Before/after refinement evidence with selected safe corners."""

    refinement_version: str = CORNER_REFINEMENT_VERSION
    accepted: bool
    before_corners: CanonicalCorners
    refined_corners: CanonicalCorners
    selected_corners: CanonicalCorners
    displacement_px: Dict[str, float]
    max_displacement_px: float = Field(ge=0.0)
    refined_edge_labels: List[str] = Field(default_factory=list)
    fallback_edge_labels: List[str] = Field(default_factory=list)
    line_confidence: Dict[str, float] = Field(default_factory=dict)
    before_edge_support: Optional[float] = None
    after_edge_support: Optional[float] = None
    reason_codes: List[RefinementReasonCode] = Field(min_length=1)
    diagnostics: Dict[str, object] = Field(default_factory=dict)


def _line_intersection(
    first: FittedDocumentLine,
    second: FittedDocumentLine,
) -> Optional[DetectorPoint]:
    determinant = first.normal_a * second.normal_b - second.normal_a * first.normal_b
    if abs(determinant) < 1e-8:
        return None
    x = (first.normal_b * second.normal_c - second.normal_b * first.normal_c) / determinant
    y = (second.normal_a * first.normal_c - first.normal_a * second.normal_c) / determinant
    if not np.isfinite((x, y)).all():
        return None
    return float(x), float(y)


def _original_edge_line(
    label: str,
    start: DetectorPoint,
    end: DetectorPoint,
) -> FittedDocumentLine:
    direction = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    length = float(np.linalg.norm(direction))
    direction /= max(length, 1e-9)
    normal = np.asarray((-direction[1], direction[0]), dtype=np.float64)
    return FittedDocumentLine(
        label=label,  # type: ignore[arg-type]
        anchor=start,
        direction=(float(direction[0]), float(direction[1])),
        normal_a=float(normal[0]),
        normal_b=float(normal[1]),
        normal_c=float(-np.dot(normal, np.asarray(start))),
        angle_deg=math.degrees(math.atan2(direction[1], direction[0])) % 180.0,
        inlier_count=0,
        sample_count=0,
        inlier_ratio=0.0,
        residual_mean_px=0.0,
        residual_p95_px=0.0,
        support_length_px=length,
        confidence=0.0,
        fallback_used=True,
        source="original_edge",
        reason="partial_refinement_original_edge",
    )


def refine_document_corners(
    before_corners: CanonicalCorners | Sequence[DetectorPoint],
    fitted_lines: FittedLineSet,
    *,
    image_size: Tuple[int, int],
    config: Optional[CornerRefinementConfig] = None,
    before_edge_support: Optional[float] = None,
    after_edge_support: Optional[float] = None,
) -> CornerRefinementResult:
    """Intersect usable fitted lines and accept only a quality-safe refinement."""
    policy = config or CornerRefinementConfig()
    before = (
        before_corners
        if isinstance(before_corners, CanonicalCorners)
        else CanonicalCorners.from_sequence(before_corners)
    )
    if len(fitted_lines.lines) != 4:
        raise ValueError("fitted_lines must contain top/right/bottom/left lines")
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image_size must contain positive dimensions")
    labels = ("top", "right", "bottom", "left")
    original_points = before.as_list()
    original_lines = [
        _original_edge_line(labels[index], original_points[index], original_points[(index + 1) % 4])
        for index in range(4)
    ]
    effective_lines: List[FittedDocumentLine] = []
    refined_edge_labels: List[str] = []
    fallback_edge_labels: List[str] = []
    line_confidence: Dict[str, float] = {}
    for index, line in enumerate(fitted_lines.lines):
        line_confidence[labels[index]] = line.confidence
        usable = (
            line.source != "none"
            and not line.fallback_used
            and line.confidence >= policy.minimum_line_confidence
        )
        if usable:
            effective_lines.append(line)
            refined_edge_labels.append(labels[index])
        else:
            effective_lines.append(original_lines[index])
            fallback_edge_labels.append(labels[index])
    if fallback_edge_labels and not policy.allow_partial_refinement:
        return _rejected_result(
            before,
            config=policy,
            image_size=image_size,
            reason_codes=[RefinementReasonCode.PARTIAL_NOT_ALLOWED],
            refined_edge_labels=refined_edge_labels,
            fallback_edge_labels=fallback_edge_labels,
            line_confidence=line_confidence,
            before_edge_support=before_edge_support,
            after_edge_support=after_edge_support,
        )
    if len(refined_edge_labels) < policy.minimum_refined_edges:
        reason = [RefinementReasonCode.NO_USABLE_LINES]
        return _rejected_result(
            before,
            config=policy,
            image_size=image_size,
            reason_codes=reason,
            refined_edge_labels=refined_edge_labels,
            fallback_edge_labels=fallback_edge_labels,
            line_confidence=line_confidence,
            before_edge_support=before_edge_support,
            after_edge_support=after_edge_support,
        )

    intersection_pairs = ((3, 0), (0, 1), (1, 2), (2, 3))
    refined_points: List[DetectorPoint] = []
    reasons: List[RefinementReasonCode] = []
    for corner_index, (first_index, second_index) in enumerate(intersection_pairs):
        point = _line_intersection(effective_lines[first_index], effective_lines[second_index])
        if point is None:
            refined_points.append(original_points[corner_index])
            reasons.append(RefinementReasonCode.NEAR_PARALLEL_LINES)
        else:
            refined_points.append(point)
    try:
        refined = CanonicalCorners.from_sequence(refined_points)
    except ValueError:
        return _rejected_result(
            before,
            config=policy,
            image_size=image_size,
            reason_codes=[*reasons, RefinementReasonCode.GEOMETRY_INVALID],
            refined_edge_labels=refined_edge_labels,
            fallback_edge_labels=fallback_edge_labels,
            line_confidence=line_confidence,
            before_edge_support=before_edge_support,
            after_edge_support=after_edge_support,
        )
    displacement = {
        label: math.dist(original_points[index], refined_points[index])
        for index, label in enumerate(("tl", "tr", "br", "bl"))
    }
    max_displacement = min(
        policy.max_corner_displacement_px,
        policy.max_corner_displacement_ratio * math.hypot(width, height),
    )
    if max(displacement.values(), default=0.0) > max_displacement:
        reasons.append(RefinementReasonCode.DISPLACEMENT_TOO_LARGE)
    geometry = validate_quadrilateral(
        refined.as_list(), image_size=image_size, config=policy.geometry
    )
    if not geometry.valid:
        reasons.append(RefinementReasonCode.GEOMETRY_INVALID)
    if (
        before_edge_support is not None
        and after_edge_support is not None
        and after_edge_support + policy.edge_support_drop_tolerance < before_edge_support
    ):
        reasons.append(RefinementReasonCode.EDGE_SUPPORT_DROP)
    if reasons:
        return _rejected_result(
            before,
            config=policy,
            image_size=image_size,
            reason_codes=list(dict.fromkeys(reasons)),
            refined=refined,
            displacement=displacement,
            refined_edge_labels=refined_edge_labels,
            fallback_edge_labels=fallback_edge_labels,
            line_confidence=line_confidence,
            before_edge_support=before_edge_support,
            after_edge_support=after_edge_support,
        )
    reason_codes = [
        RefinementReasonCode.PARTIAL if fallback_edge_labels else RefinementReasonCode.ACCEPTED
    ]
    return CornerRefinementResult(
        accepted=True,
        before_corners=before,
        refined_corners=refined,
        selected_corners=refined,
        displacement_px=displacement,
        max_displacement_px=max(displacement.values(), default=0.0),
        refined_edge_labels=refined_edge_labels,
        fallback_edge_labels=fallback_edge_labels,
        line_confidence=line_confidence,
        before_edge_support=before_edge_support,
        after_edge_support=after_edge_support,
        reason_codes=reason_codes,
        diagnostics={
            "maxAllowedDisplacementPx": max_displacement,
            "beforeCorners": before.model_dump(mode="json"),
            "afterCorners": refined.model_dump(mode="json"),
        },
    )


def _rejected_result(
    before: CanonicalCorners,
    *,
    config: CornerRefinementConfig,
    image_size: Tuple[int, int],
    reason_codes: List[RefinementReasonCode],
    refined: Optional[CanonicalCorners] = None,
    displacement: Optional[Dict[str, float]] = None,
    refined_edge_labels: Optional[List[str]] = None,
    fallback_edge_labels: Optional[List[str]] = None,
    line_confidence: Optional[Dict[str, float]] = None,
    before_edge_support: Optional[float] = None,
    after_edge_support: Optional[float] = None,
) -> CornerRefinementResult:
    candidate = refined or before
    actual_displacement = displacement or dict.fromkeys(("tl", "tr", "br", "bl"), 0.0)
    return CornerRefinementResult(
        accepted=False,
        before_corners=before,
        refined_corners=candidate,
        selected_corners=before,
        displacement_px=actual_displacement,
        max_displacement_px=max(actual_displacement.values(), default=0.0),
        refined_edge_labels=refined_edge_labels or [],
        fallback_edge_labels=fallback_edge_labels or [],
        line_confidence=line_confidence or {},
        before_edge_support=before_edge_support,
        after_edge_support=after_edge_support,
        reason_codes=list(dict.fromkeys(reason_codes)),
        diagnostics={
            "maxAllowedDisplacementPx": min(
                config.max_corner_displacement_px,
                config.max_corner_displacement_ratio * math.hypot(*image_size),
            ),
            "beforeCorners": before.model_dump(mode="json"),
            "afterCorners": candidate.model_dump(mode="json"),
        },
    )


__all__ = [
    "CORNER_REFINEMENT_VERSION",
    "CornerRefinementConfig",
    "CornerRefinementResult",
    "RefinementReasonCode",
    "refine_document_corners",
]
