"""Robust document edge-line fitting from local samples and Hough evidence."""

from __future__ import annotations

import math
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from pydantic import Field, model_validator

from attendance_scanner.contracts import BaseContract
from attendance_scanner.corner_search import CornerSearchResult
from attendance_scanner.detector import DetectorPoint
from attendance_scanner.hough_lines import HoughLineEvidence

LINE_FITTING_VERSION = "1.0"
EdgeLabel = Literal["top", "right", "bottom", "left"]


class LineFittingConfig(BaseContract):
    """Robust fitting and fallback thresholds."""

    max_residual_px: float = Field(default=3.0, gt=0.0, le=50.0)
    direction_tolerance_deg: float = Field(default=20.0, gt=0.0, le=90.0)
    minimum_inliers: int = Field(default=4, ge=2, le=100)
    minimum_inlier_ratio: float = Field(default=0.35, ge=0.0, le=1.0)
    minimum_support_length_ratio: float = Field(default=0.2, ge=0.0, le=1.0)
    hough_distance_px: float = Field(default=40.0, ge=0.0, le=500.0)
    use_hough_fallback: bool = True
    use_original_edge_fallback: bool = True

    @model_validator(mode="after")
    def validate_thresholds(self) -> "LineFittingConfig":
        if self.minimum_inliers < 2:
            raise ValueError("minimum_inliers must be at least 2")
        return self


class FittedDocumentLine(BaseContract):
    """Finite-supported line model and confidence evidence for one document edge."""

    label: EdgeLabel
    anchor: DetectorPoint
    direction: DetectorPoint
    normal_a: float
    normal_b: float
    normal_c: float
    angle_deg: float = Field(ge=0.0, lt=180.0)
    inlier_count: int = Field(ge=0)
    sample_count: int = Field(ge=0)
    inlier_ratio: float = Field(ge=0.0, le=1.0)
    residual_mean_px: float = Field(ge=0.0)
    residual_p95_px: float = Field(ge=0.0)
    support_length_px: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    fallback_used: bool = False
    source: Literal["local", "local+hough", "original_edge", "none"]
    reason: Optional[str] = None


class FittedLineSet(BaseContract):
    """Four finite-supported edge lines; intersections are intentionally deferred."""

    fitting_version: str = LINE_FITTING_VERSION
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    lines: List[FittedDocumentLine] = Field(min_length=4, max_length=4)
    diagnostics: Dict[str, object] = Field(default_factory=dict)
    config: LineFittingConfig


def _angle(point_a: DetectorPoint, point_b: DetectorPoint) -> float:
    return math.degrees(math.atan2(point_b[1] - point_a[1], point_b[0] - point_a[0])) % 180.0


def _angle_distance(first: float, second: float) -> float:
    return min(abs(first - second), 180.0 - abs(first - second))


def _line_residual(points: np.ndarray, anchor: np.ndarray, direction: np.ndarray) -> np.ndarray:
    offsets = points - anchor
    return np.abs(offsets[:, 0] * direction[1] - offsets[:, 1] * direction[0])


def _fit_direction(points: np.ndarray) -> np.ndarray:
    centered = points - np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    direction = direction / max(float(np.linalg.norm(direction)), 1e-9)
    if direction[0] < 0.0 or (abs(direction[0]) < 1e-9 and direction[1] < 0.0):
        direction = -direction
    return direction


def _line_coefficients(anchor: np.ndarray, direction: np.ndarray) -> Tuple[float, float, float]:
    normal = np.asarray((-direction[1], direction[0]), dtype=np.float64)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-9)
    return float(normal[0]), float(normal[1]), float(-np.dot(normal, anchor))


def _samples_for_edge(
    label_index: int,
    corners: Sequence[DetectorPoint],
    local_search: Optional[CornerSearchResult],
) -> List[DetectorPoint]:
    if local_search is None:
        return []
    samples: List[DetectorPoint] = []
    next_corner = local_search.corners[label_index]
    previous_corner = local_search.corners[(label_index + 1) % 4]
    samples.extend(
        sample.point
        for sample in next_corner.samples
        if sample.side == "next" and sample.edge_detected
    )
    samples.extend(
        sample.point
        for sample in previous_corner.samples
        if sample.side == "previous" and sample.edge_detected
    )
    return samples


def _hough_points_for_edge(
    label_index: int,
    corners: Sequence[DetectorPoint],
    hough: Optional[HoughLineEvidence],
    distance_px: float,
    direction_tolerance_deg: float,
) -> List[DetectorPoint]:
    if hough is None:
        return []
    start = corners[label_index]
    end = corners[(label_index + 1) % 4]
    expected_angle = _angle(start, end)
    midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
    points: List[DetectorPoint] = []
    for segment in hough.segments:
        if _angle_distance(segment.angle_deg, expected_angle) > direction_tolerance_deg:
            continue
        segment_midpoint = (
            (segment.p1[0] + segment.p2[0]) / 2.0,
            (segment.p1[1] + segment.p2[1]) / 2.0,
        )
        if math.dist(midpoint, segment_midpoint) <= distance_px:
            points.extend((segment.p1, segment.p2))
    return points


def _fallback_line(
    label: EdgeLabel,
    start: DetectorPoint,
    end: DetectorPoint,
    *,
    reason: str,
) -> FittedDocumentLine:
    direction_array = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    length = float(np.linalg.norm(direction_array))
    direction_array = direction_array / max(length, 1e-9)
    anchor = np.asarray(start, dtype=np.float64)
    a, b, c = _line_coefficients(anchor, direction_array)
    return FittedDocumentLine(
        label=label,
        anchor=(float(anchor[0]), float(anchor[1])),
        direction=(float(direction_array[0]), float(direction_array[1])),
        normal_a=a,
        normal_b=b,
        normal_c=c,
        angle_deg=_angle(start, end),
        inlier_count=0,
        sample_count=0,
        inlier_ratio=0.0,
        residual_mean_px=0.0,
        residual_p95_px=0.0,
        support_length_px=length,
        confidence=0.2,
        fallback_used=True,
        source="original_edge",
        reason=reason,
    )


def _no_fit_line(
    label: EdgeLabel,
    start: DetectorPoint,
    end: DetectorPoint,
    *,
    reason: str,
) -> FittedDocumentLine:
    """Return typed no-fit evidence without exposing the original edge as fallback."""
    return _fallback_line(label, start, end, reason=reason).model_copy(
        update={"source": "none", "fallback_used": False, "confidence": 0.0}
    )


def _fit_one_edge(
    label: EdgeLabel,
    start: DetectorPoint,
    end: DetectorPoint,
    points: Sequence[DetectorPoint],
    *,
    config: LineFittingConfig,
    source: Literal["local", "local+hough"],
) -> FittedDocumentLine:
    expected_angle = _angle(start, end)
    if len(points) < config.minimum_inliers:
        if config.use_original_edge_fallback:
            return _fallback_line(label, start, end, reason="insufficient_inliers")
        return _no_fit_line(label, start, end, reason="fit_failed")
    array = np.asarray(points, dtype=np.float64)
    if not np.isfinite(array).all():
        return _fallback_line(label, start, end, reason="nonfinite_samples")
    anchor = np.mean(array, axis=0)
    direction = _fit_direction(array)
    if (
        _angle_distance(
            math.degrees(math.atan2(direction[1], direction[0])) % 180.0, expected_angle
        )
        > config.direction_tolerance_deg
    ):
        return _fallback_line(label, start, end, reason="direction_mismatch")
    residuals = _line_residual(array, anchor, direction)
    inliers = residuals <= config.max_residual_px
    if int(inliers.sum()) >= config.minimum_inliers:
        inlier_points = array[inliers]
        anchor = np.mean(inlier_points, axis=0)
        direction = _fit_direction(inlier_points)
        residuals = _line_residual(inlier_points, anchor, direction)
    else:
        inlier_points = array[inliers]
    inlier_count = len(inlier_points)
    inlier_ratio = inlier_count / float(len(array))
    support_length = (
        float(
            np.max(np.dot(inlier_points - anchor, direction))
            - np.min(np.dot(inlier_points - anchor, direction))
        )
        if inlier_count >= 2
        else 0.0
    )
    expected_length = max(math.dist(start, end), 1e-9)
    support_ratio = min(1.0, support_length / expected_length)
    if inlier_count < config.minimum_inliers or (
        inlier_ratio < config.minimum_inlier_ratio
        and support_ratio < config.minimum_support_length_ratio
    ):
        if config.use_original_edge_fallback:
            return _fallback_line(label, start, end, reason="weak_fit")
        return _no_fit_line(label, start, end, reason="weak_fit_no_fallback")
    a, b, c = _line_coefficients(anchor, direction)
    residual_mean = float(np.mean(residuals)) if len(residuals) else 0.0
    residual_p95 = float(np.percentile(residuals, 95)) if len(residuals) else 0.0
    direction_score = max(
        0.0,
        1.0
        - _angle_distance(
            math.degrees(math.atan2(direction[1], direction[0])) % 180.0, expected_angle
        )
        / config.direction_tolerance_deg,
    )
    confidence = max(
        0.0,
        min(
            1.0,
            0.45 * inlier_ratio
            + 0.3 * support_ratio
            + 0.2 * direction_score
            + 0.05 * max(0.0, 1.0 - residual_mean / config.max_residual_px),
        ),
    )
    return FittedDocumentLine(
        label=label,
        anchor=(float(anchor[0]), float(anchor[1])),
        direction=(float(direction[0]), float(direction[1])),
        normal_a=a,
        normal_b=b,
        normal_c=c,
        angle_deg=math.degrees(math.atan2(direction[1], direction[0])) % 180.0,
        inlier_count=inlier_count,
        sample_count=len(array),
        inlier_ratio=inlier_ratio,
        residual_mean_px=residual_mean,
        residual_p95_px=residual_p95,
        support_length_px=support_length,
        confidence=confidence,
        fallback_used=False,
        source=source,
    )


def fit_document_edge_lines(
    corners: Sequence[DetectorPoint],
    *,
    source_size: Tuple[int, int],
    local_search: Optional[CornerSearchResult] = None,
    hough_evidence: Optional[HoughLineEvidence] = None,
    config: Optional[LineFittingConfig] = None,
) -> FittedLineSet:
    """Fit four finite-supported lines and keep original-edge fallback evidence."""
    policy = config or LineFittingConfig()
    if len(corners) != 4:
        raise ValueError("corners must contain exactly four points")
    width, height = source_size
    if width <= 0 or height <= 0:
        raise ValueError("source_size must contain positive dimensions")
    labels: Tuple[EdgeLabel, ...] = ("top", "right", "bottom", "left")
    lines: List[FittedDocumentLine] = []
    for index, label in enumerate(labels):
        points = _samples_for_edge(index, corners, local_search)
        source: Literal["local", "local+hough"] = "local"
        if len(points) < policy.minimum_inliers and policy.use_hough_fallback:
            points.extend(
                _hough_points_for_edge(
                    index,
                    corners,
                    hough_evidence,
                    policy.hough_distance_px,
                    policy.direction_tolerance_deg,
                )
            )
            if points:
                source = "local+hough"
        lines.append(
            _fit_one_edge(
                label,
                corners[index],
                corners[(index + 1) % 4],
                points,
                config=policy,
                source=source,
            )
        )
    return FittedLineSet(
        source_width=width,
        source_height=height,
        lines=lines,
        diagnostics={
            "lineCount": len(lines),
            "fallbackCount": sum(line.fallback_used for line in lines),
            "localSearchProvided": local_search is not None,
            "houghProvided": hough_evidence is not None,
        },
        config=policy,
    )


__all__ = [
    "FittedDocumentLine",
    "FittedLineSet",
    "LINE_FITTING_VERSION",
    "LineFittingConfig",
    "fit_document_edge_lines",
]
