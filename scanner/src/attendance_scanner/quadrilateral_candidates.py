"""Bounded quadrilateral candidates from masks, contours, Hough lines, and mixtures."""

from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import cv2
import numpy as np
from pydantic import Field, model_validator

from attendance_scanner.contracts import BaseContract
from attendance_scanner.cv_candidates import CvCandidateSet
from attendance_scanner.detector import CanonicalCorners, DetectorPoint
from attendance_scanner.geometry_validator import (
    GeometryReasonCode,
    GeometryValidationConfig,
    validate_quadrilateral,
)
from attendance_scanner.hough_lines import HoughLineEvidence, HoughLineSegment
from attendance_scanner.segmentation import SegmentationTransform

QUADRILATERAL_CANDIDATE_VERSION = "1.0"
CandidateSource = Literal["mask_fit", "contour", "hough", "mixed"]


class QuadrilateralCandidateConfig(BaseContract):
    """Bounded candidate fitting, deduplication, and geometry policy."""

    max_candidates: int = Field(default=16, ge=1, le=128)
    max_rejected_candidates: int = Field(default=32, ge=1, le=256)
    mask_approximation_epsilon_ratios: Tuple[float, ...] = (0.01, 0.02, 0.04, 0.08)
    hull_approximation_epsilon_ratios: Tuple[float, ...] = (
        0.005,
        0.008,
        0.01,
        0.015,
        0.02,
        0.025,
        0.03,
        0.035,
        0.04,
        0.05,
        0.06,
        0.08,
        0.10,
        0.12,
    )
    dedup_corner_distance_px: float = Field(default=8.0, gt=0.0, le=100.0)
    dedup_polygon_iou: float = Field(default=0.97, ge=0.0, le=1.0)
    dedup_same_source_iou: float = Field(default=0.85, ge=0.0, le=1.0)
    max_line_pairs: int = Field(default=32, ge=1, le=256)
    geometry: GeometryValidationConfig = Field(default_factory=GeometryValidationConfig)

    @model_validator(mode="after")
    def validate_epsilon(self) -> "QuadrilateralCandidateConfig":
        if not self.mask_approximation_epsilon_ratios or any(
            epsilon <= 0.0 or epsilon >= 1.0 for epsilon in self.mask_approximation_epsilon_ratios
        ):
            raise ValueError("mask_approximation_epsilon_ratios must contain values in (0,1)")
        if not self.hull_approximation_epsilon_ratios or any(
            epsilon <= 0.0 or epsilon >= 1.0 for epsilon in self.hull_approximation_epsilon_ratios
        ):
            raise ValueError("hull_approximation_epsilon_ratios must contain values in (0,1)")
        return self


class QuadrilateralCandidate(BaseContract):
    """One canonical quad with source and scoring evidence."""

    candidate_id: int = Field(ge=0)
    corners: CanonicalCorners
    source: CandidateSource
    score: float = Field(ge=0.0, le=1.0)
    mask_quad_iou: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    edge_support: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    geometry_quality: float = Field(ge=0.0, le=1.0)
    evidence: Dict[str, Any] = Field(default_factory=dict)


class RejectedQuadrilateralCandidate(BaseContract):
    """Bounded evidence for candidates rejected before final ranking/fusion."""

    candidate_id: int = Field(ge=0)
    corners: Optional[CanonicalCorners] = None
    source: CandidateSource
    reason_codes: List[GeometryReasonCode] = Field(min_length=1)
    geometry_quality: float = Field(ge=0.0, le=1.0)
    mask_quad_iou: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    evidence: Dict[str, Any] = Field(default_factory=dict)


class QuadrilateralCandidateSet(BaseContract):
    """Bounded multi-source quad pool; no final-candidate decision."""

    generator_version: str = QUADRILATERAL_CANDIDATE_VERSION
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    candidates: List[QuadrilateralCandidate] = Field(default_factory=list)
    rejected_candidates: List[RejectedQuadrilateralCandidate] = Field(default_factory=list)
    source_counts: Dict[str, int] = Field(default_factory=dict)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
    config: QuadrilateralCandidateConfig


def _mask_to_binary(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"mask must be a non-empty 2D array, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all() or float(array.min()) < 0.0 or float(array.max()) > 1.0:
            raise ValueError("floating mask must contain finite values in [0,1]")
        binary = array >= 0.5
    elif np.issubdtype(array.dtype, np.number) or array.dtype == np.bool_:
        binary = array > 0
    else:
        raise ValueError(f"unsupported mask dtype: {array.dtype}")
    return np.asarray(binary, dtype=np.uint8)


def _order_points(points: Sequence[DetectorPoint]) -> Optional[CanonicalCorners]:
    if len(points) != 4:
        return None
    sums = [point[0] + point[1] for point in points]
    differences = [point[1] - point[0] for point in points]
    indices = [
        min(range(4), key=lambda index: (sums[index], index)),
        min(range(4), key=lambda index: (differences[index], index)),
        max(range(4), key=lambda index: (sums[index], -index)),
        max(range(4), key=lambda index: (differences[index], -index)),
    ]
    if len(set(indices)) != 4:
        return None
    try:
        return CanonicalCorners.from_sequence([points[index] for index in indices])
    except ValueError:
        return None


def _polygon_iou_and_coverage_with_mask(
    corners: CanonicalCorners, mask: np.ndarray
) -> Tuple[float, float]:
    polygon = np.zeros(mask.shape, dtype=np.uint8)
    points = np.round(np.asarray(corners.as_list(), dtype=np.float32)).astype(np.int32)
    cv2.fillConvexPoly(polygon, points, 1)
    expected = mask > 0
    predicted = polygon > 0
    intersection = float(np.logical_and(expected, predicted).sum())
    union = float(np.logical_or(expected, predicted).sum())
    mask_sum = float(expected.sum())
    iou = float(intersection / union) if union > 0 else 0.0
    coverage = float(intersection / mask_sum) if mask_sum > 0 else 0.0
    return iou, coverage


def _polygon_iou_with_mask(corners: CanonicalCorners, mask: np.ndarray) -> float:
    iou, _ = _polygon_iou_and_coverage_with_mask(corners, mask)
    return iou


def _candidate_score(
    geometry_quality: float,
    mask_quad_iou: Optional[float],
    edge_support: Optional[float],
) -> float:
    score = 0.7 * geometry_quality
    if mask_quad_iou is not None:
        score += 0.2 * mask_quad_iou
    if edge_support is not None:
        score += 0.1 * edge_support
    return max(0.0, min(1.0, score))


def _line_intersection(
    first: HoughLineSegment,
    second: HoughLineSegment,
) -> Optional[DetectorPoint]:
    x1, y1 = first.p1
    x2, y2 = first.p2
    x3, y3 = second.p1
    x4, y4 = second.p2
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-9:
        return None
    determinant_first = x1 * y2 - y1 * x2
    determinant_second = x3 * y4 - y3 * x4
    return (
        (determinant_first * (x3 - x4) - (x1 - x2) * determinant_second) / denominator,
        (determinant_first * (y3 - y4) - (y1 - y2) * determinant_second) / denominator,
    )


def _hough_candidates(
    evidence: HoughLineEvidence,
    config: QuadrilateralCandidateConfig,
) -> List[Tuple[CanonicalCorners, Dict[str, Any]]]:
    horizontal = [segment for segment in evidence.segments if segment.orientation == "horizontal"]
    vertical = [segment for segment in evidence.segments if segment.orientation == "vertical"]
    horizontal.sort(
        key=lambda segment: (
            min(segment.p1[1], segment.p2[1]),
            -segment.support,
            segment.segment_id,
        )
    )
    vertical.sort(
        key=lambda segment: (
            min(segment.p1[0], segment.p2[0]),
            -segment.support,
            segment.segment_id,
        )
    )
    pairs = itertools.islice(
        itertools.product(
            itertools.combinations(horizontal, 2), itertools.combinations(vertical, 2)
        ),
        config.max_line_pairs,
    )
    candidates: List[Tuple[CanonicalCorners, Dict[str, Any]]] = []
    for horizontal_pair, vertical_pair in pairs:
        top, bottom = sorted(
            horizontal_pair, key=lambda segment: sum((segment.p1[1], segment.p2[1]))
        )
        left, right = sorted(vertical_pair, key=lambda segment: sum((segment.p1[0], segment.p2[0])))
        corners = [
            _line_intersection(top, left),
            _line_intersection(top, right),
            _line_intersection(bottom, right),
            _line_intersection(bottom, left),
        ]
        if any(corner is None for corner in corners):
            continue
        canonical = _order_points([corner for corner in corners if corner is not None])
        if canonical is None:
            continue
        candidates.append(
            (
                canonical,
                {
                    "lineIds": (
                        f"{top.segment_id},{right.segment_id},{bottom.segment_id},{left.segment_id}"
                    ),
                    "support": float(
                        np.mean([top.support, right.support, bottom.support, left.support])
                    ),
                },
            )
        )
    return candidates


def build_quadrilateral_candidates(
    *,
    image_size: Tuple[int, int],
    mask: Optional[np.ndarray] = None,
    transform: Optional[SegmentationTransform] = None,
    cv_candidates: Optional[CvCandidateSet] = None,
    hough_evidence: Optional[HoughLineEvidence] = None,
    config: Optional[QuadrilateralCandidateConfig] = None,
) -> QuadrilateralCandidateSet:
    """Build a bounded multi-source quad pool without choosing or warping a result."""
    policy = config or QuadrilateralCandidateConfig()
    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_size must contain positive width and height")
    source_mask: Optional[np.ndarray] = None
    if mask is not None:
        source_mask = _mask_to_binary(mask)
        if transform is not None and source_mask.shape != (image_height, image_width):
            source_mask = _mask_to_binary(transform.mask_to_source(source_mask))
        if source_mask.shape != (image_height, image_width):
            raise ValueError(
                f"mask shape {source_mask.shape} does not match image size "
                f"{image_height}x{image_width}"
            )

    raw_candidates: List[
        Tuple[CanonicalCorners, CandidateSource, Dict[str, Any], Optional[float]]
    ] = []
    mask_reference: Optional[CanonicalCorners] = None
    if source_mask is not None and int(source_mask.sum()) > 0:
        contours, _ = cv2.findContours(source_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea) if contours else None
        if contour is not None:
            hull = cv2.convexHull(contour)
            hull_perimeter = cv2.arcLength(hull, True)
            found_polygon_quad = False
            for epsilon_ratio in policy.hull_approximation_epsilon_ratios:
                approximation = cv2.approxPolyDP(hull, epsilon_ratio * hull_perimeter, True)
                if len(approximation) == 4:
                    canonical = _order_points(
                        [(float(point[0][0]), float(point[0][1])) for point in approximation]
                    )
                    if canonical is not None:
                        if mask_reference is None:
                            mask_reference = canonical
                        raw_candidates.append(
                            (
                                canonical,
                                "mask_fit",
                                {"epsilonRatio": epsilon_ratio, "fitMethod": "convex_hull_approx"},
                                None,
                            )
                        )
                        found_polygon_quad = True
                        break

            if image_height != image_width:
                mask_rot = cv2.rotate(source_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
                cnts_r, _ = cv2.findContours(mask_rot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts_r:
                    c_r = max(cnts_r, key=cv2.contourArea)
                    hull_r = cv2.convexHull(c_r)
                    hull_peri_r = cv2.arcLength(hull_r, True)
                    for epsilon_ratio in policy.hull_approximation_epsilon_ratios:
                        approximation_r = cv2.approxPolyDP(
                            hull_r, epsilon_ratio * hull_peri_r, True
                        )
                        if len(approximation_r) == 4:
                            mapped_points = [
                                (
                                    float(image_width - 1 - point[0][1]),
                                    float(point[0][0]),
                                )
                                for point in approximation_r
                            ]
                            canonical_r = _order_points(mapped_points)
                            if canonical_r is not None:
                                if mask_reference is None:
                                    mask_reference = canonical_r
                                raw_candidates.append(
                                    (
                                        canonical_r,
                                        "mask_fit",
                                        {
                                            "epsilonRatio": epsilon_ratio,
                                            "fitMethod": "convex_hull_approx_rot90",
                                        },
                                        None,
                                    )
                                )
                                found_polygon_quad = True
                                break

            perimeter = cv2.arcLength(contour, True)
            for epsilon_ratio in policy.mask_approximation_epsilon_ratios:
                approximation = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
                if len(approximation) == 4:
                    canonical = _order_points(
                        [(float(point[0][0]), float(point[0][1])) for point in approximation]
                    )
                    if canonical is not None:
                        if mask_reference is None:
                            mask_reference = canonical
                        raw_candidates.append(
                            (
                                canonical,
                                "mask_fit",
                                {"epsilonRatio": epsilon_ratio, "fitMethod": "contour"},
                                None,
                            )
                        )
                        found_polygon_quad = True

            if not found_polygon_quad:
                rectangle = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rectangle)
                canonical = _order_points([(float(point[0]), float(point[1])) for point in box])
                if canonical is not None:
                    if mask_reference is None:
                        mask_reference = canonical
                    raw_candidates.append(
                        (canonical, "mask_fit", {"fitMethod": "min_area_rect"}, None)
                    )

    if cv_candidates is not None:
        for candidate in cv_candidates.candidates:
            canonical = _order_points(candidate.corners.points)
            if canonical is not None:
                raw_candidates.append(
                    (
                        canonical,
                        "contour",
                        {"sourceContourIndex": candidate.source_contour_index},
                        None,
                    )
                )

    if hough_evidence is not None:
        for canonical, evidence in _hough_candidates(hough_evidence, policy):
            raw_candidates.append((canonical, "hough", evidence, evidence.get("support")))

    if (
        source_mask is not None
        and mask_reference is not None
        and cv_candidates is not None
        and cv_candidates.candidates
    ):
        contour_reference = _order_points(cv_candidates.candidates[0].corners.points)
        if contour_reference is not None:
            mixed_points = [
                (
                    (first[0] + second[0]) / 2.0,
                    (first[1] + second[1]) / 2.0,
                )
                for first, second in zip(
                    mask_reference.as_list(), contour_reference.as_list(), strict=True
                )
            ]
            mixed = _order_points(mixed_points)
            if mixed is not None:
                raw_candidates.append(
                    (
                        mixed,
                        "mixed",
                        {"sources": "mask+contour"},
                        None,
                    )
                )

    accepted: List[QuadrilateralCandidate] = []
    rejected: List[RejectedQuadrilateralCandidate] = []
    for corners, source, evidence, edge_support in raw_candidates:
        validation = validate_quadrilateral(
            corners.as_list(),
            image_size=image_size,
            config=policy.geometry,
        )
        normalized = validation.normalized_corners
        mask_quad_iou, mask_coverage = (
            _polygon_iou_and_coverage_with_mask(normalized, source_mask)
            if normalized is not None and source_mask is not None
            else (None, None)
        )
        cand_evidence = {**evidence, "quadAreaRatio": validation.area_ratio or 0.0}
        if mask_coverage is not None:
            cand_evidence["maskCoverage"] = mask_coverage
        if not validation.valid or normalized is None:
            rejected.append(
                RejectedQuadrilateralCandidate(
                    candidate_id=0,
                    corners=normalized,
                    source=source,
                    reason_codes=validation.reason_codes or [GeometryReasonCode.QUAD_DEGENERATE],
                    geometry_quality=validation.quality_score,
                    mask_quad_iou=mask_quad_iou,
                    evidence=cand_evidence,
                )
            )
            continue
        score = _candidate_score(validation.quality_score, mask_quad_iou, edge_support)
        accepted.append(
            QuadrilateralCandidate(
                candidate_id=0,
                corners=normalized,
                source=source,
                score=score,
                mask_quad_iou=mask_quad_iou,
                edge_support=edge_support,
                geometry_quality=validation.quality_score,
                evidence=cand_evidence,
            )
        )

    accepted.sort(
        key=lambda candidate: (
            -candidate.score,
            -candidate.mask_quad_iou if candidate.mask_quad_iou is not None else 0.0,
            candidate.source,
            candidate.corners.model_dump_json(),
        )
    )
    deduplicated: List[QuadrilateralCandidate] = []
    for quad_candidate in accepted:
        duplicate = False
        for existing_candidate in deduplicated:
            distances = [
                math.dist(first, second)
                for first, second in zip(
                    quad_candidate.corners.as_list(),
                    existing_candidate.corners.as_list(),
                    strict=True,
                )
            ]
            iou = _quad_iou(
                quad_candidate.corners,
                existing_candidate.corners,
                image_width,
                image_height,
            )
            is_strictly_same_source = quad_candidate.source == existing_candidate.source
            is_same_family = is_strictly_same_source or {
                quad_candidate.source,
                existing_candidate.source,
            } == {"mixed", "mask_fit"}
            if is_strictly_same_source:
                iou_threshold = min(policy.dedup_same_source_iou, 0.75)
            elif is_same_family:
                iou_threshold = policy.dedup_same_source_iou
            else:
                iou_threshold = policy.dedup_polygon_iou
            if max(distances) <= policy.dedup_corner_distance_px or iou >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            deduplicated.append(quad_candidate)
        if len(deduplicated) >= policy.max_candidates:
            break
    deduplicated = [
        quad_candidate.model_copy(update={"candidate_id": index})
        for index, quad_candidate in enumerate(deduplicated)
    ]
    rejected.sort(
        key=lambda candidate: (
            -candidate.geometry_quality,
            candidate.source,
            ",".join(reason.value for reason in candidate.reason_codes),
            candidate.corners.model_dump_json() if candidate.corners is not None else "",
        )
    )
    rejected = [
        candidate.model_copy(update={"candidate_id": index})
        for index, candidate in enumerate(rejected[: policy.max_rejected_candidates])
    ]
    source_counts: Dict[str, int] = {}
    for quad_candidate in deduplicated:
        source_counts[quad_candidate.source] = source_counts.get(quad_candidate.source, 0) + 1
    return QuadrilateralCandidateSet(
        image_width=image_width,
        image_height=image_height,
        candidates=deduplicated,
        rejected_candidates=rejected,
        source_counts=dict(sorted(source_counts.items())),
        diagnostics={
            "rawCandidateCount": len(raw_candidates),
            "acceptedCandidateCount": len(accepted),
            "deduplicatedCandidateCount": len(deduplicated),
            "rejectedCandidateCount": len(rejected),
            "maskProvided": source_mask is not None,
            "houghProvided": hough_evidence is not None,
        },
        config=policy,
    )


def _quad_iou(
    first: CanonicalCorners,
    second: CanonicalCorners,
    width: int,
    height: int,
) -> float:
    first_mask = np.zeros((height, width), dtype=np.uint8)
    second_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(first_mask, np.round(np.asarray(first.as_list())).astype(np.int32), 1)
    cv2.fillConvexPoly(second_mask, np.round(np.asarray(second.as_list())).astype(np.int32), 1)
    union = np.logical_or(first_mask, second_mask).sum()
    return float(np.logical_and(first_mask, second_mask).sum() / union) if union else 0.0


__all__ = [
    "CandidateSource",
    "QUADRILATERAL_CANDIDATE_VERSION",
    "QuadrilateralCandidate",
    "QuadrilateralCandidateConfig",
    "QuadrilateralCandidateSet",
    "RejectedQuadrilateralCandidate",
    "build_quadrilateral_candidates",
]
