"""Deterministic OpenCV quadrilateral candidate generation without warping."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from pydantic import Field, model_validator

from attendance_scanner.contracts import BaseContract
from attendance_scanner.detector import CandidateCorners, DetectorPoint
from attendance_scanner.pipeline.detect import order_corners
from attendance_scanner.pipeline.load import LoadedImage

CV_CANDIDATE_GENERATOR_VERSION = "1.0"


class CvCandidateConfig(BaseContract):
    """Preprocessing, approximation, and top-N limits for CV candidates."""

    max_dimension: int = Field(default=1200, ge=100, le=8000)
    blur_kernel_size: int = Field(default=5, ge=1, le=31)
    canny_threshold1: int = Field(default=50, ge=0, le=255)
    canny_threshold2: int = Field(default=150, ge=0, le=255)
    morph_kernel_size: int = Field(default=3, ge=1, le=31)
    morph_iterations: int = Field(default=2, ge=0, le=10)
    min_area_ratio: float = Field(default=0.01, gt=0.0, lt=1.0)
    max_area_ratio: float = Field(default=0.995, gt=0.0, le=1.0)
    min_side_ratio: float = Field(default=0.05, gt=0.0, lt=1.0)
    min_angle_deg: float = Field(default=15.0, ge=0.0, lt=90.0)
    max_angle_deg: float = Field(default=165.0, gt=90.0, le=180.0)
    approximation_epsilon_ratios: Tuple[float, ...] = (0.015, 0.02, 0.03, 0.04, 0.06)
    max_candidates: int = Field(default=8, ge=1, le=64)

    @model_validator(mode="after")
    def validate_relationships(self) -> "CvCandidateConfig":
        if self.min_area_ratio >= self.max_area_ratio:
            raise ValueError("min_area_ratio must be less than max_area_ratio")
        if self.canny_threshold1 > self.canny_threshold2:
            raise ValueError("canny_threshold1 must be <= canny_threshold2")
        if self.blur_kernel_size % 2 == 0 or self.morph_kernel_size % 2 == 0:
            raise ValueError("blur_kernel_size and morph_kernel_size must be odd")
        if not self.approximation_epsilon_ratios or any(
            epsilon <= 0.0 or epsilon >= 1.0 for epsilon in self.approximation_epsilon_ratios
        ):
            raise ValueError("approximation_epsilon_ratios must contain values in (0,1)")
        return self


class CvCandidate(BaseContract):
    """One candidate quad plus scalar diagnostics for later ranking/fusion."""

    candidate_id: int = Field(ge=0)
    corners: CandidateCorners
    contour_area_px: float = Field(gt=0.0)
    area_ratio: float = Field(gt=0.0, lt=1.0)
    perimeter_px: float = Field(gt=0.0)
    convex: bool
    confidence: float = Field(ge=0.0, le=1.0)
    source_contour_index: int = Field(ge=0)
    approximation_epsilon_ratio: float = Field(gt=0.0, lt=1.0)


class CvCandidateSet(BaseContract):
    """Top-N deterministic candidate pool and preprocessing provenance."""

    generator_version: str = CV_CANDIDATE_GENERATOR_VERSION
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    detection_width: int = Field(gt=0)
    detection_height: int = Field(gt=0)
    scale_factor: float = Field(gt=0.0, le=1.0)
    candidates: List[CvCandidate] = Field(default_factory=list)
    preprocessing: CvCandidateConfig


@dataclass(frozen=True)
class _PreparedImage:
    image: np.ndarray
    source_width: int
    source_height: int
    detection_width: int
    detection_height: int
    scale_factor: float


def _prepare_image(image: Union[LoadedImage, np.ndarray]) -> _PreparedImage:
    if isinstance(image, LoadedImage):
        bgr = image.image
    elif isinstance(image, np.ndarray):
        bgr = image
    else:
        raise TypeError("image must be LoadedImage or uint8 BGR NumPy array")
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("image must be a uint8 BGR image with shape HxWx3")
    source_height, source_width = bgr.shape[:2]
    if source_height < 10 or source_width < 10:
        raise ValueError("image is too small for candidate generation")
    return _PreparedImage(
        image=bgr,
        source_width=source_width,
        source_height=source_height,
        detection_width=source_width,
        detection_height=source_height,
        scale_factor=1.0,
    )


def _detection_copy(prepared: _PreparedImage, max_dimension: int) -> _PreparedImage:
    max_source_dimension = max(prepared.source_width, prepared.source_height)
    if max_source_dimension <= max_dimension:
        return prepared
    scale = max_dimension / float(max_source_dimension)
    detection_width = max(1, round(prepared.source_width * scale))
    detection_height = max(1, round(prepared.source_height * scale))
    resized = cv2.resize(
        prepared.image,
        (detection_width, detection_height),
        interpolation=cv2.INTER_AREA,
    )
    return _PreparedImage(
        image=resized,
        source_width=prepared.source_width,
        source_height=prepared.source_height,
        detection_width=detection_width,
        detection_height=detection_height,
        scale_factor=scale,
    )


def _angle_degrees(first: DetectorPoint, center: DetectorPoint, last: DetectorPoint) -> float:
    first_vector = np.asarray(first, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    last_vector = np.asarray(last, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    denominator = float(np.linalg.norm(first_vector) * np.linalg.norm(last_vector))
    if denominator <= 1e-9:
        return 0.0
    cosine = float(np.dot(first_vector, last_vector) / denominator)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _valid_candidate(
    points: Sequence[DetectorPoint],
    *,
    width: int,
    height: int,
    config: CvCandidateConfig,
) -> bool:
    if len(points) != 4 or len({(round(x, 5), round(y, 5)) for x, y in points}) != 4:
        return False
    if not all(math.isfinite(value) for point in points for value in point):
        return False
    if not cv2.isContourConvex(np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)):
        return False
    area = abs(float(cv2.contourArea(np.asarray(points, dtype=np.float32))))
    area_ratio = area / float(width * height)
    if area_ratio < config.min_area_ratio or area_ratio > config.max_area_ratio:
        return False
    side_lengths = [math.dist(points[index], points[(index + 1) % 4]) for index in range(4)]
    if min(side_lengths) < min(width, height) * config.min_side_ratio:
        return False
    angles = [
        _angle_degrees(points[(index - 1) % 4], points[index], points[(index + 1) % 4])
        for index in range(4)
    ]
    return all(config.min_angle_deg <= angle <= config.max_angle_deg for angle in angles)


def _candidate_confidence(
    area_ratio: float, points: Sequence[DetectorPoint], width: int, height: int
) -> float:
    side_lengths = [math.dist(points[index], points[(index + 1) % 4]) for index in range(4)]
    symmetry = 0.5 * (
        min(side_lengths[0], side_lengths[2]) / max(side_lengths[0], side_lengths[2], 1e-9)
        + min(side_lengths[1], side_lengths[3]) / max(side_lengths[1], side_lengths[3], 1e-9)
    )
    angles = [
        _angle_degrees(points[(index - 1) % 4], points[index], points[(index + 1) % 4])
        for index in range(4)
    ]
    angle_score = max(0.0, 1.0 - sum(abs(angle - 90.0) for angle in angles) / 180.0)
    area_score = min(1.0, area_ratio / 0.3)
    border_distance = min(
        min(point[0], width - point[0], point[1], height - point[1]) for point in points
    )
    border_score = max(0.0, min(1.0, border_distance / (0.05 * min(width, height))))
    return max(
        0.0, min(1.0, 0.45 * symmetry + 0.3 * angle_score + 0.2 * area_score + 0.05 * border_score)
    )


def generate_cv_candidates(
    image: Union[LoadedImage, np.ndarray],
    *,
    config: Optional[CvCandidateConfig] = None,
) -> CvCandidateSet:
    """Generate a deterministic top-N candidate pool without perspective warp."""
    policy = config or CvCandidateConfig()
    prepared = _detection_copy(_prepare_image(image), policy.max_dimension)
    gray = cv2.cvtColor(prepared.image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(
        gray,
        (policy.blur_kernel_size, policy.blur_kernel_size),
        0,
    )
    edges = cv2.Canny(blurred, policy.canny_threshold1, policy.canny_threshold2)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (policy.morph_kernel_size, policy.morph_kernel_size),
    )
    processed = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=policy.morph_iterations,
    )
    contours, _ = cv2.findContours(processed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_order = sorted(
        enumerate(contours),
        key=lambda item: (-float(cv2.contourArea(item[1])), item[0]),
    )
    candidates: List[CvCandidate] = []
    seen: set[Tuple[Tuple[float, float], ...]] = set()
    for contour_index, contour in contour_order:
        contour_area = float(cv2.contourArea(contour))
        if contour_area <= 0.0:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 1e-9:
            continue
        for epsilon_ratio in policy.approximation_epsilon_ratios:
            approximation = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
            if len(approximation) != 4:
                continue
            ordered_detection = order_corners(approximation)
            detection_points = [(float(point[0]), float(point[1])) for point in ordered_detection]
            if not _valid_candidate(
                detection_points,
                width=prepared.detection_width,
                height=prepared.detection_height,
                config=policy,
            ):
                continue
            scale_back = 1.0 / prepared.scale_factor
            source_points = [(x * scale_back, y * scale_back) for x, y in detection_points]
            identity = tuple((round(x, 3), round(y, 3)) for x, y in source_points)
            if identity in seen:
                continue
            seen.add(identity)
            source_area = abs(float(cv2.contourArea(np.asarray(source_points, dtype=np.float32))))
            source_perimeter = float(
                sum(
                    math.dist(source_points[index], source_points[(index + 1) % 4])
                    for index in range(4)
                )
            )
            source_area_ratio = source_area / float(prepared.source_width * prepared.source_height)
            confidence = _candidate_confidence(
                source_area_ratio,
                source_points,
                prepared.source_width,
                prepared.source_height,
            )
            candidates.append(
                CvCandidate(
                    candidate_id=0,
                    corners=CandidateCorners(
                        points=source_points,
                        source="cv_contour",
                        coordinate_space="original_pixels",
                        confidence=confidence,
                        diagnostics={
                            "contourAreaPx": source_area,
                            "perimeterPx": source_perimeter,
                            "convex": True,
                            "epsilonRatio": epsilon_ratio,
                        },
                    ),
                    contour_area_px=source_area,
                    area_ratio=source_area_ratio,
                    perimeter_px=source_perimeter,
                    convex=True,
                    confidence=confidence,
                    source_contour_index=contour_index,
                    approximation_epsilon_ratio=epsilon_ratio,
                )
            )

    candidates.sort(
        key=lambda candidate: (
            -candidate.confidence,
            -candidate.area_ratio,
            candidate.source_contour_index,
            candidate.approximation_epsilon_ratio,
        )
    )
    candidates = [
        candidate.model_copy(update={"candidate_id": index})
        for index, candidate in enumerate(candidates[: policy.max_candidates])
    ]
    return CvCandidateSet(
        source_width=prepared.source_width,
        source_height=prepared.source_height,
        detection_width=prepared.detection_width,
        detection_height=prepared.detection_height,
        scale_factor=prepared.scale_factor,
        candidates=candidates,
        preprocessing=policy,
    )


__all__ = [
    "CV_CANDIDATE_GENERATOR_VERSION",
    "CvCandidate",
    "CvCandidateConfig",
    "CvCandidateSet",
    "generate_cv_candidates",
]
