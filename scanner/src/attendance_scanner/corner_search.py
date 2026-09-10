"""Bounded local corner ROI and edge-sample search for refinement evidence."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from pydantic import Field, model_validator

from attendance_scanner.contracts import BaseContract
from attendance_scanner.detector import CanonicalCorners, DetectorPoint
from attendance_scanner.pipeline.load import LoadedImage

CORNER_SEARCH_VERSION = "1.0"
CornerLabel = Literal["TL", "TR", "BR", "BL"]


class CornerSearchConfig(BaseContract):
    """Resolution-aware bounded ROI and edge search settings."""

    radius_ratio: float = Field(default=0.025, gt=0.0, le=0.2)
    minimum_radius_px: int = Field(default=12, ge=2, le=256)
    maximum_radius_px: int = Field(default=160, ge=4, le=1024)
    search_band_px: int = Field(default=5, ge=1, le=32)
    samples_per_side: int = Field(default=32, ge=4, le=128)
    canny_threshold1: int = Field(default=40, ge=0, le=255)
    canny_threshold2: int = Field(default=120, ge=0, le=255)
    blur_kernel_size: int = Field(default=3, ge=1, le=15)
    debug_directory: Optional[Path] = None

    @model_validator(mode="after")
    def validate_settings(self) -> "CornerSearchConfig":
        if self.minimum_radius_px > self.maximum_radius_px:
            raise ValueError("minimum_radius_px must be <= maximum_radius_px")
        if self.blur_kernel_size % 2 == 0:
            raise ValueError("blur_kernel_size must be odd")
        if self.canny_threshold1 > self.canny_threshold2:
            raise ValueError("canny_threshold1 must be <= canny_threshold2")
        return self


class EdgeSample(BaseContract):
    """One local edge observation tied to an expected corner side."""

    side: Literal["previous", "next"]
    point: DetectorPoint
    distance_from_corner_px: float = Field(ge=0.0)
    normal_offset_px: float
    gradient_score: float = Field(ge=0.0, le=1.0)
    edge_detected: bool


class CornerRoiResult(BaseContract):
    """Evidence from one clipped corner ROI; no refined point is committed."""

    label: CornerLabel
    predicted_point: DetectorPoint
    roi_xywh: Tuple[int, int, int, int]
    radius_px: int = Field(gt=0)
    samples: List[EdgeSample] = Field(default_factory=list)
    best_candidate: Optional[DetectorPoint] = None
    confidence: float = Field(ge=0.0, le=1.0)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


class CornerSearchResult(BaseContract):
    """Four local ROI outputs with source-resolution diagnostics."""

    search_version: str = CORNER_SEARCH_VERSION
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    radius_px: int = Field(gt=0)
    corners: List[CornerRoiResult] = Field(min_length=4, max_length=4)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
    debug_artifacts: List[str] = Field(default_factory=list)


@dataclass(frozen=True)
class _PreparedCornerImage:
    bgr: np.ndarray
    width: int
    height: int


def _prepare_image(image: Union[LoadedImage, np.ndarray]) -> _PreparedCornerImage:
    if isinstance(image, LoadedImage):
        bgr = image.image
    elif isinstance(image, np.ndarray):
        bgr = image
    else:
        raise TypeError("image must be LoadedImage or uint8 BGR NumPy array")
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("image must be a uint8 BGR image with shape HxWx3")
    height, width = bgr.shape[:2]
    if width < 10 or height < 10:
        raise ValueError("image is too small for corner search")
    return _PreparedCornerImage(bgr=bgr, width=width, height=height)


def _coerce_corners(corners: Any) -> List[DetectorPoint]:
    if isinstance(corners, CanonicalCorners):
        return corners.as_list()
    if hasattr(corners, "corners"):
        corners = corners.corners
        if isinstance(corners, CanonicalCorners):
            return corners.as_list()
    if not isinstance(corners, (list, tuple)) or len(corners) != 4:
        raise ValueError("corners must contain exactly four points")
    try:
        points = [(float(point[0]), float(point[1])) for point in corners]
    except (TypeError, IndexError, ValueError) as exc:
        raise ValueError("corners must contain numeric point pairs") from exc
    if not all(math.isfinite(value) for point in points for value in point):
        raise ValueError("corners must contain finite points")
    return points


def _roi_bounds(
    point: DetectorPoint,
    radius: int,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    left = max(0, math.floor(point[0] - radius))
    top = max(0, math.floor(point[1] - radius))
    right = min(width, math.ceil(point[0] + radius + 1))
    bottom = min(height, math.ceil(point[1] + radius + 1))
    return left, top, max(1, right - left), max(1, bottom - top)


def _normalize_gradient(value: float, maximum: float) -> float:
    return max(0.0, min(1.0, value / max(maximum, 1e-6)))


def _search_side(
    gradient: np.ndarray,
    edge_map: np.ndarray,
    predicted: DetectorPoint,
    neighbor: DetectorPoint,
    *,
    side: Literal["previous", "next"],
    roi: Tuple[int, int, int, int],
    radius: int,
    band: int,
    sample_count: int,
) -> List[EdgeSample]:
    left, top, roi_width, roi_height = roi
    direction = np.asarray(neighbor, dtype=np.float64) - np.asarray(predicted, dtype=np.float64)
    side_length = float(np.linalg.norm(direction))
    if side_length <= 1e-9:
        return []
    unit = direction / side_length
    normal = np.asarray((-unit[1], unit[0]), dtype=np.float64)
    max_distance = min(radius, max(8, round(side_length * 0.35)))
    gradient_max = float(np.max(gradient))
    samples: List[EdgeSample] = []
    for distance in np.linspace(0.0, float(max_distance), sample_count):
        center = np.asarray(predicted, dtype=np.float64) + unit * float(distance)
        best: Optional[Tuple[float, float, int, int, bool]] = None
        for offset in range(-band, band + 1):
            candidate = center + normal * offset
            x = round(float(candidate[0]))
            y = round(float(candidate[1]))
            local_x = x - left
            local_y = y - top
            if not (0 <= local_x < roi_width and 0 <= local_y < roi_height):
                continue
            score = _normalize_gradient(float(gradient[local_y, local_x]), gradient_max)
            detected = bool(edge_map[local_y, local_x] > 0)
            effective_score = score if detected else score * 0.35
            if best is None or effective_score > best[0]:
                best = (effective_score, float(offset), x, y, detected)
        if best is not None:
            samples.append(
                EdgeSample(
                    side=side,
                    point=(float(best[2]), float(best[3])),
                    distance_from_corner_px=float(distance),
                    normal_offset_px=best[1],
                    gradient_score=max(0.0, min(1.0, best[0])),
                    edge_detected=best[4],
                )
            )
    return samples


def _debug_overlay(
    roi_image: np.ndarray,
    samples: Sequence[EdgeSample],
    roi: Tuple[int, int, int, int],
) -> np.ndarray:
    output = roi_image.copy()
    left, top, _, _ = roi
    for sample in samples:
        x = round(sample.point[0] - left)
        y = round(sample.point[1] - top)
        color = (0, 255, 0) if sample.edge_detected else (0, 165, 255)
        cv2.circle(output, (x, y), 2, color, -1)
    return output


def _write_debug_overlay(
    image: LoadedImage | np.ndarray,
    label: CornerLabel,
    roi_image: np.ndarray,
    samples: Sequence[EdgeSample],
    roi: Tuple[int, int, int, int],
    directory: Optional[Path],
) -> Optional[str]:
    if directory is None:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    source_path = image.path if isinstance(image, LoadedImage) else Path("<memory>")
    identity = str(source_path.resolve() if source_path.exists() else source_path).encode("utf-8")
    suffix = hashlib.sha256(identity).hexdigest()[:10]
    output = directory / f"{source_path.stem or 'image'}_{suffix}_{label}.roi.png"
    if not cv2.imwrite(str(output), _debug_overlay(roi_image, samples, roi)):
        raise OSError(f"could not write corner ROI debug artifact: {output}")
    return str(output)


def search_corner_rois(
    image: Union[LoadedImage, np.ndarray],
    corners: Any,
    *,
    config: Optional[CornerSearchConfig] = None,
) -> CornerSearchResult:
    """Search four clipped local ROIs and return edge samples only."""
    policy = config or CornerSearchConfig()
    prepared = _prepare_image(image)
    points = _coerce_corners(corners)
    diagonal = math.hypot(prepared.width, prepared.height)
    radius = max(
        policy.minimum_radius_px,
        min(policy.maximum_radius_px, round(diagonal * policy.radius_ratio)),
    )
    labels: Tuple[CornerLabel, ...] = ("TL", "TR", "BR", "BL")
    roi_results: List[CornerRoiResult] = []
    debug_artifacts: List[str] = []
    for index, label in enumerate(labels):
        predicted = points[index]
        roi = _roi_bounds(predicted, radius, prepared.width, prepared.height)
        left, top, roi_width, roi_height = roi
        crop = prepared.bgr[top : top + roi_height, left : left + roi_width]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(
            gray,
            (policy.blur_kernel_size, policy.blur_kernel_size),
            0,
        )
        gradient_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.magnitude(gradient_x, gradient_y)
        edge_map = cv2.Canny(
            blurred,
            policy.canny_threshold1,
            policy.canny_threshold2,
        )
        previous = points[(index - 1) % 4]
        following = points[(index + 1) % 4]
        samples = _search_side(
            gradient,
            edge_map,
            predicted,
            previous,
            side="previous",
            roi=roi,
            radius=radius,
            band=policy.search_band_px,
            sample_count=policy.samples_per_side,
        ) + _search_side(
            gradient,
            edge_map,
            predicted,
            following,
            side="next",
            roi=roi,
            radius=radius,
            band=policy.search_band_px,
            sample_count=policy.samples_per_side,
        )
        detected_samples = [sample for sample in samples if sample.edge_detected]
        confidence = (
            float(np.mean([sample.gradient_score for sample in detected_samples]))
            if detected_samples
            else 0.0
        )
        best_candidate = None
        if detected_samples:
            strongest = max(
                detected_samples,
                key=lambda sample: (sample.gradient_score, -sample.distance_from_corner_px),
            )
            best_candidate = strongest.point
        debug_path = _write_debug_overlay(
            image,
            label,
            crop,
            samples,
            roi,
            policy.debug_directory,
        )
        if debug_path is not None:
            debug_artifacts.append(debug_path)
        roi_results.append(
            CornerRoiResult(
                label=label,
                predicted_point=predicted,
                roi_xywh=roi,
                radius_px=radius,
                samples=samples,
                best_candidate=best_candidate,
                confidence=confidence,
                diagnostics={
                    "edgeSampleCount": len(samples),
                    "detectedEdgeSampleCount": len(detected_samples),
                    "previousNeighbor": str(previous),
                    "nextNeighbor": str(following),
                },
            )
        )
    return CornerSearchResult(
        source_width=prepared.width,
        source_height=prepared.height,
        radius_px=radius,
        corners=roi_results,
        diagnostics={
            "config": policy.model_dump(mode="json", by_alias=True),
            "roiCount": 4,
            "totalSampleCount": sum(len(result.samples) for result in roi_results),
        },
        debug_artifacts=debug_artifacts,
    )


__all__ = [
    "CORNER_SEARCH_VERSION",
    "CornerRoiResult",
    "CornerSearchConfig",
    "CornerSearchResult",
    "EdgeSample",
    "search_corner_rois",
]
