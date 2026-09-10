"""Detection-edge construction and quadrilateral boundary support scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
from pydantic import Field, model_validator

from attendance_scanner.contracts import BaseContract
from attendance_scanner.detector import CanonicalCorners, DetectorPoint
from attendance_scanner.pipeline.load import LoadedImage

EDGE_SUPPORT_VERSION = "1.0"


class EdgeSupportConfig(BaseContract):
    """Versioned edge-map and band-scoring configuration."""

    max_dimension: int = Field(default=1200, ge=100, le=8000)
    blur_kernel_size: int = Field(default=5, ge=1, le=31)
    canny_threshold1: int = Field(default=50, ge=0, le=255)
    canny_threshold2: int = Field(default=150, ge=0, le=255)
    morph_kernel_size: int = Field(default=3, ge=1, le=31)
    morph_close_iterations: int = Field(default=1, ge=0, le=3)
    morph_dilate_iterations: int = Field(default=0, ge=0, le=2)
    band_ratio: float = Field(default=0.01, gt=0.0, le=0.1)
    minimum_band_px: int = Field(default=4, ge=1, le=32)
    samples_per_edge: int = Field(default=80, ge=16, le=500)

    @model_validator(mode="after")
    def validate_settings(self) -> "EdgeSupportConfig":
        if self.blur_kernel_size % 2 == 0 or self.morph_kernel_size % 2 == 0:
            raise ValueError("blur_kernel_size and morph_kernel_size must be odd")
        if self.canny_threshold1 > self.canny_threshold2:
            raise ValueError("canny_threshold1 must be <= canny_threshold2")
        return self


@dataclass(frozen=True)
class EdgeMap:
    """Detection-resolution edge evidence and source mapping metadata."""

    edges: np.ndarray
    source_width: int
    source_height: int
    detection_width: int
    detection_height: int
    scale_factor: float
    config: EdgeSupportConfig


class EdgeSupportResult(BaseContract):
    """Per-edge and aggregate support evidence for one candidate quad."""

    scorer_version: str = EDGE_SUPPORT_VERSION
    edge_scores: Dict[str, float]
    overall_score: float = Field(ge=0.0, le=1.0)
    weakest_edge_score: float = Field(ge=0.0, le=1.0)
    band_detection_px: int = Field(gt=0)
    samples_per_edge: int = Field(gt=0)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


def _prepare_detection_copy(
    image: Union[LoadedImage, np.ndarray], config: EdgeSupportConfig
) -> Tuple[np.ndarray, int, int, int, int, float]:
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
        raise ValueError("image is too small for edge support")
    max_dimension = max(source_width, source_height)
    if max_dimension <= config.max_dimension:
        return bgr.copy(), source_width, source_height, source_width, source_height, 1.0
    scale = config.max_dimension / float(max_dimension)
    detection_width = max(1, round(source_width * scale))
    detection_height = max(1, round(source_height * scale))
    return (
        cv2.resize(bgr, (detection_width, detection_height), interpolation=cv2.INTER_AREA),
        source_width,
        source_height,
        detection_width,
        detection_height,
        scale,
    )


def build_edge_map(
    image: Union[LoadedImage, np.ndarray],
    *,
    config: EdgeSupportConfig | None = None,
) -> EdgeMap:
    """Build a bounded Canny/morphology edge map at detection resolution."""
    policy = config or EdgeSupportConfig()
    detection, source_width, source_height, detection_width, detection_height, scale = (
        _prepare_detection_copy(image, policy)
    )
    gray = cv2.cvtColor(detection, cv2.COLOR_BGR2GRAY)
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
    if policy.morph_close_iterations:
        edges = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=policy.morph_close_iterations,
        )
    if policy.morph_dilate_iterations:
        edges = cv2.dilate(edges, kernel, iterations=policy.morph_dilate_iterations)
    return EdgeMap(
        edges=edges,
        source_width=source_width,
        source_height=source_height,
        detection_width=detection_width,
        detection_height=detection_height,
        scale_factor=scale,
        config=policy,
    )


def _coerce_corners(corners: Any) -> List[DetectorPoint]:
    if isinstance(corners, CanonicalCorners):
        return corners.as_list()
    if hasattr(corners, "corners"):
        corners = corners.corners
        if isinstance(corners, CanonicalCorners):
            return corners.as_list()
        if hasattr(corners, "points"):
            corners = corners.points
    if not isinstance(corners, (list, tuple)) or len(corners) != 4:
        raise ValueError("corners must contain exactly four points")
    try:
        points = [(float(point[0]), float(point[1])) for point in corners]
    except (TypeError, IndexError, ValueError) as exc:
        raise ValueError("corners must contain numeric point pairs") from exc
    if not all(math.isfinite(value) for point in points for value in point):
        raise ValueError("corners must contain finite points")
    return points


def score_quad_edge_support(
    edge_map: EdgeMap,
    corners: Any,
    *,
    band_ratio: float | None = None,
) -> EdgeSupportResult:
    """Score edge pixels in a narrow band around each original-space quad edge."""
    points = _coerce_corners(corners)
    effective_band_ratio = band_ratio if band_ratio is not None else edge_map.config.band_ratio
    if not 0.0 < effective_band_ratio <= 0.1:
        raise ValueError("band_ratio must be in (0,0.1]")
    band_source_px = max(
        edge_map.config.minimum_band_px,
        round(effective_band_ratio * min(edge_map.source_width, edge_map.source_height)),
    )
    band_detection_px = max(1, round(band_source_px * edge_map.scale_factor))
    height, width = edge_map.edges.shape[:2]
    scores: Dict[str, float] = {}
    labels = ("top", "right", "bottom", "left")
    for index, label in enumerate(labels):
        start = points[index]
        end = points[(index + 1) % 4]
        start_detection = (start[0] * edge_map.scale_factor, start[1] * edge_map.scale_factor)
        end_detection = (end[0] * edge_map.scale_factor, end[1] * edge_map.scale_factor)
        dx = end_detection[0] - start_detection[0]
        dy = end_detection[1] - start_detection[1]
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            scores[label] = 0.0
            continue
        normal = (-dy / length, dx / length)
        samples = max(edge_map.config.samples_per_edge, round(length))
        supported = 0
        for fraction in np.linspace(0.0, 1.0, samples):
            center_x = start_detection[0] + dx * float(fraction)
            center_y = start_detection[1] + dy * float(fraction)
            found = False
            for offset in range(-band_detection_px, band_detection_px + 1):
                x = round(center_x + normal[0] * offset)
                y = round(center_y + normal[1] * offset)
                if 0 <= x < width and 0 <= y < height and edge_map.edges[y, x] > 0:
                    found = True
                    break
            supported += int(found)
        scores[label] = supported / float(samples)
    return EdgeSupportResult(
        edge_scores=scores,
        overall_score=float(np.mean(list(scores.values()))) if scores else 0.0,
        weakest_edge_score=min(scores.values()) if scores else 0.0,
        band_detection_px=band_detection_px,
        samples_per_edge=edge_map.config.samples_per_edge,
        diagnostics={
            "configVersion": EDGE_SUPPORT_VERSION,
            "config": edge_map.config.model_dump(mode="json", by_alias=True),
            "sourceSize": f"{edge_map.source_width}x{edge_map.source_height}",
            "detectionSize": f"{edge_map.detection_width}x{edge_map.detection_height}",
            "scaleFactor": edge_map.scale_factor,
        },
    )


__all__ = [
    "EDGE_SUPPORT_VERSION",
    "EdgeMap",
    "EdgeSupportConfig",
    "EdgeSupportResult",
    "build_edge_map",
    "score_quad_edge_support",
]
