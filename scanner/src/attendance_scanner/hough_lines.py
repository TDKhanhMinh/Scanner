"""Hough line evidence and bounded border-aware line extrapolation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from pydantic import Field, model_validator

from attendance_scanner.contracts import BaseContract
from attendance_scanner.detector import DetectorPoint
from attendance_scanner.edge_support import EdgeMap, EdgeSupportConfig, build_edge_map
from attendance_scanner.pipeline.load import LoadedImage

HOUGH_LINE_VERSION = "1.0"
LineOrientation = Literal["horizontal", "vertical", "diagonal"]


class HoughLineConfig(BaseContract):
    """Bounded Hough, clustering, and extrapolation configuration."""

    max_dimension: int = Field(default=1200, ge=100, le=8000)
    blur_kernel_size: int = Field(default=5, ge=1, le=31)
    canny_threshold1: int = Field(default=50, ge=0, le=255)
    canny_threshold2: int = Field(default=150, ge=0, le=255)
    morph_kernel_size: int = Field(default=3, ge=1, le=31)
    morph_close_iterations: int = Field(default=1, ge=0, le=3)
    morph_dilate_iterations: int = Field(default=0, ge=0, le=2)
    hough_rho_px: float = Field(default=1.0, gt=0.0, le=10.0)
    hough_theta_deg: float = Field(default=1.0, gt=0.0, le=10.0)
    hough_threshold: int = Field(default=40, ge=1, le=500)
    min_line_length_ratio: float = Field(default=0.08, gt=0.0, le=1.0)
    max_line_gap_px: int = Field(default=12, ge=0, le=200)
    max_segments: int = Field(default=64, ge=1, le=500)
    max_clusters: int = Field(default=24, ge=1, le=100)
    angle_cluster_deg: float = Field(default=8.0, gt=0.0, le=45.0)
    distance_cluster_px: float = Field(default=24.0, gt=0.0, le=500.0)
    border_margin_ratio: float = Field(default=0.02, ge=0.0, le=0.2)
    max_extrapolation_ratio: float = Field(default=3.0, gt=0.0, le=20.0)

    @model_validator(mode="after")
    def validate_settings(self) -> "HoughLineConfig":
        if self.blur_kernel_size % 2 == 0 or self.morph_kernel_size % 2 == 0:
            raise ValueError("blur_kernel_size and morph_kernel_size must be odd")
        if self.canny_threshold1 > self.canny_threshold2:
            raise ValueError("canny_threshold1 must be <= canny_threshold2")
        return self


class HoughLineSegment(BaseContract):
    """One source-space line segment with support and border diagnostics."""

    segment_id: int = Field(ge=0)
    p1: DetectorPoint
    p2: DetectorPoint
    angle_deg: float = Field(ge=0.0, lt=180.0)
    length_px: float = Field(gt=0.0)
    support: float = Field(ge=0.0, le=1.0)
    orientation: LineOrientation
    border_contact: List[str] = Field(default_factory=list)
    cluster_id: Optional[int] = Field(default=None, ge=0)


class HoughLineCluster(BaseContract):
    """A deterministic group of near-parallel lines in one image region."""

    cluster_id: int = Field(ge=0)
    segment_ids: List[int] = Field(min_length=1)
    mean_angle_deg: float = Field(ge=0.0, lt=180.0)
    mean_distance_px: float
    support: float = Field(ge=0.0, le=1.0)
    orientation: LineOrientation


class HoughLineEvidence(BaseContract):
    """Bounded line pool; never contains a final quadrilateral decision."""

    generator_version: str = HOUGH_LINE_VERSION
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    detection_width: int = Field(gt=0)
    detection_height: int = Field(gt=0)
    scale_factor: float = Field(gt=0.0, le=1.0)
    segments: List[HoughLineSegment] = Field(default_factory=list)
    clusters: List[HoughLineCluster] = Field(default_factory=list)
    preprocessing: HoughLineConfig
    diagnostics: Dict[str, Union[str, int, float, bool]] = Field(default_factory=dict)


@dataclass(frozen=True)
class _LineGeometry:
    p1: DetectorPoint
    p2: DetectorPoint
    angle_deg: float
    length_px: float
    distance_px: float


def _normalized_angle(p1: DetectorPoint, p2: DetectorPoint) -> float:
    angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])) % 180.0
    return angle


def _orientation(angle: float) -> LineOrientation:
    if min(angle, 180.0 - angle) <= 15.0:
        return "horizontal"
    if abs(angle - 90.0) <= 15.0:
        return "vertical"
    return "diagonal"


def _line_distance(p1: DetectorPoint, p2: DetectorPoint, width: int, height: int) -> float:
    midpoint = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
    canonical_angle = _normalized_angle(p1, p2)
    if canonical_angle > 90.0:
        canonical_angle -= 180.0
    angle = math.radians(canonical_angle)
    normal = (-math.sin(angle), math.cos(angle))
    center = (width / 2.0, height / 2.0)
    return (midpoint[0] - center[0]) * normal[0] + (midpoint[1] - center[1]) * normal[1]


def _border_contact(
    p1: DetectorPoint,
    p2: DetectorPoint,
    width: int,
    height: int,
    margin: float,
) -> List[str]:
    points = (p1, p2)
    edges: List[str] = []
    if any(point[0] <= margin for point in points):
        edges.append("left")
    if any(point[0] >= width - 1 - margin for point in points):
        edges.append("right")
    if any(point[1] <= margin for point in points):
        edges.append("top")
    if any(point[1] >= height - 1 - margin for point in points):
        edges.append("bottom")
    return edges


def _segment_support(edge_map: np.ndarray, p1: DetectorPoint, p2: DetectorPoint) -> float:
    height, width = edge_map.shape[:2]
    length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    samples = max(16, round(length))
    supported = 0
    for fraction in np.linspace(0.0, 1.0, samples):
        x = round(p1[0] + (p2[0] - p1[0]) * float(fraction))
        y = round(p1[1] + (p2[1] - p1[1]) * float(fraction))
        if 0 <= x < width and 0 <= y < height and edge_map[y, x] > 0:
            supported += 1
    return supported / float(samples)


def _geometry(segment: HoughLineSegment, width: int, height: int) -> _LineGeometry:
    return _LineGeometry(
        p1=segment.p1,
        p2=segment.p2,
        angle_deg=segment.angle_deg,
        length_px=segment.length_px,
        distance_px=_line_distance(segment.p1, segment.p2, width, height),
    )


def _angle_distance(first: float, second: float) -> float:
    return min(abs(first - second), 180.0 - abs(first - second))


def _circular_mean_angle(angles: Sequence[float]) -> float:
    """Average line orientations on a modulo-180 circle."""
    if not angles:
        return 0.0
    sine = sum(math.sin(math.radians(2.0 * angle)) for angle in angles)
    cosine = sum(math.cos(math.radians(2.0 * angle)) for angle in angles)
    return (math.degrees(math.atan2(sine, cosine)) / 2.0) % 180.0


def _cluster_segments(
    segments: List[HoughLineSegment],
    width: int,
    height: int,
    config: HoughLineConfig,
) -> List[HoughLineCluster]:
    clusters: List[List[HoughLineSegment]] = []
    geometries: List[List[_LineGeometry]] = []
    for segment in segments:
        geometry = _geometry(segment, width, height)
        assigned = False
        for cluster, cluster_geometries in zip(clusters, geometries, strict=True):
            reference = cluster_geometries[0]
            if (
                _angle_distance(geometry.angle_deg, reference.angle_deg) <= config.angle_cluster_deg
                and abs(geometry.distance_px - reference.distance_px) <= config.distance_cluster_px
            ):
                cluster.append(segment)
                cluster_geometries.append(geometry)
                assigned = True
                break
        if not assigned:
            clusters.append([segment])
            geometries.append([geometry])
    ranked = sorted(
        zip(clusters, geometries, strict=True),
        key=lambda pair: (
            -max(segment.support for segment in pair[0]),
            -sum(segment.length_px for segment in pair[0]),
            pair[0][0].segment_id,
        ),
    )[: config.max_clusters]
    results: List[HoughLineCluster] = []
    for cluster_id, (cluster, cluster_geometries) in enumerate(ranked):
        mean_angle = _circular_mean_angle([item.angle_deg for item in cluster_geometries])
        mean_distance = sum(item.distance_px for item in cluster_geometries) / len(
            cluster_geometries
        )
        orientation = max(
            ("horizontal", "vertical", "diagonal"),
            key=lambda candidate: sum(segment.orientation == candidate for segment in cluster),
        )
        results.append(
            HoughLineCluster(
                cluster_id=cluster_id,
                segment_ids=[segment.segment_id for segment in cluster],
                mean_angle_deg=mean_angle,
                mean_distance_px=mean_distance,
                support=float(np.mean([segment.support for segment in cluster])),
                orientation=orientation,  # type: ignore[arg-type]
            )
        )
        for segment in cluster:
            segment.cluster_id = cluster_id
    return results


def detect_hough_lines(
    image: Union[LoadedImage, np.ndarray],
    *,
    config: Optional[HoughLineConfig] = None,
    edge_map: Optional[EdgeMap] = None,
) -> HoughLineEvidence:
    """Detect and cluster bounded line evidence without selecting a final quad."""
    policy = config or HoughLineConfig()
    if edge_map is None:
        edge_map = build_edge_map(
            image,
            config=EdgeSupportConfig(
                max_dimension=policy.max_dimension,
                blur_kernel_size=policy.blur_kernel_size,
                canny_threshold1=policy.canny_threshold1,
                canny_threshold2=policy.canny_threshold2,
                morph_kernel_size=policy.morph_kernel_size,
                morph_close_iterations=policy.morph_close_iterations,
                morph_dilate_iterations=policy.morph_dilate_iterations,
            ),
        )
    min_length = policy.min_line_length_ratio * min(
        edge_map.detection_width, edge_map.detection_height
    )
    lines = cv2.HoughLinesP(
        edge_map.edges,
        rho=policy.hough_rho_px,
        theta=math.radians(policy.hough_theta_deg),
        threshold=policy.hough_threshold,
        minLineLength=min_length,
        maxLineGap=policy.max_line_gap_px,
    )
    detected: List[HoughLineSegment] = []
    if lines is not None:
        raw_lines = sorted(
            (np.asarray(line).reshape(-1).tolist() for line in lines),
            key=lambda values: (
                min(values[0], values[2]),
                min(values[1], values[3]),
                max(values[0], values[2]),
                max(values[1], values[3]),
            ),
        )
        for values in raw_lines:
            x1, y1, x2, y2 = (float(value) for value in values)
            scale_back = 1.0 / edge_map.scale_factor
            p1 = (x1 * scale_back, y1 * scale_back)
            p2 = (x2 * scale_back, y2 * scale_back)
            length = math.dist(p1, p2)
            if length <= 0.0:
                continue
            angle = _normalized_angle(p1, p2)
            detected.append(
                HoughLineSegment(
                    segment_id=len(detected),
                    p1=p1,
                    p2=p2,
                    angle_deg=angle,
                    length_px=length,
                    support=_segment_support(
                        edge_map.edges,
                        (x1, y1),
                        (x2, y2),
                    ),
                    orientation=_orientation(angle),
                    border_contact=_border_contact(
                        p1,
                        p2,
                        edge_map.source_width,
                        edge_map.source_height,
                        max(edge_map.source_width, edge_map.source_height)
                        * policy.border_margin_ratio,
                    ),
                )
            )
    detected.sort(key=lambda segment: (-segment.support, -segment.length_px, segment.segment_id))
    detected = [
        segment.model_copy(update={"segment_id": index})
        for index, segment in enumerate(detected[: policy.max_segments])
    ]
    clusters = _cluster_segments(
        detected,
        edge_map.source_width,
        edge_map.source_height,
        policy,
    )
    return HoughLineEvidence(
        source_width=edge_map.source_width,
        source_height=edge_map.source_height,
        detection_width=edge_map.detection_width,
        detection_height=edge_map.detection_height,
        scale_factor=edge_map.scale_factor,
        segments=detected,
        clusters=clusters,
        preprocessing=policy,
        diagnostics={
            "rawLineCount": int(0 if lines is None else len(lines)),
            "keptSegmentCount": len(detected),
            "clusterCount": len(clusters),
            "houghThreshold": policy.hough_threshold,
        },
    )


def extrapolate_line_to_bounds(
    segment: HoughLineSegment,
    *,
    image_size: Tuple[int, int],
    max_extrapolation_ratio: float = 3.0,
) -> HoughLineSegment:
    """Extend a line to image bounds while limiting distance beyond observed segment."""
    width, height = image_size
    if max_extrapolation_ratio <= 0.0:
        raise ValueError("max_extrapolation_ratio must be positive")
    p1 = np.asarray(segment.p1, dtype=np.float64)
    p2 = np.asarray(segment.p2, dtype=np.float64)
    direction = p2 - p1
    length = float(np.linalg.norm(direction))
    if length <= 1e-9:
        raise ValueError("Cannot extrapolate a zero-length line")
    unit = direction / length
    intersections: List[Tuple[float, np.ndarray]] = []
    for coordinate, value in ((0, 0.0), (0, float(width - 1)), (1, 0.0), (1, float(height - 1))):
        denominator = unit[coordinate]
        if abs(denominator) <= 1e-9:
            continue
        parameter = (value - p1[coordinate]) / denominator
        point = p1 + parameter * unit
        other = 1 - coordinate
        limit = float(height - 1 if coordinate == 0 else width - 1)
        if -1e-6 <= point[other] <= limit + 1e-6:
            intersections.append((parameter, point))
    if len(intersections) < 2:
        return segment
    intersections.sort(key=lambda item: item[0])
    min_parameter, max_parameter = intersections[0][0], intersections[-1][0]
    limit = max_extrapolation_ratio * length
    min_parameter = max(min_parameter, -limit)
    max_parameter = min(max_parameter, length + limit)
    result_p1 = p1 + min_parameter * unit
    result_p2 = p1 + max_parameter * unit
    return segment.model_copy(
        update={
            "p1": (float(result_p1[0]), float(result_p1[1])),
            "p2": (float(result_p2[0]), float(result_p2[1])),
            "length_px": float(np.linalg.norm(result_p2 - result_p1)),
            "border_contact": ["extrapolated"],
        }
    )


__all__ = [
    "HOUGH_LINE_VERSION",
    "HoughLineCluster",
    "HoughLineConfig",
    "HoughLineEvidence",
    "HoughLineSegment",
    "detect_hough_lines",
    "extrapolate_line_to_bounds",
]
