"""Occlusion evidence extraction: paper boundary ridge detection and containment scoring."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from pydantic import Field, model_validator

from attendance_scanner.contracts import BaseContract
from attendance_scanner.detector import CanonicalCorners, DetectorPoint
from attendance_scanner.edge_support import EdgeMap, build_edge_map
from attendance_scanner.pipeline.load import LoadedImage

OCCLUSION_EVIDENCE_VERSION = "1.0"


class OcclusionEvidenceConfig(BaseContract):
    """Configuration for paper boundary ridge detection and candidate containment."""

    min_defect_depth_px: float = Field(default=10.0, ge=1.0)
    min_defect_depth_ratio: float = Field(default=0.015, gt=0.0, le=0.1)
    min_ridge_length_px: float = Field(default=50.0, ge=10.0)
    min_ridge_edge_support: float = Field(default=0.35, ge=0.0, le=1.0)
    min_ridge_continuity_ratio: float = Field(default=0.60, ge=0.0, le=1.0)
    continuity_window_px: float = Field(default=20.0, ge=4.0)
    containment_margin_px: float = Field(default=15.0, ge=1.0)
    alignment_margin_px: float = Field(default=12.0, ge=1.0)
    max_allowed_oob_ratio: float = Field(default=0.05, gt=0.0, le=0.2)
    max_ridges: int = Field(default=4, ge=1, le=16)
    max_t_junctions: int = Field(default=8, ge=1, le=32)

    @model_validator(mode="after")
    def validate_relationships(self) -> "OcclusionEvidenceConfig":
        if self.containment_margin_px < self.alignment_margin_px:
            raise ValueError("containment_margin_px should be >= alignment_margin_px")
        return self


class OcclusionRidge(BaseContract):
    """One verified physical paper boundary ridge running across an overlapping document mask."""

    ridge_id: int = Field(ge=0)
    p1: DetectorPoint
    p2: DetectorPoint
    length_px: float = Field(gt=0.0)
    edge_support: float = Field(ge=0.0, le=1.0)
    continuity_ratio: float = Field(ge=0.0, le=1.0)
    angle_deg: float = Field(ge=0.0, lt=180.0)
    defect_depths: Tuple[float, float]
    is_verified: bool


class OcclusionEvidence(BaseContract):
    """Aggregate occlusion evidence across an image; bounded and typed."""

    evidence_version: str = OCCLUSION_EVIDENCE_VERSION
    has_overlapping_cues: bool = False
    verified_ridges: List[OcclusionRidge] = Field(default_factory=list)
    t_junctions: List[DetectorPoint] = Field(default_factory=list)
    diagnostics: Dict[str, Union[str, int, float, bool]] = Field(default_factory=dict)


def _angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 <= 1e-9 or n2 <= 1e-9:
        return 0.0
    cos_theta = float(np.dot(v1, v2) / (n1 * n2))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return float(math.degrees(math.acos(cos_theta)))


def _check_ridge_continuity(
    p1_src: DetectorPoint,
    p2_src: DetectorPoint,
    edge_map: EdgeMap,
    config: OcclusionEvidenceConfig,
) -> Tuple[float, float]:
    """Measure edge support and windowed continuity in detection space."""
    scale = edge_map.scale_factor
    p1_det = np.array([p1_src[0] * scale, p1_src[1] * scale], dtype=np.float64)
    p2_det = np.array([p2_src[0] * scale, p2_src[1] * scale], dtype=np.float64)

    length_det = float(np.linalg.norm(p2_det - p1_det))
    if length_det <= 1e-9:
        return 0.0, 0.0

    num_samples = max(10, int(round(length_det)))
    step_samples = np.linspace(0.0, 1.0, num_samples)
    edges = edge_map.edges
    h_det, w_det = edges.shape[:2]

    # Sample points along the segment with a small transverse tolerance (+-2 px)
    sample_hits = np.zeros(num_samples, dtype=bool)
    dx = p2_det[0] - p1_det[0]
    dy = p2_det[1] - p1_det[1]
    normal = np.array([-dy / length_det, dx / length_det])

    for i, t in enumerate(step_samples):
        center = p1_det + t * (p2_det - p1_det)
        found = False
        for offset in range(-2, 3):
            x = int(round(center[0] + normal[0] * offset))
            y = int(round(center[1] + normal[1] * offset))
            if 0 <= x < w_det and 0 <= y < h_det and edges[y, x] > 0:
                found = True
                break
        sample_hits[i] = found

    overall_support = float(np.mean(sample_hits)) if len(sample_hits) > 0 else 0.0

    # Continuity check: partition the segment into windows of size continuity_window_px
    window_det_px = max(2.0, config.continuity_window_px * scale)
    num_windows = max(1, int(math.ceil(length_det / window_det_px)))
    samples_per_window = max(1, num_samples // num_windows)

    supported_windows = 0
    for w_idx in range(num_windows):
        start_idx = w_idx * samples_per_window
        end_idx = min(num_samples, (w_idx + 1) * samples_per_window)
        if start_idx < end_idx and np.any(sample_hits[start_idx:end_idx]):
            supported_windows += 1

    continuity_ratio = float(supported_windows / num_windows) if num_windows > 0 else 0.0
    return overall_support, continuity_ratio


def _find_corner_ridges(
    pt1: DetectorPoint,
    pt2: DetectorPoint,
    edge_map: EdgeMap,
    policy: OcclusionEvidenceConfig,
    depth1: float,
    depth2: float,
    ridge_id_start: int,
) -> Tuple[List[OcclusionRidge], Optional[DetectorPoint]]:
    """Detect two-segment L-shaped occlusion ridge meeting at an internal corner."""
    scale = edge_map.scale_factor
    edges = edge_map.edges
    p1_det = np.array([pt1[0] * scale, pt1[1] * scale], dtype=np.float64)
    p2_det = np.array([pt2[0] * scale, pt2[1] * scale], dtype=np.float64)
    direct_dist_det = float(np.linalg.norm(p1_det - p2_det))
    if direct_dist_det < 1e-6:
        return [], None

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    tol = max(10.0, 15.0 * scale)

    for c in contours:
        pts = c[:, 0].astype(np.float64)
        d1 = np.linalg.norm(pts - p1_det, axis=1)
        d2 = np.linalg.norm(pts - p2_det, axis=1)
        if np.min(d1) <= tol and np.min(d2) <= tol:
            i1 = int(np.argmin(d1))
            i2 = int(np.argmin(d2))
            if abs(i1 - i2) >= 6:
                sub = pts[min(i1, i2) : max(i1, i2) + 1]
                path_len = float(np.sum(np.linalg.norm(np.diff(sub, axis=0), axis=1)))
                # An internal L-corner path has length <= 1.8 * direct_dist (sqrt(2) approx 1.41)
                if path_len <= 1.8 * direct_dist_det:
                    approx = cv2.approxPolyDP(sub.astype(np.float32), max(3.0, 5.0 * scale), False)
                    verts = approx.reshape(-1, 2)
                    if len(verts) >= 3:
                        for corner_det in verts[1:-1]:
                            c_src: DetectorPoint = (
                                float(corner_det[0] / scale),
                                float(corner_det[1] / scale),
                            )
                            v1 = np.array([pt1[0] - c_src[0], pt1[1] - c_src[1]], dtype=np.float64)
                            v2 = np.array([pt2[0] - c_src[0], pt2[1] - c_src[1]], dtype=np.float64)
                            angle = _angle_between_vectors(v1, v2)
                            if not (50.0 <= angle <= 130.0):
                                continue
                            dist1 = float(np.linalg.norm(v1))
                            dist2 = float(np.linalg.norm(v2))
                            if (
                                dist1 < policy.min_ridge_length_px
                                or dist2 < policy.min_ridge_length_px
                            ):
                                continue
                            s1, cont1 = _check_ridge_continuity(pt1, c_src, edge_map, policy)
                            s2, cont2 = _check_ridge_continuity(c_src, pt2, edge_map, policy)
                            if (
                                s1 >= policy.min_ridge_edge_support
                                and cont1 >= policy.min_ridge_continuity_ratio
                                and s2 >= policy.min_ridge_edge_support
                                and cont2 >= policy.min_ridge_continuity_ratio
                            ):
                                ang1 = math.degrees(
                                    math.atan2(abs(c_src[1] - pt1[1]), abs(c_src[0] - pt1[0]))
                                )
                                ang2 = math.degrees(
                                    math.atan2(abs(pt2[1] - c_src[1]), abs(pt2[0] - c_src[0]))
                                )
                                r1 = OcclusionRidge(
                                    ridge_id=ridge_id_start,
                                    p1=pt1,
                                    p2=c_src,
                                    length_px=round(dist1, 2),
                                    edge_support=round(s1, 4),
                                    continuity_ratio=round(cont1, 4),
                                    angle_deg=round(ang1, 2),
                                    defect_depths=(
                                        round(depth1, 2),
                                        round((depth1 + depth2) / 2.0, 2),
                                    ),
                                    is_verified=True,
                                )
                                r2 = OcclusionRidge(
                                    ridge_id=ridge_id_start + 1,
                                    p1=c_src,
                                    p2=pt2,
                                    length_px=round(dist2, 2),
                                    edge_support=round(s2, 4),
                                    continuity_ratio=round(cont2, 4),
                                    angle_deg=round(ang2, 2),
                                    defect_depths=(
                                        round((depth1 + depth2) / 2.0, 2),
                                        round(depth2, 2),
                                    ),
                                    is_verified=True,
                                )
                                return [r1, r2], c_src
    return [], None


def detect_occlusion_evidence(
    image: Union[LoadedImage, np.ndarray],
    mask: Optional[np.ndarray],
    *,
    edge_map: Optional[EdgeMap] = None,
    config: Optional[OcclusionEvidenceConfig] = None,
) -> OcclusionEvidence:
    """Extract physical paper boundary occlusion ridges and T-junctions."""
    policy = config or OcclusionEvidenceConfig()
    active_edge_map = edge_map if edge_map is not None else build_edge_map(image)

    if mask is None or int(mask.sum()) == 0:
        return OcclusionEvidence(
            diagnostics={"reason": "no_mask", "scaleFactor": active_edge_map.scale_factor}
        )

    binary_mask = (mask > 0).astype(np.uint8)
    image_height, image_width = binary_mask.shape[:2]
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return OcclusionEvidence(
            diagnostics={"reason": "no_contour", "scaleFactor": active_edge_map.scale_factor}
        )

    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        return OcclusionEvidence(
            diagnostics={"reason": "small_contour", "scaleFactor": active_edge_map.scale_factor}
        )

    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 3:
        return OcclusionEvidence(
            diagnostics={"reason": "no_hull", "scaleFactor": active_edge_map.scale_factor}
        )

    defects = cv2.convexityDefects(contour, hull)
    if defects is None or len(defects) == 0:
        return OcclusionEvidence(
            diagnostics={"reason": "no_defects", "scaleFactor": active_edge_map.scale_factor}
        )

    effective_min_depth = max(
        policy.min_defect_depth_px,
        policy.min_defect_depth_ratio * min(image_width, image_height),
    )

    deep_defects: List[Tuple[int, float, DetectorPoint]] = []
    for i in range(len(defects)):
        row = defects[i, 0] if defects.ndim == 3 else defects[i]
        f_idx = int(row[2])
        depth = float(row[3]) / 256.0
        if depth >= effective_min_depth:
            pt = (float(contour[f_idx][0][0]), float(contour[f_idx][0][1]))
            deep_defects.append((f_idx, depth, pt))

    if len(deep_defects) < 2:
        return OcclusionEvidence(
            diagnostics={
                "reason": "insufficient_defects",
                "deepDefectCount": len(deep_defects),
                "scaleFactor": active_edge_map.scale_factor,
            }
        )

    # Sort defects by depth descending
    deep_defects.sort(key=lambda d: -d[1])
    candidate_defects = deep_defects[: policy.max_t_junctions]

    contour_len = len(contour)
    diffs = np.diff(contour[:, 0], axis=0, append=contour[:1, 0])
    segment_lengths = np.linalg.norm(diffs, axis=1)
    total_arc_length = float(np.sum(segment_lengths))
    if total_arc_length <= 1e-6:
        return OcclusionEvidence(
            diagnostics={
                "reason": "degenerate_contour",
                "scaleFactor": active_edge_map.scale_factor,
            }
        )
    cumsum_arc = np.insert(np.cumsum(segment_lengths), 0, 0.0)

    verified_ridges: List[OcclusionRidge] = []
    t_junctions: List[DetectorPoint] = []
    ridge_id_counter = 0

    # Evaluate defect pairs
    min_dist_px = 0.20 * min(image_width, image_height)
    for i in range(len(candidate_defects)):
        for j in range(i + 1, len(candidate_defects)):
            idx1, depth1, pt1 = candidate_defects[i]
            idx2, depth2, pt2 = candidate_defects[j]

            # 1. Cyclic arc-length distance check: must be >= 15% of total contour perimeter
            arc_between = abs(cumsum_arc[idx1] - cumsum_arc[idx2])
            cyclic_dist = min(arc_between, total_arc_length - arc_between)
            if cyclic_dist < 0.15 * total_arc_length:
                continue

            # 2. Euclidean distance check
            dist = math.dist(pt1, pt2)
            if dist < min_dist_px or dist < policy.min_ridge_length_px:
                continue

            # 3. Symmetric T-junction angle check at both defect points (<= 170 deg indentation)
            step = max(3, contour_len // 50)
            prev1 = contour[(idx1 - step) % contour_len][0]
            next1 = contour[(idx1 + step) % contour_len][0]
            v_in1 = np.array(prev1 - contour[idx1][0], dtype=np.float64)
            v_out1 = np.array(next1 - contour[idx1][0], dtype=np.float64)
            corner_angle1 = _angle_between_vectors(v_in1, v_out1)
            if corner_angle1 > 170.0:  # Not a significant indentation
                continue

            prev2 = contour[(idx2 - step) % contour_len][0]
            next2 = contour[(idx2 + step) % contour_len][0]
            v_in2 = np.array(prev2 - contour[idx2][0], dtype=np.float64)
            v_out2 = np.array(next2 - contour[idx2][0], dtype=np.float64)
            corner_angle2 = _angle_between_vectors(v_in2, v_out2)
            if corner_angle2 > 170.0:  # Not a significant indentation
                continue

            # Verify ridge direction is transverse to contour tangents (>= 25 deg) at both endpoints
            v_ridge = np.array([pt2[0] - pt1[0], pt2[1] - pt1[1]], dtype=np.float64)
            angle_with_in1 = _angle_between_vectors(v_ridge, v_in1)
            angle_with_out1 = _angle_between_vectors(v_ridge, v_out1)
            angle_with_in2 = _angle_between_vectors(-v_ridge, v_in2)
            angle_with_out2 = _angle_between_vectors(-v_ridge, v_out2)
            if (
                angle_with_in1 < 25.0
                or angle_with_out1 < 25.0
                or angle_with_in2 < 25.0
                or angle_with_out2 < 25.0
            ):
                continue

            # 4. Measure edge continuity and support along ridge in detection space
            support, continuity = _check_ridge_continuity(pt1, pt2, active_edge_map, policy)
            if (
                support >= policy.min_ridge_edge_support
                and continuity >= policy.min_ridge_continuity_ratio
            ):
                angle_deg = math.degrees(math.atan2(abs(pt2[1] - pt1[1]), abs(pt2[0] - pt1[0])))
                verified_ridges.append(
                    OcclusionRidge(
                        ridge_id=ridge_id_counter,
                        p1=pt1,
                        p2=pt2,
                        length_px=round(dist, 2),
                        edge_support=round(support, 4),
                        continuity_ratio=round(continuity, 4),
                        angle_deg=round(angle_deg, 2),
                        defect_depths=(round(depth1, 2), round(depth2, 2)),
                        is_verified=True,
                    )
                )
                ridge_id_counter += 1
                if pt1 not in t_junctions:
                    t_junctions.append(pt1)
                if pt2 not in t_junctions:
                    t_junctions.append(pt2)
            else:
                # 5. Fallback check: Two-segment L-shaped corner ridge
                corner_ridges, corner_pt = _find_corner_ridges(
                    pt1,
                    pt2,
                    active_edge_map,
                    policy,
                    depth1,
                    depth2,
                    ridge_id_counter,
                )
                if corner_ridges and corner_pt is not None:
                    verified_ridges.extend(corner_ridges)
                    ridge_id_counter += len(corner_ridges)
                    for pt_to_add in (pt1, pt2, corner_pt):
                        if pt_to_add not in t_junctions:
                            t_junctions.append(pt_to_add)

            if len(verified_ridges) >= policy.max_ridges:
                break
        if len(verified_ridges) >= policy.max_ridges:
            break

    has_overlapping = len(verified_ridges) > 0
    return OcclusionEvidence(
        has_overlapping_cues=has_overlapping,
        verified_ridges=verified_ridges,
        t_junctions=t_junctions,
        diagnostics={
            "deepDefectCount": len(deep_defects),
            "verifiedRidgeCount": len(verified_ridges),
            "effectiveMinDepth": round(effective_min_depth, 2),
            "scaleFactor": active_edge_map.scale_factor,
        },
    )


def _coerce_points(corners: Any) -> List[DetectorPoint]:
    if isinstance(corners, CanonicalCorners):
        return corners.as_list()
    if hasattr(corners, "as_list"):
        return corners.as_list()
    if hasattr(corners, "corners"):
        c = corners.corners
        if hasattr(c, "as_list"):
            return c.as_list()
        if hasattr(c, "points"):
            return c.points
    if hasattr(corners, "points"):
        return corners.points
    return [(float(p[0]), float(p[1])) for p in corners]


def evaluate_candidate_occlusion(
    candidate_corners: Any,
    occlusion: OcclusionEvidence,
    *,
    config: Optional[OcclusionEvidenceConfig] = None,
) -> Dict[str, Any]:
    """Evaluate segment-level containment and alignment for one candidate quad."""
    policy = config or OcclusionEvidenceConfig()
    if not occlusion.has_overlapping_cues or not occlusion.verified_ridges:
        return {
            "enclosed_occlusion_ridges": 0,
            "enclosed_occlusion_length": 0.0,
            "aligns_with_occlusion_ridge": False,
            "has_overlapping_cues": False,
        }

    points = _coerce_points(candidate_corners)
    poly = np.array(points, dtype=np.float32)

    enclosed_count = 0
    enclosed_length = 0.0
    aligns = False

    for ridge in occlusion.verified_ridges:
        p1 = np.array(ridge.p1, dtype=np.float64)
        p2 = np.array(ridge.p2, dtype=np.float64)

        # Sample 20 points along the segment (excluding extreme ends)
        num_samples = 20
        samples = [p1 + t * (p2 - p1) for t in np.linspace(0.05, 0.95, num_samples)]
        dists = [
            float(cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), measureDist=True))
            for p in samples
        ]

        # Points strictly inside by at least containment_margin_px
        inside_count = sum(1 for d in dists if d >= policy.containment_margin_px)
        inside_ratio = inside_count / float(num_samples)

        # Points close to polygon edge (alignment)
        aligned_count = sum(1 for d in dists if abs(d) <= policy.alignment_margin_px)
        aligned_ratio = aligned_count / float(num_samples)

        if inside_ratio >= 0.70:
            enclosed_count += 1
            enclosed_length += ridge.length_px * inside_ratio
        elif aligned_ratio >= 0.70:
            aligns = True

    return {
        "enclosed_occlusion_ridges": enclosed_count,
        "enclosed_occlusion_length": round(enclosed_length, 2),
        "aligns_with_occlusion_ridge": aligns,
        "has_overlapping_cues": occlusion.has_overlapping_cues,
    }


DEFAULT_OCCLUSION_LENGTH_SCALE_RATIO: float = 0.25


def compute_occlusion_penalty_ratio(
    candidate_corners: Any,
    enclosed_length: float,
    scale_ratio: float = DEFAULT_OCCLUSION_LENGTH_SCALE_RATIO,
) -> float:
    """Calculate the ratio of enclosed occlusion ridge length relative to candidate scale.

    Formula:
      perimeter = polygon perimeter of candidate corners
      denom = max(1e-6, scale_ratio * perimeter)
      penalty_ratio = min(1.0, enclosed_length / denom)
    """
    if enclosed_length <= 0.0:
        return 0.0
    points = _coerce_points(candidate_corners)
    poly = np.array(points, dtype=np.float32)
    perimeter = float(cv2.arcLength(poly, closed=True))
    denom = max(1e-6, scale_ratio * perimeter)
    return float(min(1.0, max(0.0, enclosed_length / denom)))


__all__ = [
    "DEFAULT_OCCLUSION_LENGTH_SCALE_RATIO",
    "OCCLUSION_EVIDENCE_VERSION",
    "OcclusionEvidence",
    "OcclusionEvidenceConfig",
    "OcclusionRidge",
    "compute_occlusion_penalty_ratio",
    "detect_occlusion_evidence",
    "evaluate_candidate_occlusion",
]
