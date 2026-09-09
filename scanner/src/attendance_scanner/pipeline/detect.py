"""Document boundary detection using classical OpenCV contour analysis.

This module is the second stage of the document scanning pipeline. It operates on a
detection-size copy of the source image to identify the 4-corner document contour,
validates its geometric quadrilateral properties (convexity, side lengths, interior angles,
area ratio), orders the vertices into canonical [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
order, and scales the coordinates back to the original source image bounds.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from pydantic import Field, model_validator

from attendance_scanner.contracts import BaseContract
from attendance_scanner.pipeline.load import LoadedImage


class DetectionRejectionReason(str, Enum):
    """Canonical rejection reasons for candidate document quadrilaterals."""

    DOCUMENT_CLIPPED = "DOCUMENT_CLIPPED"
    UNSAFE_GEOMETRY = "UNSAFE_GEOMETRY"
    LOW_BOUNDARY_SUPPORT = "LOW_BOUNDARY_SUPPORT"
    INNER_TABLE_COLLAPSE = "INNER_TABLE_COLLAPSE"


class DetectionConfig(BaseContract):
    """Configurable thresholds and parameters for document contour detection."""

    max_dimension: int = Field(default=1200, ge=100, le=8000)
    blur_kernel_size: int = Field(default=5, ge=1, le=31)
    canny_threshold1: int = Field(default=50, ge=0, le=255)
    canny_threshold2: int = Field(default=150, ge=0, le=255)
    morph_kernel_size: int = Field(default=3, ge=1, le=31)
    morph_iterations: int = Field(default=2, ge=0, le=10)
    min_area_ratio: float = Field(default=0.05, gt=0.0, lt=1.0)
    max_area_ratio: float = Field(default=0.98, gt=0.0, le=1.0)
    approx_epsilon_ratios: List[float] = Field(default_factory=lambda: [0.015, 0.02, 0.03, 0.04])
    min_side_ratio: float = Field(default=0.05, gt=0.0, lt=1.0)
    min_angle_deg: float = Field(default=45.0, ge=0.0, lt=180.0)
    max_angle_deg: float = Field(default=135.0, gt=0.0, le=180.0)
    paper_brightness_percentile: float = Field(default=75.0, ge=50.0, le=95.0)
    paper_min_threshold: int = Field(default=100, ge=0, le=255)
    paper_max_threshold: int = Field(default=230, ge=0, le=255)
    paper_morph_kernel_size: int = Field(default=61, ge=9, le=201)
    paper_illum_kernel_size: int = Field(default=51, ge=9, le=201)
    paper_candidate_score_bonus: float = Field(default=1.35, ge=1.0, le=1.5)
    paper_min_solidity: float = Field(default=0.75, ge=0.5, le=1.0)
    paper_min_ink_ratio: float = Field(default=0.01, ge=0.0, le=1.0)
    boundary_margin_ratio: float = Field(default=0.015, ge=0.0, le=0.1)
    table_downrank_factor: float = Field(default=0.6, ge=0.1, le=1.0)
    table_inner_ink_threshold: float = Field(default=0.02, ge=0.0, le=0.5)

    @model_validator(mode="after")
    def validate_threshold_relationships(self) -> "DetectionConfig":
        """Validate logical relationships between configuration parameters."""
        if self.min_area_ratio >= self.max_area_ratio:
            raise ValueError(
                f"min_area_ratio ({self.min_area_ratio}) must be strictly less than "
                f"max_area_ratio ({self.max_area_ratio})"
            )
        if self.min_angle_deg >= self.max_angle_deg:
            raise ValueError(
                f"min_angle_deg ({self.min_angle_deg}) must be strictly less than "
                f"max_angle_deg ({self.max_angle_deg})"
            )
        if self.canny_threshold1 > self.canny_threshold2:
            raise ValueError(
                f"canny_threshold1 ({self.canny_threshold1}) must be <= "
                f"canny_threshold2 ({self.canny_threshold2})"
            )
        return self


class DetectionResult(BaseContract):
    """Result of document boundary detection containing 4 corners in original coordinates."""

    detected: bool = True
    accepted: bool = True
    clipped: bool = False
    rejection_reason: Optional[DetectionRejectionReason] = None
    # Ordered clockwise: [TL, TR, BR, BL]
    corners: List[Tuple[float, float]] = Field(default_factory=list)
    confidence: float = 0.0  # Quality score between 0.0 and 1.0
    area_ratio: float = 0.0  # Detected document area relative to original image area
    scale_factor: float = 1.0  # Scale factor applied to detection copy
    diagnostics: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_detection_state(self) -> "DetectionResult":
        """Keep accepted/rejected/clipped states internally consistent."""
        if self.accepted and self.clipped:
            raise ValueError("A clipped detection cannot be accepted")
        if self.accepted and self.rejection_reason is not None:
            raise ValueError("An accepted detection cannot have a rejection reason")
        if not self.accepted and self.rejection_reason is None:
            raise ValueError("A rejected detection must have a rejection reason")
        if self.accepted and len(self.corners) != 4:
            raise ValueError("An accepted detection must contain exactly four corners")
        return self

    @property
    def corners_array(self) -> np.ndarray:
        """Return corners as a (4, 2) float32 NumPy array."""
        if not self.corners:
            return np.empty((0, 2), dtype=np.float32)
        return np.array(self.corners, dtype=np.float32)


@dataclass(frozen=True)
class BoundaryCollisionResult:
    """Diagnostic outcome of border collision assessment for candidate corners."""

    is_clipped: bool
    reasons: List[str]
    out_of_bounds_count: int
    border_touch_edges: List[str]
    min_border_distance: float


def assess_boundary_collision(
    corners: Union[np.ndarray, List[Tuple[float, float]]],
    image_size: Tuple[int, int],  # (width, height)
    *,
    margin_ratio: float = 0.015,
    paper_mask: Optional[np.ndarray] = None,
) -> BoundaryCollisionResult:
    """Pure helper to assess whether 4 candidate corners collide with or exceed image boundaries.

    A candidate is judged as clipped if:
    1. Any corner point lies strictly outside image bounds [0, width] x [0, height]
       (accounting for floating point epsilon).
    2. Any corner point lies within the margin AND an edge runs along that image border.
    3. An edge lies within the margin AND the paper mask continues across that border.
    """
    w, h = image_size
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    reasons: List[str] = []
    border_touch_edges: List[str] = []

    margin_x = float(w * margin_ratio)
    margin_y = float(h * margin_ratio)

    eps = 0.5
    out_of_bounds = 0
    min_dist = float("inf")

    for pt in pts:
        x, y = float(pt[0]), float(pt[1])
        d_left = x
        d_right = w - x
        d_top = y
        d_bottom = h - y
        pt_min_dist = min(d_left, d_right, d_top, d_bottom)
        min_dist = min(min_dist, pt_min_dist)

        if x < -eps or x > w + eps or y < -eps or y > h + eps:
            out_of_bounds += 1

    if out_of_bounds > 0:
        reasons.append(f"{out_of_bounds}_corners_out_of_bounds")

    # Check edges running along border
    num_pts = len(pts)
    for i in range(num_pts):
        p1 = pts[i]
        p2 = pts[(i + 1) % num_pts]

        if p1[1] <= margin_y and p2[1] <= margin_y:
            border_touch_edges.append("top")
            if min(p1[1], p2[1]) <= 1.0:
                reasons.append("edge_touches_top_border")
        if p1[1] >= h - margin_y and p2[1] >= h - margin_y:
            border_touch_edges.append("bottom")
            if max(p1[1], p2[1]) >= h - 1.0:
                reasons.append("edge_touches_bottom_border")
        if p1[0] <= margin_x and p2[0] <= margin_x:
            border_touch_edges.append("left")
            if min(p1[0], p2[0]) <= 1.0:
                reasons.append("edge_touches_left_border")
        if p1[0] >= w - margin_x and p2[0] >= w - margin_x:
            border_touch_edges.append("right")
            if max(p1[0], p2[0]) >= w - 1.0:
                reasons.append("edge_touches_right_border")

    if paper_mask is not None and paper_mask.shape[:2] == (h, w):
        border_masks = {
            "top": paper_mask[0, :],
            "bottom": paper_mask[-1, :],
            "left": paper_mask[:, 0],
            "right": paper_mask[:, -1],
        }
        for edge_name, border_strip in border_masks.items():
            if border_strip.size > 0:
                white_ratio = float(np.count_nonzero(border_strip) / border_strip.size)
                if white_ratio > 0.08 and edge_name in border_touch_edges:
                    reasons.append(f"paper_mask_exceeds_{edge_name}_border")

        # Attribute border-crossing paper to the candidate's connected component.
        # This catches an inner table surrounded by a clipped sheet while avoiding
        # unrelated small bright fragments elsewhere along the image border.
        if pts.size > 0:
            _, labels, stats, _ = cv2.connectedComponentsWithStats(
                (paper_mask > 0).astype(np.uint8), connectivity=8
            )
            center = np.mean(pts, axis=0)
            center_x = int(np.clip(round(float(center[0])), 0, w - 1))
            center_y = int(np.clip(round(float(center[1])), 0, h - 1))
            component_label = int(labels[center_y, center_x])
            if component_label > 0:
                component_area = float(stats[component_label, cv2.CC_STAT_AREA])
                if component_area >= paper_mask.size * 0.05:
                    component_left = int(stats[component_label, cv2.CC_STAT_LEFT])
                    component_top = int(stats[component_label, cv2.CC_STAT_TOP])
                    component_width = int(stats[component_label, cv2.CC_STAT_WIDTH])
                    component_height = int(stats[component_label, cv2.CC_STAT_HEIGHT])
                    component_edges = {
                        "left": component_left <= 0,
                        "top": component_top <= 0,
                        "right": component_left + component_width >= w,
                        "bottom": component_top + component_height >= h,
                    }
                    for edge_name, touches_border in component_edges.items():
                        if touches_border:
                            border_touch_edges.append(edge_name)
                            reasons.append(f"candidate_paper_component_touches_{edge_name}_border")

    is_clipped = len(reasons) > 0
    return BoundaryCollisionResult(
        is_clipped=is_clipped,
        reasons=list(dict.fromkeys(reasons)),
        out_of_bounds_count=out_of_bounds,
        border_touch_edges=list(dict.fromkeys(border_touch_edges)),
        min_border_distance=min_dist,
    )


def _evaluate_outer_margin_ink(
    gray: np.ndarray,
    quad: np.ndarray,
    det_w: int,
    det_h: int,
    paper_threshold: float = 120.0,
    ink_threshold: int = 100,
) -> float:
    """Estimate ink density on paper in the margin band immediately outside the candidate quad.

    An inner table frame is surrounded by paper that contains headers/text.
    A true outer paper boundary is surrounded by desk/background (no paper color support).
    """
    poly = quad.astype(np.int32).reshape(-1, 1, 2)
    mask_inside = np.zeros((det_h, det_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask_inside, poly, 255)

    margin_k = max(9, int(round(min(det_w, det_h) * 0.03)))
    if margin_k % 2 == 0:
        margin_k += 1
    margin_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (margin_k, margin_k))
    mask_expanded = cv2.dilate(mask_inside, margin_kernel)
    mask_margin = cv2.subtract(mask_expanded, mask_inside)

    margin_pixels = int(np.count_nonzero(mask_margin))
    if margin_pixels == 0:
        return 0.0

    # Paper color support: pixels in the outer margin that have bright paper color
    paper_pixels = int(np.count_nonzero((gray >= paper_threshold * 0.75) & (mask_margin > 0)))
    paper_support = float(paper_pixels / margin_pixels)
    # If outer margin has minimal paper support (mostly dark desk/background),
    # this candidate is an outer document edge on a desk, not an inner table.
    if paper_support < 0.20:
        return 0.0

    ink_pixels = int(np.count_nonzero((gray < ink_threshold) & (mask_margin > 0)))
    return float(ink_pixels / margin_pixels)


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Sort 4 corner points into canonical clockwise order: [TL, TR, BR, BL].

    The ordering algorithm:
    - Top-Left (TL): point with the minimum sum of coordinates (x + y)
    - Bottom-Right (BR): point with the maximum sum of coordinates (x + y)
    - Top-Right (TR): point with the minimum difference (y - x)
    - Bottom-Left (BL): point with the maximum difference (y - x)

    Args:
        pts: Array of 4 points with shape (4, 2) or (4, 1, 2).

    Returns:
        Array of shape (4, 2) ordered [TL, TR, BR, BL] with dtype np.float32.
    """
    points = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    # Sum of coordinates: x + y
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]  # Top-left
    rect[2] = points[np.argmax(s)]  # Bottom-right

    # Difference of coordinates: y - x
    diff = points[:, 1] - points[:, 0]
    rect[1] = points[np.argmin(diff)]  # Top-right
    rect[3] = points[np.argmax(diff)]  # Bottom-left

    return rect


def _calculate_angle_deg(p_prev: np.ndarray, p_curr: np.ndarray, p_next: np.ndarray) -> float:
    """Calculate the interior angle in degrees at p_curr formed by vectors to p_prev and p_next."""
    v1 = p_prev - p_curr
    v2 = p_next - p_curr

    norm1 = float(np.linalg.norm(v1))
    norm2 = float(np.linalg.norm(v2))

    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0

    cosine = float(np.dot(v1, v2) / (norm1 * norm2))
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def _validate_quadrilateral(
    ordered_corners: np.ndarray,
    det_w: int,
    det_h: int,
    config: DetectionConfig,
) -> Tuple[bool, float, Dict[str, Any]]:
    """Validate that a 4-point polygon is a plausible document quadrilateral.

    Checks:
    1. Convexity.
    2. Area ratio within [min_area_ratio, max_area_ratio].
    3. Minimum side lengths.
    4. Interior angles at each vertex within [min_angle_deg, max_angle_deg].
    5. Corners within detection image boundaries.

    Returns:
        Tuple of (is_valid, confidence_score, diagnostics_dict).
    """
    pts = ordered_corners.astype(np.float32)
    poly_int = pts.reshape(4, 1, 2).astype(np.int32)

    # 1. Convexity
    if not cv2.isContourConvex(poly_int):
        return False, 0.0, {"reason": "non_convex"}

    # 2. Area ratio
    contour_area = float(cv2.contourArea(pts))
    img_area = float(det_w * det_h)
    area_ratio = contour_area / img_area

    if area_ratio < config.min_area_ratio or area_ratio > config.max_area_ratio:
        return False, 0.0, {"reason": "area_ratio_out_of_bounds", "area_ratio": area_ratio}

    # 3. Side lengths
    # Points ordered: TL (0), TR (1), BR (2), BL (3)
    d_top = float(np.linalg.norm(pts[1] - pts[0]))
    d_right = float(np.linalg.norm(pts[2] - pts[1]))
    d_bottom = float(np.linalg.norm(pts[3] - pts[2]))
    d_left = float(np.linalg.norm(pts[0] - pts[3]))

    min_dimension = min(det_w, det_h)
    min_required_side = min_dimension * config.min_side_ratio

    if min(d_top, d_right, d_bottom, d_left) < min_required_side:
        return False, 0.0, {"reason": "side_too_short", "sides": [d_top, d_right, d_bottom, d_left]}

    # 4. Angles
    angles: List[float] = []
    for i in range(4):
        prev_pt = pts[(i - 1) % 4]
        curr_pt = pts[i]
        next_pt = pts[(i + 1) % 4]
        ang = _calculate_angle_deg(prev_pt, curr_pt, next_pt)
        if ang < config.min_angle_deg or ang > config.max_angle_deg:
            return False, 0.0, {"reason": "angle_out_of_bounds", "corner_index": i, "angle": ang}
        angles.append(ang)

    # 5. Boundary sanity (-5 to det + 5 margin)
    margin = 5.0
    for pt in pts:
        x, y = float(pt[0]), float(pt[1])
        if x < -margin or x > det_w + margin or y < -margin or y > det_h + margin:
            return False, 0.0, {"reason": "corner_out_of_image_bounds", "point": [x, y]}

    # Compute quality confidence score (0.0 to 1.0)
    # - Opposite side ratio symmetry
    ratio_tb = min(d_top, d_bottom) / max(d_top, d_bottom, 1e-6)
    ratio_lr = min(d_left, d_right) / max(d_left, d_right, 1e-6)
    side_symmetry_score = 0.5 * (ratio_tb + ratio_lr)

    # - Orthogonality score: how close angles are to 90 degrees
    angle_deviations = [abs(a - 90.0) for a in angles]
    mean_deviation = sum(angle_deviations) / 4.0
    orthogonality_score = max(0.0, 1.0 - (mean_deviation / 45.0))

    # - Area score: penalize extremes
    area_score = min(1.0, area_ratio / 0.3) if area_ratio < 0.3 else 1.0

    confidence = float(0.4 * side_symmetry_score + 0.4 * orthogonality_score + 0.2 * area_score)
    confidence = max(0.0, min(1.0, confidence))

    diagnostics = {
        "area_ratio": area_ratio,
        "side_lengths": [d_top, d_right, d_bottom, d_left],
        "angles": angles,
        "side_symmetry_score": side_symmetry_score,
        "orthogonality_score": orthogonality_score,
    }

    return True, confidence, diagnostics


def detect_document_boundary(
    image: Union[np.ndarray, LoadedImage],
    config: Optional[DetectionConfig] = None,
) -> Optional[DetectionResult]:
    """Detect the 4-corner document boundary on an image using classical OpenCV.

    Pipeline steps:
    1. Extract raw OpenCV BGR array from ndarray or LoadedImage.
    2. Scale down to detection copy (bounded by config.max_dimension, never upscaled).
    3. Convert to grayscale, apply Gaussian blur, Canny edge detection, and morphology.
    4. Find candidate contours and sort by descending area.
    5. Attempt polygonal approximation (approxPolyDP) across epsilon ratios.
    6. Validate candidate geometry (convexity, side lengths, angles, boundary constraints).
    7. Rank candidates and pick the highest-confidence quadrilateral.
    8. Map corners back to original image space and reject candidates that collide
       with the source image boundaries; rejected points are never clamped for warping.

    Args:
        image: Source OpenCV BGR array (shape H, W, 3) or LoadedImage instance.
        config: Optional detection configuration. Uses defaults if omitted.

    Returns:
        DetectionResult if a valid document boundary was detected, or None if no plausible
        document quadrilateral was found. Never throws an exception on detection failure.
    """
    if config is None:
        config = DetectionConfig()
    elif isinstance(config, dict):
        config = DetectionConfig(**config)

    # 1. Extract BGR image
    if isinstance(image, LoadedImage):
        bgr = image.image
    elif isinstance(image, np.ndarray):
        bgr = image
    else:
        return None

    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        return None

    orig_h, orig_w = bgr.shape[:2]
    if orig_h < 10 or orig_w < 10:
        return None

    try:
        # 2. Scale down to detection size (never upscale)
        max_dim = max(orig_w, orig_h)
        if max_dim > config.max_dimension:
            scale_factor = float(config.max_dimension / max_dim)
            det_w = int(round(orig_w * scale_factor))
            det_h = int(round(orig_h * scale_factor))
            det_image = cv2.resize(bgr, (det_w, det_h), interpolation=cv2.INTER_AREA)
        else:
            scale_factor = 1.0
            det_w, det_h = orig_w, orig_h
            det_image = bgr.copy()  # Ensure input is never mutated

        # 3. Preprocessing: grayscale, blur, edge detection, morphology
        gray = cv2.cvtColor(det_image, cv2.COLOR_BGR2GRAY)
        ksize = config.blur_kernel_size
        if ksize % 2 == 0:
            ksize += 1
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)

        # Multi-pass edge extraction: Canny edges + Otsu threshold fallback
        edges = cv2.Canny(blurred, config.canny_threshold1, config.canny_threshold2)
        morph_k = cv2.getStructuringElement(
            cv2.MORPH_RECT, (config.morph_kernel_size, config.morph_kernel_size)
        )
        dilated = cv2.dilate(edges, morph_k, iterations=config.morph_iterations)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, morph_k)

        # 4. Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour_list = list(contours)

        # If few contours found, try Otsu thresholding fallback for high-contrast white pages
        if len(contour_list) < 2:
            _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            otsu_contours, _ = cv2.findContours(otsu_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contour_list.extend(otsu_contours)

        # 5. Sort contours by area descending
        contour_list.sort(key=cv2.contourArea, reverse=True)

        img_area = float(det_w * det_h)
        min_area = img_area * config.min_area_ratio
        max_area = img_area * config.max_area_ratio

        candidates: List[
            Tuple[float, np.ndarray, float, Dict[str, Any], BoundaryCollisionResult]
        ] = []
        rejected_candidates: List[
            Tuple[
                float,
                np.ndarray,
                float,
                Dict[str, Any],
                BoundaryCollisionResult,
                DetectionRejectionReason,
            ]
        ] = []

        # Baseline paper threshold from blurred image percentile
        paper_threshold = float(
            np.clip(
                np.percentile(blurred, config.paper_brightness_percentile) - 1.0,
                config.paper_min_threshold,
                config.paper_max_threshold,
            )
        )
        _, paper_mask = cv2.threshold(blurred, paper_threshold, 255, cv2.THRESH_BINARY)

        # Optional illumination normalization for shadow compensation on detection mask only:
        if config.paper_illum_kernel_size > 0:
            k_illum = config.paper_illum_kernel_size
            if k_illum % 2 == 0:
                k_illum += 1
            scale_ref = min(det_w, det_h) / 900.0
            effective_k = max(9, int(round(k_illum * scale_ref)))
            if effective_k % 2 == 0:
                effective_k += 1
            illum_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (effective_k, effective_k))
            bg_est = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, illum_kernel)
            # Floor to prevent dividing dark desk by itself (which makes desk 255)
            dark_floor = max(int(paper_threshold * 0.85), config.paper_min_threshold)
            bg_safe = np.maximum(bg_est, dark_floor)
            norm_gray = cv2.divide(blurred, bg_safe, scale=255)
            norm_gray[blurred < dark_floor] = 0
            _, shadow_mask = cv2.threshold(norm_gray, 220, 255, cv2.THRESH_BINARY)
            shadow_mask[blurred < dark_floor] = 0
            paper_mask = cv2.bitwise_or(paper_mask, shadow_mask)
        paper_kernel_size = config.paper_morph_kernel_size
        scale_ref = min(det_w, det_h) / 900.0
        effective_paper_k = max(7, int(round(paper_kernel_size * scale_ref)))
        if effective_paper_k % 2 == 0:
            effective_paper_k += 1
        paper_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (effective_paper_k, effective_paper_k)
        )
        paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_CLOSE, paper_kernel)
        paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_OPEN, paper_kernel)
        paper_contours, _ = cv2.findContours(paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for paper_contour in sorted(paper_contours, key=cv2.contourArea, reverse=True):
            paper_area = float(cv2.contourArea(paper_contour))
            if paper_area < min_area:
                break
            if paper_area > max_area:
                continue
            paper_hull = cv2.convexHull(paper_contour)
            paper_hull_area = float(cv2.contourArea(paper_hull))
            paper_solidity = paper_area / max(paper_hull_area, 1e-6)
            if paper_solidity < config.paper_min_solidity:
                continue
            paper_region_mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.drawContours(paper_region_mask, [paper_contour], -1, 255, -1)
            paper_ink_ratio = float(
                ((gray < config.paper_min_threshold) & (paper_region_mask > 0)).sum()
                / max((paper_region_mask > 0).sum(), 1)
            )
            if paper_ink_ratio < config.paper_min_ink_ratio:
                continue
            paper_perimeter = cv2.arcLength(paper_hull, True)
            if paper_perimeter < 1e-6:
                continue
            min_rect_raw = cv2.boxPoints(cv2.minAreaRect(paper_contour)).astype(np.float32)
            min_rect = order_corners(min_rect_raw)
            # Assess boundary collision without clamping
            rect_collision = assess_boundary_collision(
                min_rect,
                (det_w, det_h),
                margin_ratio=config.boundary_margin_ratio,
                paper_mask=paper_mask,
            )

            min_rect_valid, min_rect_conf, min_rect_diag = _validate_quadrilateral(
                min_rect, det_w, det_h, config
            )
            if not min_rect_valid and rect_collision.is_clipped:
                rejected_candidates.append(
                    (
                        0.0,
                        min_rect,
                        max(float(cv2.contourArea(min_rect)) / img_area, 0.0),
                        {
                            **min_rect_diag,
                            "candidate_source": "paper_min_area_rect",
                            "is_clipped": True,
                            "collision_reasons": rect_collision.reasons,
                        },
                        rect_collision,
                        DetectionRejectionReason.DOCUMENT_CLIPPED,
                    )
                )
            if min_rect_valid:
                rect_area = float(cv2.contourArea(min_rect))
                fill_ratio = paper_area / max(rect_area, 1e-6)
                if fill_ratio < 0.85:
                    # Non-convex / hollow shape (e.g. L-shape) has low bounding box fill ratio
                    continue
                min_rect_area_ratio = float(rect_area / img_area)
                outer_ink = _evaluate_outer_margin_ink(
                    gray, min_rect, det_w, det_h, paper_threshold=paper_threshold
                )
                table_penalty = 1.0
                if outer_ink > config.table_inner_ink_threshold:
                    table_penalty = max(
                        config.table_downrank_factor,
                        1.0 - (outer_ink / 0.15) * (1.0 - config.table_downrank_factor),
                    )
                min_rect_diag = {
                    **min_rect_diag,
                    "candidate_source": "paper_min_area_rect",
                    "paper_threshold": paper_threshold,
                    "paper_solidity": paper_solidity,
                    "paper_ink_ratio": paper_ink_ratio,
                    "outer_margin_ink": outer_ink,
                    "table_penalty": table_penalty,
                    "is_clipped": rect_collision.is_clipped,
                    "collision_reasons": rect_collision.reasons,
                }
                min_rect_score = min_rect_conf * (0.6 + 0.4 * min_rect_area_ratio) * table_penalty
                if outer_ink > config.table_inner_ink_threshold:
                    min_rect_diag["rejection_reason"] = (
                        DetectionRejectionReason.INNER_TABLE_COLLAPSE.value
                    )
                    rejected_candidates.append(
                        (
                            min_rect_score,
                            min_rect,
                            min_rect_area_ratio,
                            min_rect_diag,
                            rect_collision,
                            DetectionRejectionReason.INNER_TABLE_COLLAPSE,
                        )
                    )
                else:
                    candidates.append(
                        (
                            min_rect_score,
                            min_rect,
                            min_rect_area_ratio,
                            min_rect_diag,
                            rect_collision,
                        )
                    )

            for eps_ratio in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08):
                paper_approx = cv2.approxPolyDP(paper_hull, eps_ratio * paper_perimeter, True)
                if len(paper_approx) != 4:
                    continue
                ordered = order_corners(paper_approx)
                approx_collision = assess_boundary_collision(
                    ordered,
                    (det_w, det_h),
                    margin_ratio=config.boundary_margin_ratio,
                    paper_mask=paper_mask,
                )
                is_valid, conf, diag = _validate_quadrilateral(ordered, det_w, det_h, config)
                if not is_valid and approx_collision.is_clipped:
                    rejected_candidates.append(
                        (
                            0.0,
                            ordered,
                            max(float(cv2.contourArea(ordered)) / img_area, 0.0),
                            {
                                **diag,
                                "candidate_source": "paper_mask",
                                "is_clipped": True,
                                "collision_reasons": approx_collision.reasons,
                            },
                            approx_collision,
                            DetectionRejectionReason.DOCUMENT_CLIPPED,
                        )
                    )
                if is_valid:
                    area_ratio = float(cv2.contourArea(ordered) / img_area)
                    outer_ink = _evaluate_outer_margin_ink(
                        gray, ordered, det_w, det_h, paper_threshold=paper_threshold
                    )
                    table_penalty = 1.0
                    if outer_ink > config.table_inner_ink_threshold:
                        table_penalty = max(
                            config.table_downrank_factor,
                            1.0 - (outer_ink / 0.15) * (1.0 - config.table_downrank_factor),
                        )
                    diag = {
                        **diag,
                        "candidate_source": "paper_mask",
                        "paper_threshold": paper_threshold,
                        "paper_solidity": paper_solidity,
                        "paper_ink_ratio": paper_ink_ratio,
                        "outer_margin_ink": outer_ink,
                        "table_penalty": table_penalty,
                        "is_clipped": approx_collision.is_clipped,
                        "collision_reasons": approx_collision.reasons,
                    }
                    score = (
                        conf
                        * (0.6 + 0.4 * area_ratio)
                        * config.paper_candidate_score_bonus
                        * table_penalty
                    )
                    if outer_ink > config.table_inner_ink_threshold:
                        diag["rejection_reason"] = (
                            DetectionRejectionReason.INNER_TABLE_COLLAPSE.value
                        )
                        rejected_candidates.append(
                            (
                                score,
                                ordered,
                                area_ratio,
                                diag,
                                approx_collision,
                                DetectionRejectionReason.INNER_TABLE_COLLAPSE,
                            )
                        )
                    else:
                        candidates.append((score, ordered, area_ratio, diag, approx_collision))
                    break

        # 6. Evaluate candidate contours
        for c in contour_list:
            area = float(cv2.contourArea(c))
            if area < min_area:
                # Since contours are sorted descending, subsequent contours are also too small
                break
            if area > max_area:
                continue

            peri = cv2.arcLength(c, True)
            if peri < 1e-6:
                continue

            for eps_ratio in config.approx_epsilon_ratios:
                approx = cv2.approxPolyDP(c, eps_ratio * peri, True)
                if len(approx) == 4:
                    ordered = order_corners(approx)
                    contour_collision = assess_boundary_collision(
                        ordered,
                        (det_w, det_h),
                        margin_ratio=config.boundary_margin_ratio,
                        paper_mask=paper_mask,
                    )
                    is_valid, conf, diag = _validate_quadrilateral(ordered, det_w, det_h, config)
                    if not is_valid and contour_collision.is_clipped:
                        rejected_candidates.append(
                            (
                                0.0,
                                ordered,
                                max(float(cv2.contourArea(ordered)) / img_area, 0.0),
                                {
                                    **diag,
                                    "candidate_source": "edge_contour",
                                    "is_clipped": True,
                                    "collision_reasons": contour_collision.reasons,
                                },
                                contour_collision,
                                DetectionRejectionReason.DOCUMENT_CLIPPED,
                            )
                        )
                    if is_valid:
                        area_ratio = float(cv2.contourArea(ordered) / img_area)
                        outer_ink = _evaluate_outer_margin_ink(
                            gray, ordered, det_w, det_h, paper_threshold=paper_threshold
                        )
                        table_penalty = 1.0
                        if outer_ink > config.table_inner_ink_threshold:
                            table_penalty = max(
                                config.table_downrank_factor,
                                1.0 - (outer_ink / 0.15) * (1.0 - config.table_downrank_factor),
                            )
                        score = conf * (0.6 + 0.4 * area_ratio) * table_penalty
                        diag = {
                            **diag,
                            "candidate_source": "edge_contour",
                            "outer_margin_ink": outer_ink,
                            "table_penalty": table_penalty,
                            "is_clipped": contour_collision.is_clipped,
                            "collision_reasons": contour_collision.reasons,
                        }
                        if outer_ink > config.table_inner_ink_threshold:
                            diag["rejection_reason"] = (
                                DetectionRejectionReason.INNER_TABLE_COLLAPSE.value
                            )
                            rejected_candidates.append(
                                (
                                    score,
                                    ordered,
                                    area_ratio,
                                    diag,
                                    contour_collision,
                                    DetectionRejectionReason.INNER_TABLE_COLLAPSE,
                                )
                            )
                        else:
                            candidates.append((score, ordered, area_ratio, diag, contour_collision))
                        break  # Found best approximation for this contour

        if not candidates:
            if rejected_candidates:
                clipped_rejections = [
                    candidate for candidate in rejected_candidates if candidate[4].is_clipped
                ]
                rejection_pool = clipped_rejections or rejected_candidates
                rejection_pool.sort(key=lambda item: item[0], reverse=True)
                (
                    best_score,
                    best_det_corners,
                    best_area_ratio,
                    best_diag,
                    best_coll,
                    rejected_reason,
                ) = rejection_pool[0]
                inv_scale = 1.0 / scale_factor
                orig_corners_arr = best_det_corners * inv_scale
                orig_corners_list = [
                    (float(orig_corners_arr[i, 0]), float(orig_corners_arr[i, 1])) for i in range(4)
                ]
                return DetectionResult(
                    detected=True,
                    accepted=False,
                    clipped=best_coll.is_clipped,
                    rejection_reason=(
                        DetectionRejectionReason.DOCUMENT_CLIPPED
                        if best_coll.is_clipped
                        else rejected_reason
                    ),
                    corners=orig_corners_list,
                    confidence=best_score,
                    area_ratio=best_area_ratio,
                    scale_factor=scale_factor,
                    diagnostics={
                        **best_diag,
                        "collision_reasons": best_coll.reasons,
                        "total_contours": len(contour_list),
                        "candidates_found": len(candidates),
                        "rejected_candidates": len(rejected_candidates),
                        "detection_size": [det_w, det_h],
                        "original_size": [orig_w, orig_h],
                    },
                )
            return None

        inv_scale = 1.0 / scale_factor

        unclipped_candidates = [c for c in candidates if not c[4].is_clipped]

        if unclipped_candidates:
            paper_candidates = [
                candidate
                for candidate in unclipped_candidates
                if candidate[3].get("candidate_source") in {"paper_mask", "paper_min_area_rect"}
            ]
            if paper_candidates:
                unclipped_candidates = paper_candidates
            unclipped_candidates.sort(key=lambda item: item[0], reverse=True)
            best_score, best_det_corners, best_area_ratio, best_diag, _ = unclipped_candidates[0]
            orig_corners_arr = best_det_corners * inv_scale
            orig_corners_list = [
                (float(orig_corners_arr[i, 0]), float(orig_corners_arr[i, 1])) for i in range(4)
            ]
            orig_collision = assess_boundary_collision(
                orig_corners_arr,
                (orig_w, orig_h),
                margin_ratio=config.boundary_margin_ratio,
            )
            if orig_collision.is_clipped:
                return DetectionResult(
                    detected=True,
                    accepted=False,
                    clipped=True,
                    rejection_reason=DetectionRejectionReason.DOCUMENT_CLIPPED,
                    corners=orig_corners_list,
                    confidence=best_score,
                    area_ratio=best_area_ratio,
                    scale_factor=scale_factor,
                    diagnostics={
                        **best_diag,
                        "collision_reasons": orig_collision.reasons,
                        "border_touch_edges": orig_collision.border_touch_edges,
                        "min_border_distance": orig_collision.min_border_distance,
                        "total_contours": len(contour_list),
                        "candidates_found": len(candidates),
                        "detection_size": [det_w, det_h],
                        "original_size": [orig_w, orig_h],
                    },
                )
            return DetectionResult(
                detected=True,
                accepted=True,
                clipped=False,
                rejection_reason=None,
                corners=orig_corners_list,
                confidence=best_score,
                area_ratio=best_area_ratio,
                scale_factor=scale_factor,
                diagnostics={
                    **best_diag,
                    "total_contours": len(contour_list),
                    "candidates_found": len(candidates),
                    "detection_size": [det_w, det_h],
                    "original_size": [orig_w, orig_h],
                },
            )

        # Only clipped candidates found
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_det_corners, best_area_ratio, best_diag, best_coll = candidates[0]
        orig_corners_arr = best_det_corners * inv_scale
        orig_corners_list = [
            (float(orig_corners_arr[i, 0]), float(orig_corners_arr[i, 1])) for i in range(4)
        ]
        orig_collision = assess_boundary_collision(
            orig_corners_arr,
            (orig_w, orig_h),
            margin_ratio=config.boundary_margin_ratio,
        )
        all_reasons = list(dict.fromkeys(best_coll.reasons + orig_collision.reasons))
        return DetectionResult(
            detected=True,
            accepted=False,
            clipped=True,
            rejection_reason=DetectionRejectionReason.DOCUMENT_CLIPPED,
            corners=orig_corners_list,
            confidence=best_score,
            area_ratio=best_area_ratio,
            scale_factor=scale_factor,
            diagnostics={
                **best_diag,
                "collision_reasons": all_reasons,
                "border_touch_edges": orig_collision.border_touch_edges,
                "min_border_distance": orig_collision.min_border_distance,
                "total_contours": len(contour_list),
                "candidates_found": len(candidates),
                "detection_size": [det_w, det_h],
                "original_size": [orig_w, orig_h],
            },
        )
    except Exception:
        # Classical contour detection should never crash orchestrator
        return None
