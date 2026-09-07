"""Document boundary detection using classical OpenCV contour analysis.

This module is the second stage of the document scanning pipeline. It operates on a
detection-size copy of the source image to identify the 4-corner document contour,
validates its geometric quadrilateral properties (convexity, side lengths, interior angles,
area ratio), orders the vertices into canonical [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
order, and scales the coordinates back to the original source image bounds.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from pydantic import Field, model_validator

from attendance_scanner.contracts import BaseContract
from attendance_scanner.pipeline.load import LoadedImage


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
    corners: List[Tuple[float, float]]  # Ordered clockwise: [TL, TR, BR, BL]
    confidence: float  # Quality score between 0.0 and 1.0
    area_ratio: float  # Detected document area relative to original image area
    scale_factor: float  # Scale factor applied to detection copy
    diagnostics: Dict[str, Any] = Field(default_factory=dict)

    @property
    def corners_array(self) -> np.ndarray:
        """Return corners as a (4, 2) float32 NumPy array."""
        return np.array(self.corners, dtype=np.float32)


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
    8. Map corners back to original image space and clamp to source image boundaries.

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

        if not contour_list:
            return None

        # 5. Sort contours by area descending
        contour_list.sort(key=cv2.contourArea, reverse=True)

        img_area = float(det_w * det_h)
        min_area = img_area * config.min_area_ratio
        max_area = img_area * config.max_area_ratio

        candidates: List[Tuple[float, np.ndarray, float, Dict[str, Any]]] = []

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
                    is_valid, conf, diag = _validate_quadrilateral(ordered, det_w, det_h, config)
                    if is_valid:
                        area_ratio = float(cv2.contourArea(ordered) / img_area)
                        # Candidate score balances area and confidence
                        score = conf * (0.6 + 0.4 * area_ratio)
                        candidates.append((score, ordered, area_ratio, diag))
                        break  # Found best approximation for this contour

        if not candidates:
            return None

        # 7. Select candidate with highest score
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_det_corners, best_area_ratio, best_diag = candidates[0]

        # 8. Map corners back to original image space
        inv_scale = 1.0 / scale_factor
        orig_corners_arr = best_det_corners * inv_scale

        # Clamp coordinates to original image bounds
        orig_corners_arr[:, 0] = np.clip(orig_corners_arr[:, 0], 0.0, float(orig_w - 1))
        orig_corners_arr[:, 1] = np.clip(orig_corners_arr[:, 1], 0.0, float(orig_h - 1))

        # Convert to list of (x, y) tuples
        orig_corners_list: List[Tuple[float, float]] = [
            (float(orig_corners_arr[i, 0]), float(orig_corners_arr[i, 1])) for i in range(4)
        ]

        return DetectionResult(
            detected=True,
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
    except Exception:
        # Classical CV errors (e.g. cv2.error on degenerate dimensions) must never crash caller
        return None
