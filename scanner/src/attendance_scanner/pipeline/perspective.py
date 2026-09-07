"""Document perspective correction and 4-point quadrilateral warping.

This module is the third stage of the document scanning pipeline. Given an image and
the 4 detected document corners, it orders the corners canonically [TL, TR, BR, BL],
calculates unwarped destination dimensions (width and height) from the maximum opposite-side
Euclidean distances, validates against degenerate/collinear corner configurations, computes
the 3x3 perspective homography matrix, and applies `cv2.warpPerspective` to produce a rectified
rectangular document while strictly preserving the natural aspect ratio and orientation.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from pydantic import Field

from attendance_scanner.contracts import BaseContract
from attendance_scanner.pipeline.detect import DetectionResult, order_corners
from attendance_scanner.pipeline.load import LoadedImage


class DegenerateCornersError(ValueError):
    """Raised when corner points are degenerate, collinear, or yield non-positive dimensions."""


class PerspectiveConfig(BaseContract):
    """Configuration options for perspective transformation."""

    min_dimension: int = Field(default=10, ge=3, le=100)
    interpolation: int = cv2.INTER_LINEAR
    border_mode: int = cv2.BORDER_REPLICATE


@dataclass
class WarpedDocument:
    """Result of perspective transformation containing rectified image and transformation data."""

    image: np.ndarray  # Warped BGR array: shape (height, width, 3), dtype np.uint8
    width: int  # Rectified width in pixels
    height: int  # Rectified height in pixels
    transform_matrix: np.ndarray  # 3x3 perspective transformation matrix (float64)
    source_corners: np.ndarray  # (4, 2) ordered source corners in clockwise [TL, TR, BR, BL]

    @property
    def shape(self) -> Tuple[int, ...]:
        """Array shape (height, width, channels)."""
        return self.image.shape

    @property
    def channels(self) -> int:
        """Number of color channels (3 for BGR)."""
        return 3 if self.image.ndim == 3 else 1


def compute_destination_dimensions(ordered_corners: np.ndarray) -> Tuple[int, int]:
    """Calculate rectangular destination dimensions from maximum opposite-side distances.

    Given ordered corners [TL, TR, BR, BL]:
    - Top edge width: ||TR - TL||
    - Bottom edge width: ||BR - BL||
    - Destination width: max(round(top_width), round(bottom_width))
    - Right edge height: ||BR - TR||
    - Left edge height: ||BL - TL||
    - Destination height: max(round(right_height), round(left_height))

    Args:
        ordered_corners: Array of shape (4, 2) ordered [TL, TR, BR, BL].

    Returns:
        Tuple of (width, height) in pixels as positive integers.
    """
    pts = ordered_corners.astype(np.float32)

    # TL (0), TR (1), BR (2), BL (3)
    width_top = float(np.linalg.norm(pts[1] - pts[0]))
    width_bottom = float(np.linalg.norm(pts[2] - pts[3]))
    dst_w = max(int(round(width_top)), int(round(width_bottom)))

    height_right = float(np.linalg.norm(pts[2] - pts[1]))
    height_left = float(np.linalg.norm(pts[3] - pts[0]))
    dst_h = max(int(round(height_right)), int(round(height_left)))

    return dst_w, dst_h


def warp_perspective(
    image: Union[np.ndarray, LoadedImage],
    corners: Union[np.ndarray, List[Tuple[float, float]], DetectionResult],
    config: Optional[Union[PerspectiveConfig, int]] = None,
    *,
    min_dimension: Optional[int] = None,
) -> WarpedDocument:
    """Rectify a perspective-distorted document quadrilateral into an upright rectangular image.

    Pipeline sequence:
    1. Extract raw OpenCV BGR image from ndarray or LoadedImage.
    2. Extract and format corner points as (4, 2) float32 array.
    3. Validate numerical validity (no NaNs, Infs, or incorrect point counts).
    4. Order corners into canonical clockwise order [TL, TR, BR, BL].
    5. Validate non-degeneracy (non-collinear, area > 0, side lengths >= min_dimension).
    6. Compute destination dimensions (W, H) from maximum opposite-side lengths.
    7. Form destination grid: [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]].
    8. Compute 3x3 transformation matrix M using `cv2.getPerspectiveTransform`.
    9. Warp image using `cv2.warpPerspective`.

    Args:
        image: Source OpenCV BGR array (shape H, W, 3) or LoadedImage instance.
        corners: 4 corner points as (4, 2) ndarray, list of (x, y) tuples, or DetectionResult.
        config: Optional PerspectiveConfig instance or int min_dimension for backward compatibility.
        min_dimension: Keyword-only minimum allowed dimension in pixels.

    Returns:
        WarpedDocument with rectified BGR image, dimensions, and transformation metadata.

    Raises:
        DegenerateCornersError: If corners are collinear, zero-area, smaller than min_dimension,
                                contain non-finite values, or have invalid shapes.
        ValueError: If input image is invalid or empty.
    """
    if isinstance(config, PerspectiveConfig):
        cfg = config
    elif isinstance(config, int):
        cfg = PerspectiveConfig(min_dimension=config)
    elif min_dimension is not None:
        cfg = PerspectiveConfig(min_dimension=min_dimension)
    else:
        cfg = PerspectiveConfig()
    # 1. Extract BGR image
    if isinstance(image, LoadedImage):
        bgr = image.image
    elif isinstance(image, np.ndarray):
        bgr = image
    else:
        raise ValueError(f"Expected np.ndarray or LoadedImage, got {type(image)}")

    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError(
            f"Expected 3-channel uint8 BGR image, got shape {getattr(bgr, 'shape', None)} "
            f"and dtype {getattr(bgr, 'dtype', None)}"
        )

    # 2. Extract corner points
    if isinstance(corners, DetectionResult):
        raw_pts = corners.corners_array
    elif isinstance(corners, np.ndarray):
        raw_pts = corners.astype(np.float32)
    elif isinstance(corners, (list, tuple)):
        raw_pts = np.array(corners, dtype=np.float32)
    else:
        raise DegenerateCornersError(f"Unsupported corners type: {type(corners)}")

    # 3. Shape and numerical sanity validation
    try:
        pts = raw_pts.reshape(4, 2).astype(np.float32)
    except Exception as exc:
        raise DegenerateCornersError(
            f"Corners must be shape (4, 2) or convertible to 4 points, got {raw_pts.shape}"
        ) from exc

    if not np.all(np.isfinite(pts)):
        raise DegenerateCornersError("Corner coordinates contain non-finite values (NaN or Inf)")

    # 4. Canonical clockwise ordering: [TL, TR, BR, BL]
    ordered_src = order_corners(pts)

    # 5. Non-degeneracy validation: check polygon area and collinearity
    area = float(cv2.contourArea(ordered_src))
    if area < 1e-2:
        raise DegenerateCornersError(
            f"Corner polygon is degenerate or collinear (area={area:.4f} < 0.01)"
        )

    # 6. Compute destination width and height from max opposite-side Euclidean distances
    dst_w, dst_h = compute_destination_dimensions(ordered_src)

    if dst_w < cfg.min_dimension or dst_h < cfg.min_dimension:
        raise DegenerateCornersError(
            f"Calculated destination dimensions ({dst_w}x{dst_h}) are smaller than "
            f"minimum dimension threshold ({cfg.min_dimension}px)"
        )

    # 7. Define destination coordinates
    dst_pts = np.array(
        [
            [0.0, 0.0],
            [float(dst_w - 1), 0.0],
            [float(dst_w - 1), float(dst_h - 1)],
            [0.0, float(dst_h - 1)],
        ],
        dtype=np.float32,
    )

    # 8. Compute perspective transformation matrix
    try:
        matrix = cv2.getPerspectiveTransform(ordered_src, dst_pts)
    except cv2.error as cv_err:
        raise DegenerateCornersError(
            f"OpenCV failed to compute perspective transform matrix: {cv_err}"
        ) from cv_err

    if matrix is None or not np.all(np.isfinite(matrix)):
        raise DegenerateCornersError("Computed perspective transform matrix is singular or invalid")

    # 9. Warp image to rectified rectangle
    # BORDER_REPLICATE avoids artificial black border halos at outer boundary pixels
    warped = cv2.warpPerspective(
        bgr,
        matrix,
        (dst_w, dst_h),
        flags=cfg.interpolation,
        borderMode=cfg.border_mode,
    )

    return WarpedDocument(
        image=warped,
        width=dst_w,
        height=dst_h,
        transform_matrix=matrix,
        source_corners=ordered_src,
    )


__all__ = [
    "DegenerateCornersError",
    "PerspectiveConfig",
    "WarpedDocument",
    "compute_destination_dimensions",
    "warp_perspective",
]
