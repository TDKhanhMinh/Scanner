"""Scan pipeline modules for image loading, detection, perspective warp, and enhancement."""

from attendance_scanner.pipeline.detect import (
    DetectionConfig,
    DetectionResult,
    detect_document_boundary,
    order_corners,
)
from attendance_scanner.pipeline.load import (
    LoadedImage,
    LoadedImageMetadata,
    load_image,
)
from attendance_scanner.pipeline.perspective import (
    DegenerateCornersError,
    WarpedDocument,
    compute_destination_dimensions,
    warp_perspective,
)

__all__ = [
    "load_image",
    "LoadedImage",
    "LoadedImageMetadata",
    "DetectionConfig",
    "DetectionResult",
    "detect_document_boundary",
    "order_corners",
    "DegenerateCornersError",
    "WarpedDocument",
    "compute_destination_dimensions",
    "warp_perspective",
]
