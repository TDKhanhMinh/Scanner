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

__all__ = [
    "load_image",
    "LoadedImage",
    "LoadedImageMetadata",
    "DetectionConfig",
    "DetectionResult",
    "detect_document_boundary",
    "order_corners",
]
