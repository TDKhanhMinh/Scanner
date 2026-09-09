"""Scan pipeline modules for image loading, detection, perspective warp, and enhancement."""

from attendance_scanner.pipeline.detect import (
    BoundaryCollisionResult,
    DetectionConfig,
    DetectionRejectionReason,
    DetectionResult,
    assess_boundary_collision,
    detect_document_boundary,
    order_corners,
)
from attendance_scanner.pipeline.enhance import (
    EnhancementConfig,
    enhance_bw,
    enhance_color,
    enhance_gray,
    enhance_image,
)
from attendance_scanner.pipeline.load import (
    LoadedImage,
    LoadedImageMetadata,
    load_image,
)
from attendance_scanner.pipeline.orchestrator import (
    PipelineConfig,
    ResizeConfig,
    SingleScanDiagnostics,
    SingleScanResult,
    scan_one,
)
from attendance_scanner.pipeline.perspective import (
    DegenerateCornersError,
    PerspectiveConfig,
    WarpedDocument,
    compute_destination_dimensions,
    warp_perspective,
)

__all__ = [
    "load_image",
    "LoadedImage",
    "LoadedImageMetadata",
    "BoundaryCollisionResult",
    "DetectionConfig",
    "DetectionRejectionReason",
    "DetectionResult",
    "assess_boundary_collision",
    "detect_document_boundary",
    "order_corners",
    "DegenerateCornersError",
    "PerspectiveConfig",
    "WarpedDocument",
    "compute_destination_dimensions",
    "warp_perspective",
    "EnhancementConfig",
    "enhance_image",
    "enhance_gray",
    "enhance_bw",
    "enhance_color",
    "ResizeConfig",
    "PipelineConfig",
    "SingleScanDiagnostics",
    "SingleScanResult",
    "scan_one",
]
