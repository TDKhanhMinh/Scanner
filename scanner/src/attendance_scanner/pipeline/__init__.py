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
    enhance_smart_document,
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
    PerspectiveV2Config,
    WarpedDocument,
    compute_destination_dimensions,
    warp_perspective,
    warp_perspective_v2,
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
    "PerspectiveV2Config",
    "WarpedDocument",
    "compute_destination_dimensions",
    "warp_perspective",
    "warp_perspective_v2",
    "EnhancementConfig",
    "enhance_image",
    "enhance_gray",
    "enhance_bw",
    "enhance_color",
    "enhance_smart_document",
    "ResizeConfig",
    "PipelineConfig",
    "SingleScanDiagnostics",
    "SingleScanResult",
    "scan_one",
]
