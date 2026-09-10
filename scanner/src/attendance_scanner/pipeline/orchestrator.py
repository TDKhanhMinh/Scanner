"""Single-image scan pipeline orchestration (scan_one).

Composes the discrete scanning stages into a unified, pure in-memory pipeline:
1. Load image (with EXIF rotation and transparency handling)
2. Detect document boundary (OpenCV edge/contour/quadrilateral detection)
3. Optional Warp (rectify perspective if quad detected; safe fallback if not)
4. Enhance (Gray, B&W, Color Enhanced, or Smart Document filters)
5. Size Normalization (no upscaling, proportional downscaling if oversized)
6. Result assembly (SingleScanResult with typed diagnostics and warnings)
"""

import base64
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import cv2
import numpy as np
from pydantic import Field

from attendance_scanner.contracts import (
    BaseContract,
    DetectionFailureReason,
    DetectionPreview,
    DetectionPreviewCandidate,
    FileProcessingStatus,
    FileResult,
    ImageDecodeError,
    ImageProcessError,
    PageIdentity,
    PreviewPoint,
    ScanMode,
    ScannerWarningCode,
)
from attendance_scanner.detector import DocumentDetectionResult, DocumentDetector
from attendance_scanner.detector_modes import internal_detector_mode, normalize_detector_mode
from attendance_scanner.diagnostics import summarize_detection_failure
from attendance_scanner.page_classification import classify_page
from attendance_scanner.pipeline.detect import (
    DetectionConfig,
    DetectionRejectionReason,
    DetectionResult,
    detect_document_boundary,
)
from attendance_scanner.pipeline.enhance import EnhancementConfig, enhance_image
from attendance_scanner.pipeline.load import LoadedImage, LoadedImageMetadata, load_image
from attendance_scanner.pipeline.perspective import (
    DegenerateCornersError,
    PerspectiveConfig,
    WarpedDocument,
    warp_perspective,
)


class ResizeConfig(BaseContract):
    """Configuration for size normalization and downscaling.

    Invariants:
    - Never upscales small images.
    - Proportionally downscales oversized images exceeding max_dimension or max_pixels.
    - Preserves aspect ratio.
    """

    enabled: bool = True
    max_dimension: Optional[int] = Field(default=3200, gt=10)
    max_pixels: Optional[int] = Field(default=None, gt=100)
    interpolation: int = cv2.INTER_AREA
    warn_on_downscale: bool = False


class PipelineConfig(BaseContract):
    """Unified configuration for the full single-image attendance scan pipeline."""

    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    perspective: PerspectiveConfig = Field(
        default_factory=lambda: PerspectiveConfig(target_aspect_ratio=math.sqrt(2.0))
    )
    enhancement: EnhancementConfig = Field(default_factory=EnhancementConfig)
    resize: ResizeConfig = Field(default_factory=ResizeConfig)
    warp_fallback_to_full: bool = True
    # Attendance forms use a landscape output canvas; generic documents can opt
    # into the source orientation with ``preferred_orientation=natural``.
    preferred_orientation: Literal["natural", "landscape", "portrait"] = "landscape"


class SingleScanDiagnostics(BaseContract):
    """Detailed diagnostics and telemetry for a single image scan execution."""

    document_detected: bool = False
    detection_confidence: Optional[float] = None
    detection_area_ratio: Optional[float] = None
    original_width: int = 0
    original_height: int = 0
    output_width: int = 0
    output_height: int = 0
    downscale_ratio: float = 1.0
    orientation_rotation_degrees: int = 0
    warning_codes: List[str] = Field(default_factory=list)
    stage_durations_ms: Dict[str, float] = Field(default_factory=dict)
    total_duration_ms: float = 0.0

    @property
    def warning(self) -> Optional[str]:
        """Joined string of warning codes or None if clean."""
        if not self.warning_codes:
            return None
        return ", ".join(self.warning_codes)


@dataclass
class SingleScanResult:
    """Result of executing the single-image scan pipeline in memory."""

    image: np.ndarray  # Processed uint8 array: shape (H, W) for GRAY/BW, (H, W, 3) for COLOR
    document_detected: bool
    original_width: int
    original_height: int
    output_width: int
    output_height: int
    mode: ScanMode
    warning_codes: List[str] = field(default_factory=list)
    diagnostics: SingleScanDiagnostics = field(default_factory=SingleScanDiagnostics)
    page_identity: PageIdentity = field(default_factory=PageIdentity)
    detector_name: str = "v1_cv"
    detector_mode: Optional[str] = None
    detector_model_version: Optional[str] = "opencv-classical"
    detector_model_checksum: Optional[str] = None
    detection_fallback_used: bool = False
    detection_quality_summary: Dict[str, Union[str, int, float, bool, None]] = field(
        default_factory=dict
    )
    detection_reason: Optional[DetectionFailureReason] = None
    detection_reason_codes: List[DetectionFailureReason] = field(default_factory=list)
    detection_user_message: Optional[str] = None
    detection_preview: Optional[DetectionPreview] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the processed output image array."""
        return self.image.shape

    @property
    def channels(self) -> int:
        """Number of color channels (1 for GRAY/BW, 3 for COLOR)."""
        return 1 if self.image.ndim == 2 else self.image.shape[2]

    @property
    def has_warning(self) -> bool:
        """Whether any warnings were flagged during processing."""
        return len(self.warning_codes) > 0

    @property
    def warning(self) -> Optional[str]:
        """Joined string of warning codes or None if clean."""
        if not self.warning_codes:
            return None
        return ", ".join(self.warning_codes)

    def to_file_result(
        self,
        relative_path: str,
        employee_name: str,
        target_relative_pdf: str,
    ) -> FileResult:
        """Convert single-scan result into a contract-compliant FileResult."""
        status = FileProcessingStatus.WARNING if self.has_warning else FileProcessingStatus.SUCCESS
        return FileResult(
            relative_path=relative_path,
            employee_name=employee_name,
            target_relative_pdf=target_relative_pdf,
            status=status,
            document_detected=self.document_detected,
            warning=self.warning,
            duration_ms=int(round(self.diagnostics.total_duration_ms)),
        )


def _prepare_loaded_image(
    source: Union[str, Path, LoadedImage, np.ndarray],
) -> Tuple[LoadedImage, int, int]:
    """Normalize input source into a LoadedImage and extract original dimensions."""
    if isinstance(source, (str, Path)):
        loaded = load_image(source)
        return loaded, loaded.metadata.original_width, loaded.metadata.original_height

    if isinstance(source, LoadedImage):
        return source, source.metadata.original_width, source.metadata.original_height

    if isinstance(source, np.ndarray):
        if source.dtype != np.uint8:
            raise ValueError(f"NumPy array input must have dtype=uint8, got {source.dtype}")
        h, w = source.shape[:2]
        if source.ndim == 2:
            bgr = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
        elif source.ndim == 3 and source.shape[2] == 3:
            bgr = source.copy()
        elif source.ndim == 3 and source.shape[2] == 4:
            bgr = cv2.cvtColor(source, cv2.COLOR_BGRA2BGR)
        else:
            raise ValueError(f"Unsupported NumPy array shape: {source.shape}")

        metadata = LoadedImageMetadata(
            original_width=w,
            original_height=h,
            width=w,
            height=h,
            original_mode="BGR",
            normalized_mode="BGR",
            has_transparency=False,
        )
        return (
            LoadedImage(path=Path("<memory>"), image=bgr, metadata=metadata),
            w,
            h,
        )

    raise TypeError(
        f"Unsupported source type: {type(source)}. Must be str, Path, LoadedImage, or np.ndarray."
    )


def _normalize_size(
    image: np.ndarray,
    config: ResizeConfig,
) -> Tuple[np.ndarray, float]:
    """Downscale image if exceeding size budget; never upscales.

    Returns:
        Tuple of (resized_or_original_image, scale_factor)
    """
    if not config.enabled:
        return image, 1.0

    h, w = image.shape[:2]
    scale = 1.0

    # 1. Check max_dimension constraint
    if config.max_dimension is not None:
        max_dim = max(h, w)
        if max_dim > config.max_dimension:
            scale = min(scale, config.max_dimension / float(max_dim))

    # 2. Check max_pixels constraint
    if config.max_pixels is not None:
        total_pixels = h * w
        if total_pixels > config.max_pixels:
            scale = min(scale, math.sqrt(config.max_pixels / float(total_pixels)))

    # Invariant: Never upscale
    if scale >= 1.0:
        return image, 1.0

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(image, (new_w, new_h), interpolation=config.interpolation)
    return resized, scale


def _preview_points(
    points: Optional[List[Tuple[float, float]]],
) -> Optional[List[PreviewPoint]]:
    if points is None or len(points) != 4:
        return None
    return [PreviewPoint(x=float(x), y=float(y)) for x, y in points]


def _encode_bounded_preview(image: np.ndarray, max_dimension: int = 1280) -> Optional[str]:
    """Encode a bounded JPEG preview so the frontend never loads the source image at full size."""
    height, width = image.shape[:2]
    target_dimension = max_dimension
    for _ in range(4):
        scale = min(1.0, target_dimension / float(max(height, width)))
        preview = image
        if scale < 1.0:
            preview = cv2.resize(
                image,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        for quality in (82, 72, 60):
            success, encoded = cv2.imencode(
                ".jpg",
                preview,
                [int(cv2.IMWRITE_JPEG_QUALITY), quality],
            )
            if not success:
                continue
            payload = base64.b64encode(encoded.tobytes()).decode("ascii")
            data_url = f"data:image/jpeg;base64,{payload}"
            if len(payload) <= 1_900_000:
                return data_url
        target_dimension = max(256, round(target_dimension * 0.75))
    return None


def _build_detection_preview(
    *,
    loaded: LoadedImage,
    v2_detection: Optional[DocumentDetectionResult],
    detection: Optional[DetectionResult],
    document_detected: bool,
    detector_name: str,
    detector_model_version: Optional[str],
    fallback_used: bool,
    confidence: Optional[float],
    reason_codes: List[DetectionFailureReason],
    warning_codes: List[str],
) -> DetectionPreview:
    """Build bounded, source-coordinate evidence for the in-app diagnostic overlay."""
    candidates: List[DetectionPreviewCandidate] = []
    final_corners: Optional[List[PreviewPoint]] = None
    mask_available = False
    if v2_detection is not None:
        final_corners = (
            _preview_points(v2_detection.corners.as_list())
            if document_detected and v2_detection.corners is not None
            else None
        )
        mask_available = v2_detection.evidence.mask_confidence is not None
        for candidate in v2_detection.candidate_corners:
            preview_candidate = _preview_points(candidate.points)
            if preview_candidate is not None:
                candidates.append(
                    DetectionPreviewCandidate(
                        source=candidate.source,
                        corners=preview_candidate,
                        confidence=candidate.confidence,
                    )
                )
    elif detection is not None:
        final_corners = _preview_points(detection.corners if document_detected else None)
        if detection.corners:
            rejected_candidate = _preview_points(detection.corners)
            if rejected_candidate is not None:
                candidates.append(
                    DetectionPreviewCandidate(
                        source="v1_cv",
                        corners=rejected_candidate,
                        confidence=detection.confidence,
                    )
                )

    return DetectionPreview(
        source_width=loaded.width,
        source_height=loaded.height,
        final_corners=final_corners,
        candidate_corners=candidates,
        mask_available=mask_available,
        preview_image_data_url=_encode_bounded_preview(loaded.image),
        confidence=confidence,
        fallback_used=fallback_used,
        detector_name=detector_name,
        model_version=detector_model_version,
        reason_code=(reason_codes[0].value if reason_codes else None),
        reason_codes=[reason.value for reason in reason_codes],
        warning_codes=warning_codes,
    )


def scan_one(
    source: Union[str, Path, LoadedImage, np.ndarray],
    mode: Union[ScanMode, str] = ScanMode.GRAY,
    *,
    config: Optional[PipelineConfig] = None,
    detector: Optional[DocumentDetector] = None,
    detector_mode: Optional[str] = None,
    debug_diagnostics: bool = False,
) -> SingleScanResult:
    """Execute the single-image scan pipeline in memory.

    Pipeline stages:
    1. Load: decodes source into standardized BGR LoadedImage with EXIF orientation.
    2. Detect: finds 4-corner document quad. Returns None if unconfident or degenerate.
    3. Optional Warp: rectifies document perspective. If detector returned None,
       safely falls back to full normalized image with warning DOCUMENT_NOT_DETECTED.
    4. Enhance: applies the selected scan mode, including Smart Document.
    5. Resize: downscales oversized images proportionally without upscaling.

    Args:
        source: File path, LoadedImage instance, or uint8 NumPy array.
        mode: Scan filter mode (GRAY default, BW, or COLOR).
        config: Optional pipeline configuration overrides.

    Returns:
        SingleScanResult with processed image array and execution diagnostics.
    """
    t_start = time.perf_counter()
    pipeline_cfg = config or PipelineConfig()
    stage_durations: Dict[str, float] = {}
    warning_codes: List[str] = []

    # Parse and validate ScanMode
    if isinstance(mode, str):
        mode_str = mode.strip().lower()
        if mode_str in ("color_enhanced", "colored"):
            scan_mode = ScanMode.COLOR
        else:
            scan_mode = ScanMode(mode_str)
    else:
        scan_mode = mode

    # Stage 1: Load image
    t_load_start = time.perf_counter()
    loaded, orig_w, orig_h = _prepare_loaded_image(source)
    stage_durations["load_ms"] = (time.perf_counter() - t_load_start) * 1000.0

    normalized_detector_mode = (
        normalize_detector_mode(detector_mode) if detector_mode is not None else None
    )

    # Stage 2: Detect document boundary. The legacy path remains the default for
    # direct library callers; product/benchmark modes opt into the provider-neutral
    # V2 detector contract without exposing model names to the caller.
    t_detect_start = time.perf_counter()
    v2_detection: Optional[DocumentDetectionResult] = None
    detection: Optional[DetectionResult] = None
    if detector is not None or normalized_detector_mode not in {None, "classic", "v1_cv"}:
        if detector is None:
            from attendance_scanner.hybrid import create_detector

            assert normalized_detector_mode is not None
            detector = create_detector(internal_detector_mode(normalized_detector_mode))
        v2_detection = detector.detect(loaded)
    else:
        detection = detect_document_boundary(
            loaded,
            config=pipeline_cfg.detection,
        )
    stage_durations["detect_ms"] = (time.perf_counter() - t_detect_start) * 1000.0

    detector_mode_value: Optional[str]
    if v2_detection is not None:
        v2_corners = v2_detection.canonical_points()
        document_detected = bool(
            v2_detection.detected and v2_corners is not None and not v2_detection.geometry.clipped
        )
        detection_confidence = v2_detection.confidence
        detection_area_ratio = v2_detection.geometry.area_ratio
        accepted_corners: Optional[Union[List[Tuple[float, float]], np.ndarray]] = (
            v2_corners if document_detected else None
        )
        detection_clipped = v2_detection.geometry.clipped
        warning_codes.extend(v2_detection.warnings)
    else:
        document_detected = bool(
            detection is not None and detection.accepted and not detection.clipped
        )
        detection_confidence = detection.confidence if detection else None
        detection_area_ratio = detection.area_ratio if detection else None
        accepted_corners = (
            detection.corners if document_detected and detection is not None else None
        )
        detection_clipped = bool(detection is not None and detection.clipped)

    # Stage 3: Optional Perspective Warp
    t_warp_start = time.perf_counter()
    warped_or_full: Union[WarpedDocument, LoadedImage, np.ndarray]
    orientation_rotation_degrees = 0
    page_identity = PageIdentity()

    if accepted_corners is not None:
        try:
            classification_perspective = pipeline_cfg.perspective.model_copy(
                update={"target_aspect_ratio": None}
            )
            classification_warp = warp_perspective(
                loaded,
                accepted_corners,
                config=classification_perspective,
            )
            classification_candidates = [(0, classify_page(classification_warp.image))]
            if loaded.metadata.exif_orientation is None:
                classification_candidates.extend(
                    (
                        turns,
                        classify_page(
                            np.ascontiguousarray(np.rot90(classification_warp.image, turns))
                        ),
                    )
                    for turns in (1, 3)
                )
            selected_turns, page_identity = max(
                classification_candidates,
                key=lambda candidate: candidate[1].confidence or 0.0,
            )
            page_identity.diagnostics["classificationTurns"] = selected_turns
            warped_or_full = warp_perspective(
                loaded,
                accepted_corners,
                config=pipeline_cfg.perspective,
            )
        except (DegenerateCornersError, cv2.error, ValueError):
            if pipeline_cfg.warp_fallback_to_full:
                # Safe fallback to full unwarped image
                warped_or_full = loaded
                document_detected = False
                warning_codes.append(ScannerWarningCode.WARP_FALLBACK.value)
                warning_codes.append(ScannerWarningCode.DOCUMENT_NOT_DETECTED.value)
            else:
                raise
    else:
        # Fallback rule: detector rejected or failed -> use full unwarped image with warning
        warped_or_full = loaded
        if detection_clipped or (
            detection is not None
            and detection.rejection_reason == DetectionRejectionReason.DOCUMENT_CLIPPED
        ):
            warning_codes.append(ScannerWarningCode.DOCUMENT_CLIPPED.value)
        else:
            warning_codes.append(ScannerWarningCode.DOCUMENT_NOT_DETECTED.value)

    if pipeline_cfg.preferred_orientation != "natural":
        orientation_source = getattr(warped_or_full, "image", warped_or_full)
        orientation_array = np.asarray(orientation_source)
        source_height, source_width = orientation_array.shape[:2]
        should_rotate = (
            pipeline_cfg.preferred_orientation == "landscape" and source_height > source_width
        ) or (pipeline_cfg.preferred_orientation == "portrait" and source_width > source_height)
        if should_rotate:
            # The attendance template is read left-to-right after a 90-degree
            # counter-clockwise rotation for portrait-oriented camera captures.
            turns = 1 if pipeline_cfg.preferred_orientation == "landscape" else 3
            warped_or_full = np.ascontiguousarray(np.rot90(orientation_array, turns))
            orientation_rotation_degrees = 90 if turns == 1 else 270

    stage_durations["warp_ms"] = (time.perf_counter() - t_warp_start) * 1000.0

    # Stage 4: Scan enhancement filters
    t_enhance_start = time.perf_counter()
    try:
        enhanced = enhance_image(
            warped_or_full,
            mode=scan_mode,
            config=pipeline_cfg.enhancement,
        )
    except Exception as exc:
        if isinstance(exc, (ImageDecodeError, ImageProcessError)):
            raise
        source_str = str(getattr(loaded, "path", "<image>"))
        raise ImageProcessError(
            f"Enhancement failed ({type(exc).__name__}): {exc}",
            path=source_str,
        ) from exc
    stage_durations["enhance_ms"] = (time.perf_counter() - t_enhance_start) * 1000.0

    # Stage 5: Size normalization (no-upscale, downscale oversized)
    t_resize_start = time.perf_counter()
    try:
        final_image, downscale_ratio = _normalize_size(enhanced, pipeline_cfg.resize)
    except Exception as exc:
        if isinstance(exc, (ImageDecodeError, ImageProcessError)):
            raise
        source_str = str(getattr(loaded, "path", "<image>"))
        raise ImageProcessError(
            f"Size normalization failed ({type(exc).__name__}): {exc}",
            path=source_str,
        ) from exc
    if downscale_ratio < 1.0 and pipeline_cfg.resize.warn_on_downscale:
        warning_codes.append(ScannerWarningCode.IMAGE_DOWNSCALED.value)
    stage_durations["resize_ms"] = (time.perf_counter() - t_resize_start) * 1000.0

    total_duration_ms = (time.perf_counter() - t_start) * 1000.0
    out_h, out_w = final_image.shape[:2]

    # Deduplicate warnings preserving order
    unique_warnings = list(dict.fromkeys(warning_codes))
    detection_summary = summarize_detection_failure(
        document_detected=document_detected,
        warning_codes=unique_warnings,
        fallback_used=not document_detected,
    )

    diagnostics = SingleScanDiagnostics(
        document_detected=document_detected,
        detection_confidence=detection_confidence,
        detection_area_ratio=detection_area_ratio,
        original_width=orig_w,
        original_height=orig_h,
        output_width=out_w,
        output_height=out_h,
        downscale_ratio=downscale_ratio,
        orientation_rotation_degrees=orientation_rotation_degrees,
        warning_codes=unique_warnings,
        stage_durations_ms=stage_durations,
        total_duration_ms=total_duration_ms,
    )

    if v2_detection is not None:
        detector_name = v2_detection.detector_version
        detector_mode_value = normalized_detector_mode
        detector_model_version = v2_detection.model_version
        detector_fallback_used = v2_detection.fallback_used or not document_detected
        detection_quality_summary: Dict[str, Union[str, int, float, bool, None]] = {
            "confidence": detection_confidence,
            "areaRatio": detection_area_ratio,
            "candidateCount": len(v2_detection.candidate_corners),
            "warningCount": len(unique_warnings),
            "fallbackUsed": detector_fallback_used,
        }
        if debug_diagnostics and v2_detection.decision_trace is not None:
            detection_quality_summary.update(
                {
                    "decisionPath": v2_detection.decision_trace.path,
                    "segmentationState": v2_detection.decision_trace.segmentation_state,
                    "rankingStatus": v2_detection.decision_trace.ranking_status,
                }
            )
    else:
        detector_name = "v1_cv"
        detector_mode_value = normalized_detector_mode or scan_mode.value
        detector_model_version = "opencv-classical"
        detector_fallback_used = not document_detected
        detection_quality_summary = {
            "confidence": detection_confidence,
            "areaRatio": detection_area_ratio,
            "warningCount": len(unique_warnings),
        }

    detection_preview = None
    if debug_diagnostics or unique_warnings:
        detection_preview = _build_detection_preview(
            loaded=loaded,
            v2_detection=v2_detection,
            detection=detection,
            document_detected=document_detected,
            detector_name=detector_name,
            detector_model_version=detector_model_version,
            fallback_used=detector_fallback_used,
            confidence=detection_confidence,
            reason_codes=list(detection_summary.reason_codes),
            warning_codes=unique_warnings,
        )

    return SingleScanResult(
        image=final_image,
        document_detected=document_detected,
        original_width=orig_w,
        original_height=orig_h,
        output_width=out_w,
        output_height=out_h,
        mode=scan_mode,
        warning_codes=unique_warnings,
        diagnostics=diagnostics,
        page_identity=page_identity,
        detector_name=detector_name,
        detector_mode=detector_mode_value,
        detector_model_version=detector_model_version,
        detector_model_checksum=None,
        detection_fallback_used=detector_fallback_used,
        detection_quality_summary=detection_quality_summary,
        detection_reason=detection_summary.primary_reason,
        detection_reason_codes=list(detection_summary.reason_codes),
        detection_user_message=detection_summary.user_message,
        detection_preview=detection_preview,
    )
