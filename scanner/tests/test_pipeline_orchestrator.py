"""Unit and integration tests for single-image scan pipeline orchestration (scan_one).

Covers:
- Pipeline composition: load -> detect -> optional warp -> enhance -> resize -> result.
- Detection success path vs detector fallback path (DOCUMENT_NOT_DETECTED warning).
- ScanMode dispatch: GRAY, BW, COLOR, and SMART_DOCUMENT output invariants.
- Size normalization: no upscaling, proportional downscaling with aspect ratio preservation.
- Warp exception safe fallback (WARP_FALLBACK).
- Input versatility (Path, str, LoadedImage, NumPy 2D/3D).
- Immutability and error handling (corrupted/missing files, invalid types).
- Conversion to FileResult contract.
"""

import hashlib
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from PIL import Image

from attendance_scanner.contracts import (
    FileProcessingStatus,
    ImageDecodeError,
    ImageProcessError,
    ScanMode,
    ScannerErrorCode,
    ScannerWarningCode,
)
from attendance_scanner.pipeline.detect import DetectionRejectionReason, DetectionResult
from attendance_scanner.pipeline.load import load_image
from attendance_scanner.pipeline.orchestrator import (
    PipelineConfig,
    ResizeConfig,
    SingleScanResult,
    scan_one,
)
from attendance_scanner.pipeline.perspective import DegenerateCornersError, WarpedDocument


def _create_synthetic_document_image(
    file_path: Path,
    width: int = 800,
    height: int = 600,
) -> Path:
    """Create a high-contrast document image on a dark desk."""
    img = Image.new("RGB", (width, height), color=(30, 30, 30))
    # Inset document sheet
    for y in range(80, height - 80):
        for x in range(100, width - 100):
            img.putpixel((x, y), (240, 240, 240))

    img.save(file_path)
    return file_path


def test_scan_one_happy_path_with_document_detected(tmp_path: Path):
    """Verify end-to-end pipeline detects quad, rectifies perspective, and produces clean result."""
    img_path = _create_synthetic_document_image(tmp_path / "sheet_clean.png", 800, 600)

    result = scan_one(img_path, mode=ScanMode.GRAY)

    assert isinstance(result, SingleScanResult)
    assert result.document_detected is True
    assert not result.has_warning
    assert result.warning is None
    assert result.warning_codes == []
    assert result.original_width == 800
    assert result.original_height == 600
    assert result.output_width > 500
    assert result.output_height > 350
    assert result.image.ndim == 2
    assert result.image.dtype == np.uint8
    assert result.mode == ScanMode.GRAY

    # Diagnostics check
    diag = result.diagnostics
    assert diag.document_detected is True
    assert diag.detection_confidence is not None and diag.detection_confidence > 0.5
    assert diag.detection_area_ratio is not None and diag.detection_area_ratio > 0.3
    assert diag.total_duration_ms > 0
    assert "load_ms" in diag.stage_durations_ms
    assert "detect_ms" in diag.stage_durations_ms
    assert "warp_ms" in diag.stage_durations_ms
    assert "enhance_ms" in diag.stage_durations_ms
    assert "resize_ms" in diag.stage_durations_ms

    # Contract conversion check
    file_res = result.to_file_result(
        relative_path="NV01/sheet_clean.png",
        employee_name="NV01",
        target_relative_pdf="NV01/sheet_clean.pdf",
    )
    assert file_res.status == FileProcessingStatus.SUCCESS
    assert file_res.document_detected is True
    assert file_res.warning is None
    assert file_res.relative_path == "NV01/sheet_clean.png"
    assert file_res.target_relative_pdf == "NV01/sheet_clean.pdf"


def test_scan_one_detector_fallback_rule(tmp_path: Path):
    """Verify fallback rule: if detector returns None, use full normalized image with warning."""
    # Blank gray image without any document contours
    blank_path = tmp_path / "blank_desk.png"
    Image.new("RGB", (600, 400), color=(128, 128, 128)).save(blank_path)

    result = scan_one(blank_path, mode=ScanMode.GRAY)

    # Acceptance: Detector fail path still succeeds with warning
    assert result.document_detected is False
    assert result.has_warning is True
    assert ScannerWarningCode.DOCUMENT_NOT_DETECTED.value in result.warning_codes
    assert result.warning == ScannerWarningCode.DOCUMENT_NOT_DETECTED.value
    # Full unwarped dimensions preserved
    assert result.output_width == 600
    assert result.output_height == 400
    assert result.image.shape == (400, 600)

    # Contract conversion check: status must be WARNING
    file_res = result.to_file_result("NV02/blank.png", "NV02", "NV02/blank.pdf")
    assert file_res.status == FileProcessingStatus.WARNING
    assert file_res.document_detected is False
    assert file_res.warning == ScannerWarningCode.DOCUMENT_NOT_DETECTED.value


def test_scan_one_mode_dispatch(tmp_path: Path):
    """Verify enhancement mode dispatch for all supported output modes."""
    img_path = _create_synthetic_document_image(tmp_path / "modes.png", 500, 400)

    # 1. Gray mode (default)
    res_gray = scan_one(img_path, mode=ScanMode.GRAY)
    assert res_gray.channels == 1
    assert res_gray.image.ndim == 2
    assert res_gray.image.dtype == np.uint8

    # 2. B&W (Magic Pro) mode
    res_bw = scan_one(img_path, mode=ScanMode.BW)
    assert res_bw.channels == 3
    assert res_bw.image.ndim == 3
    assert res_bw.image.dtype == np.uint8

    # 3. Color Enhanced mode
    res_color = scan_one(img_path, mode=ScanMode.COLOR)
    assert res_color.channels == 3
    assert res_color.image.ndim == 3
    assert res_color.image.shape[2] == 3
    assert res_color.image.dtype == np.uint8

    # 4. Smart Document mode
    res_smart = scan_one(img_path, mode=ScanMode.SMART_DOCUMENT)
    assert res_smart.channels == 3
    assert res_smart.image.ndim == 3
    assert res_smart.image.shape[2] == 3
    assert res_smart.image.dtype == np.uint8

    # String mode support
    res_str = scan_one(img_path, mode="color_enhanced")
    assert res_str.mode == ScanMode.COLOR
    assert res_str.channels == 3

    # Invalid mode
    with pytest.raises(ValueError):
        scan_one(img_path, mode="invalid_filter")


def test_scan_one_resize_rules(tmp_path: Path):
    """Verify resize invariants: never upscales, proportionally downscales oversized."""
    # 1. Small image: never upscale even with max_dimension set
    small_path = tmp_path / "small.png"
    Image.new("RGB", (200, 150), color=(100, 100, 100)).save(small_path)

    cfg_small = PipelineConfig(
        resize=ResizeConfig(max_dimension=1000, warn_on_downscale=True),
    )
    res_small = scan_one(small_path, config=cfg_small)
    assert res_small.output_width == 200
    assert res_small.output_height == 150
    assert res_small.diagnostics.downscale_ratio == 1.0
    assert ScannerWarningCode.IMAGE_DOWNSCALED.value not in res_small.warning_codes

    # 2. Large image: downscaled proportionally preserving aspect ratio
    large_path = tmp_path / "large.png"
    # 1600 x 800 (aspect ratio 2:1)
    Image.new("RGB", (1600, 800), color=(100, 100, 100)).save(large_path)

    cfg_large = PipelineConfig(
        resize=ResizeConfig(max_dimension=800, warn_on_downscale=True),
    )
    res_large = scan_one(large_path, config=cfg_large)
    assert res_large.output_width == 800
    assert res_large.output_height == 400
    assert pytest.approx(res_large.diagnostics.downscale_ratio, rel=1e-2) == 0.5
    assert ScannerWarningCode.IMAGE_DOWNSCALED.value in res_large.warning_codes

    # 3. Max pixels budget constraint
    cfg_pixels = PipelineConfig(
        resize=ResizeConfig(max_pixels=200_000, max_dimension=None),
    )
    res_pixels = scan_one(large_path, config=cfg_pixels)
    assert res_pixels.output_width * res_pixels.output_height <= 200_500


def test_scan_one_warp_failure_fallback_rule(tmp_path: Path):
    """Verify pipeline catches degenerate warp errors and falls back to full image safely."""
    img_path = _create_synthetic_document_image(tmp_path / "warp_fail.png", 600, 400)

    # Mock warp_perspective to raise DegenerateCornersError
    with patch(
        "attendance_scanner.pipeline.orchestrator.warp_perspective",
        side_effect=DegenerateCornersError("Degenerate quadrilateral"),
    ):
        result = scan_one(img_path, config=PipelineConfig(warp_fallback_to_full=True))

        assert result.document_detected is False
        assert ScannerWarningCode.WARP_FALLBACK.value in result.warning_codes
        assert ScannerWarningCode.DOCUMENT_NOT_DETECTED.value in result.warning_codes
        assert result.output_width == 600
        assert result.output_height == 400


def test_scan_one_input_types_versatility(tmp_path: Path):
    """Verify scan_one accepts Path, str, LoadedImage, and NumPy array."""
    img_path = _create_synthetic_document_image(tmp_path / "types.png", 400, 300)

    # 1. str path
    res_str = scan_one(str(img_path))
    assert res_str.output_width > 0

    # 2. Path object
    res_path = scan_one(img_path)
    assert res_path.output_width > 0

    # 3. LoadedImage instance
    loaded = load_image(img_path)
    res_loaded = scan_one(loaded)
    assert res_loaded.output_width > 0

    # 4. NumPy 3D BGR array
    raw_bgr = np.full((300, 400, 3), 120, dtype=np.uint8)
    res_bgr = scan_one(raw_bgr)
    assert res_bgr.output_width == 400
    assert res_bgr.output_height == 300

    # 5. NumPy 2D grayscale array
    raw_gray = np.full((250, 350), 150, dtype=np.uint8)
    res_gray = scan_one(raw_gray)
    assert res_gray.output_width == 350
    assert res_gray.output_height == 250

    # 6. Invalid types
    with pytest.raises(TypeError):
        scan_one(12345)  # type: ignore

    with pytest.raises(ValueError):
        scan_one(np.full((100, 100), 1.0, dtype=np.float32))


def test_scan_one_error_handling_invalid_file(tmp_path: Path):
    """Verify non-existent or corrupted file raises ImageDecodeError."""
    # 1. Missing file
    missing = tmp_path / "non_existent.jpg"
    with pytest.raises(ImageDecodeError) as exc_info:
        scan_one(missing)
    assert exc_info.value.code == ScannerErrorCode.IMAGE_DECODE_FAILED

    # 2. Corrupt file
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"NOT_A_VALID_IMAGE_DATA")
    with pytest.raises(ImageDecodeError) as exc_info:
        scan_one(corrupt)
    assert exc_info.value.code == ScannerErrorCode.IMAGE_DECODE_FAILED


def test_scan_one_input_immutability():
    """Verify input numpy array is never mutated during execution."""
    bgr = np.full((300, 400, 3), 100, dtype=np.uint8)
    cv2.rectangle(bgr, (50, 50), (350, 250), (255, 255, 255), -1)
    original_hash = hashlib.sha256(bgr.tobytes()).hexdigest()

    _ = scan_one(bgr, mode=ScanMode.GRAY)
    _ = scan_one(bgr, mode=ScanMode.BW)
    _ = scan_one(bgr, mode=ScanMode.COLOR)

    assert hashlib.sha256(bgr.tobytes()).hexdigest() == original_hash


def test_scan_one_mocked_detector_isolation_found():
    """Verify orchestration isolation when detector explicitly finds a document."""
    bgr = np.full((300, 400, 3), 200, dtype=np.uint8)
    corners_list = [(10.0, 10.0), (390.0, 10.0), (390.0, 290.0), (10.0, 290.0)]
    dummy_corners = np.array(corners_list, dtype=np.float32)
    fake_detection = DetectionResult(
        corners=corners_list,
        confidence=0.95,
        area_ratio=0.85,
        scale_factor=1.0,
    )
    fake_warped = WarpedDocument(
        image=np.full((280, 380, 3), 220, dtype=np.uint8),
        width=380,
        height=280,
        transform_matrix=np.eye(3, dtype=np.float64),
        source_corners=dummy_corners,
    )

    with (
        patch(
            "attendance_scanner.pipeline.orchestrator.detect_document_boundary",
            return_value=fake_detection,
        ) as mock_detect,
        patch(
            "attendance_scanner.pipeline.orchestrator.warp_perspective",
            return_value=fake_warped,
        ) as mock_warp,
    ):
        result = scan_one(bgr, mode=ScanMode.GRAY)

        mock_detect.assert_called_once()
        assert mock_warp.call_count == 2
        assert result.document_detected is True
        assert result.diagnostics.document_detected is True
        assert result.warning_codes == []
        assert result.diagnostics.warning_codes == []
        assert result.diagnostics.warning is None
        assert result.output_width == 380
        assert result.output_height == 280


def test_scan_one_normalizes_detected_portrait_output_to_landscape():
    """A detected portrait capture is rotated without changing pixel content mode."""
    bgr = np.full((500, 300, 3), 200, dtype=np.uint8)
    corners_list = [(10.0, 10.0), (290.0, 10.0), (290.0, 490.0), (10.0, 490.0)]
    dummy_corners = np.array(corners_list, dtype=np.float32)
    fake_detection = DetectionResult(
        corners=corners_list,
        confidence=0.95,
        area_ratio=0.85,
        scale_factor=1.0,
    )
    fake_warped = WarpedDocument(
        image=np.full((500, 300, 3), 220, dtype=np.uint8),
        width=300,
        height=500,
        transform_matrix=np.eye(3, dtype=np.float64),
        source_corners=dummy_corners,
    )

    with (
        patch(
            "attendance_scanner.pipeline.orchestrator.detect_document_boundary",
            return_value=fake_detection,
        ),
        patch(
            "attendance_scanner.pipeline.orchestrator.warp_perspective",
            return_value=fake_warped,
        ),
    ):
        result = scan_one(bgr, mode=ScanMode.GRAY)

    assert result.output_width == 500
    assert result.output_height == 300
    assert result.diagnostics.orientation_rotation_degrees == 90


def test_scan_one_mocked_detector_isolation_not_found():
    """Verify orchestration isolation when detector explicitly returns None."""
    bgr = np.full((300, 400, 3), 150, dtype=np.uint8)

    with (
        patch(
            "attendance_scanner.pipeline.orchestrator.detect_document_boundary",
            return_value=None,
        ) as mock_detect,
        patch(
            "attendance_scanner.pipeline.orchestrator.warp_perspective",
        ) as mock_warp,
    ):
        result = scan_one(bgr, mode=ScanMode.GRAY)

        mock_detect.assert_called_once()
        mock_warp.assert_not_called()
        assert result.document_detected is False
        assert result.diagnostics.document_detected is False
        assert result.warning_codes == [ScannerWarningCode.DOCUMENT_NOT_DETECTED.value]
        assert result.diagnostics.warning_codes == [ScannerWarningCode.DOCUMENT_NOT_DETECTED.value]
        assert result.diagnostics.warning == ScannerWarningCode.DOCUMENT_NOT_DETECTED.value
        assert result.warning == ScannerWarningCode.DOCUMENT_NOT_DETECTED.value
        assert result.output_width == 400
        assert result.output_height == 300


def test_scan_one_clipped_document_skips_warp_and_records_warning():
    """Verify orchestrator skips warp when detection is clipped and records DOCUMENT_CLIPPED."""
    bgr = np.full((300, 400, 3), 150, dtype=np.uint8)

    clipped_detection = DetectionResult(
        detected=True,
        accepted=False,
        clipped=True,
        rejection_reason=DetectionRejectionReason.DOCUMENT_CLIPPED,
        corners=[(0.0, 0.0), (350.0, 0.0), (350.0, 280.0), (0.0, 280.0)],
        confidence=0.85,
        area_ratio=0.7,
        scale_factor=1.0,
    )

    with (
        patch(
            "attendance_scanner.pipeline.orchestrator.detect_document_boundary",
            return_value=clipped_detection,
        ) as mock_detect,
        patch(
            "attendance_scanner.pipeline.orchestrator.warp_perspective",
        ) as mock_warp,
    ):
        result = scan_one(bgr, mode=ScanMode.GRAY)

        mock_detect.assert_called_once()
        mock_warp.assert_not_called()
        assert result.document_detected is False
        assert result.diagnostics.document_detected is False
        assert ScannerWarningCode.DOCUMENT_CLIPPED.value in result.warning_codes
        assert ScannerWarningCode.DOCUMENT_NOT_DETECTED.value not in result.warning_codes
        assert result.output_width == 400
        assert result.output_height == 300


def test_scan_one_enhancement_failure_raises_typed_process_error():
    """Verify internal enhancement/OpenCV exceptions are wrapped in typed ImageProcessError."""
    bgr = np.full((100, 100, 3), 100, dtype=np.uint8)

    with patch(
        "attendance_scanner.pipeline.orchestrator.enhance_image",
        side_effect=cv2.error("Simulated OpenCV enhancement error"),
    ):
        with pytest.raises(ImageProcessError) as exc_info:
            scan_one(bgr)

        assert exc_info.value.code == ScannerErrorCode.UNEXPECTED_ERROR
        assert "Enhancement failed" in str(exc_info.value)
