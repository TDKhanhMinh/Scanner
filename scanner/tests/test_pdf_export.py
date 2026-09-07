"""Unit and integration tests for atomic single-page PDF export (export_single_page_pdf).

Covers:
- Grayscale array export (1-channel uint8, mode="L").
- B&W binary array export (mode="L").
- Color BGR array export with verified BGR -> RGB channel order.
- SingleScanResult direct pipeline integration.
- Direct PIL Image input.
- Atomic replacement of existing PDF targets.
- Injected write/commit failure cleanup (safety guarantee: old PDF preserved, temp files unlinked).
- Lazy creation of nested parent directories.
- Invalid input error handling (raises PdfWriteError).
"""

import io
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from PIL import Image, PdfParser

from attendance_scanner.contracts import (
    PdfWriteError,
    ScanMode,
    ScannerErrorCode,
)
from attendance_scanner.pdf_export import (
    PdfExportConfig,
    PdfExportResult,
    export_single_page_pdf,
)
from attendance_scanner.pipeline.orchestrator import scan_one


def _verify_pdf_is_single_page(pdf_path: Path) -> None:
    """Verify that a generated PDF exists, has %PDF header, and contains exactly 1 page."""
    assert pdf_path.exists()
    content = pdf_path.read_bytes()
    assert content.startswith(b"%PDF-")
    assert b"%%EOF" in content

    # Verify using Pillow native PDF parser
    parser = PdfParser.PdfParser(filename=str(pdf_path))
    assert len(parser.pages) == 1

    page_obj = parser.read_indirect(parser.pages[0])
    assert b"Resources" in page_obj
    assert b"XObject" in page_obj[b"Resources"]
    assert b"image" in page_obj[b"Resources"][b"XObject"]


def test_export_single_page_pdf_grayscale(tmp_path: Path):
    """Verify exporting 2D grayscale uint8 array creates a valid 1-page PDF."""
    target_pdf = tmp_path / "gray_doc.pdf"
    h, w = 400, 300
    gray_arr = np.full((h, w), 180, dtype=np.uint8)
    # Draw dark rectangle representing text/table
    cv2.rectangle(gray_arr, (40, 40), (260, 360), 30, -1)

    result = export_single_page_pdf(gray_arr, target_pdf, config=PdfExportConfig(dpi=150.0))

    assert isinstance(result, PdfExportResult)
    assert result.pdf_path == str(target_pdf.resolve())
    assert result.page_count == 1
    assert result.width_px == w
    assert result.height_px == h
    assert result.mode == "L"
    assert result.file_size_bytes > 100

    _verify_pdf_is_single_page(target_pdf)


def test_export_single_page_pdf_bw_binary(tmp_path: Path):
    """Verify exporting binary {0, 255} array creates a valid 1-page PDF."""
    target_pdf = tmp_path / "bw_doc.pdf"
    bw_arr = np.zeros((200, 200), dtype=np.uint8)
    bw_arr[50:150, 50:150] = 255

    result = export_single_page_pdf(bw_arr, target_pdf)

    assert result.page_count == 1
    assert result.mode == "L"
    _verify_pdf_is_single_page(target_pdf)


def test_export_single_page_pdf_color_bgr_channel_order(tmp_path: Path):
    """Verify OpenCV BGR array is converted to RGB correctly preserving stamp/ink color."""
    target_pdf = tmp_path / "color_doc.pdf"
    h, w = 100, 100
    bgr_arr = np.zeros((h, w, 3), dtype=np.uint8)

    # Top-left 50x50: Pure Red in OpenCV BGR is (B=0, G=0, R=255)
    bgr_arr[0:50, 0:50] = [0, 0, 255]
    # Bottom-right 50x50: Pure Blue in OpenCV BGR is (B=255, G=0, R=0)
    bgr_arr[50:100, 50:100] = [255, 0, 0]

    result = export_single_page_pdf(bgr_arr, target_pdf)

    assert result.page_count == 1
    assert result.mode == "RGB"
    assert result.width_px == 100
    assert result.height_px == 100
    _verify_pdf_is_single_page(target_pdf)

    # Extract embedded image stream from PDF to verify color channel mapping
    parser = PdfParser.PdfParser(filename=str(target_pdf))
    page_obj = parser.read_indirect(parser.pages[0])
    stream = parser.read_indirect(page_obj[b"Resources"][b"XObject"][b"image"])
    embedded_img = Image.open(io.BytesIO(stream.buf))
    assert embedded_img.mode == "RGB"

    # Top-left must be Red: (255, 0, 0)
    tl_pixel = embedded_img.getpixel((25, 25))
    assert isinstance(tl_pixel, tuple)
    assert tl_pixel[0] > 200 and tl_pixel[2] < 50, f"Expected Red, got {tl_pixel}"

    # Bottom-right must be Blue: (0, 0, 255)
    br_pixel = embedded_img.getpixel((75, 75))
    assert isinstance(br_pixel, tuple)
    assert br_pixel[2] > 200 and br_pixel[0] < 50, f"Expected Blue, got {br_pixel}"


def test_export_single_page_pdf_with_single_scan_result(tmp_path: Path):
    """Verify SingleScanResult from scan_one can be passed directly to export_single_page_pdf."""
    # 1. Create synthetic source file
    img_path = tmp_path / "sample.png"
    Image.new("RGB", (300, 200), color=(50, 50, 50)).save(img_path)

    # 2. Run scan_one in Color mode
    scan_res = scan_one(img_path, mode=ScanMode.COLOR)

    # 3. Export to PDF directly
    target_pdf = tmp_path / "scanned_output.pdf"
    result = export_single_page_pdf(scan_res, target_pdf)

    assert result.page_count == 1
    assert result.width_px == scan_res.output_width
    assert result.height_px == scan_res.output_height
    _verify_pdf_is_single_page(target_pdf)


def test_export_single_page_pdf_direct_pil_image(tmp_path: Path):
    """Verify direct PIL.Image.Image instance is exported correctly."""
    target_pdf = tmp_path / "pil_doc.pdf"
    pil_img = Image.new("RGB", (120, 80), color=(200, 150, 100))

    result = export_single_page_pdf(pil_img, target_pdf)

    assert result.page_count == 1
    assert result.width_px == 120
    assert result.height_px == 80
    _verify_pdf_is_single_page(target_pdf)


def test_export_single_page_pdf_atomic_replace_existing(tmp_path: Path):
    """Verify existing PDF at target_path is atomically replaced with new content."""
    target_pdf = tmp_path / "existing.pdf"
    old_content = b"%PDF-1.4\nOLD_OBSOLETE_CONTENT\n%%EOF"
    target_pdf.write_bytes(old_content)

    new_arr = np.full((150, 150), 220, dtype=np.uint8)
    result = export_single_page_pdf(new_arr, target_pdf)

    assert result.pdf_path == str(target_pdf.resolve())
    assert target_pdf.read_bytes() != old_content
    _verify_pdf_is_single_page(target_pdf)

    # Ensure no leftover temp files exist
    temp_files = list(tmp_path.glob(".*.tmp.*"))
    assert temp_files == []


def test_export_single_page_pdf_failure_safety_and_cleanup(tmp_path: Path):
    """Verify injected commit failure unlinks temp file and preserves existing target intact."""
    target_pdf = tmp_path / "precious.pdf"
    original_bytes = b"%PDF-1.4\nORIGINAL_PRESERVED_CONTENT\n%%EOF"
    target_pdf.write_bytes(original_bytes)

    new_arr = np.full((100, 100), 200, dtype=np.uint8)

    # Simulate filesystem replace failure (e.g. permission or lock error)
    with patch("os.replace", side_effect=OSError("Simulated permission denied")):
        with pytest.raises(PdfWriteError) as exc_info:
            export_single_page_pdf(new_arr, target_pdf)

        assert exc_info.value.code == ScannerErrorCode.PDF_WRITE_FAILED
        assert "Simulated permission denied" in str(exc_info.value)

    # 1. Target file must remain unchanged
    assert target_pdf.read_bytes() == original_bytes

    # 2. Temporary files must be cleaned up
    temp_files = list(tmp_path.glob(".*.tmp.*"))
    assert temp_files == []


def test_export_single_page_pdf_lazy_parent_directory_creation(tmp_path: Path):
    """Verify nested non-existent parent directories are created lazily."""
    nested_pdf = tmp_path / "a" / "b" / "c" / "deep_doc.pdf"
    assert not nested_pdf.parent.exists()

    arr = np.full((80, 80), 100, dtype=np.uint8)
    result = export_single_page_pdf(arr, nested_pdf)

    assert nested_pdf.parent.exists()
    assert result.page_count == 1
    _verify_pdf_is_single_page(nested_pdf)


def test_export_single_page_pdf_invalid_input_raises_pdf_write_error(tmp_path: Path):
    """Verify invalid input types or float arrays raise PdfWriteError."""
    target_pdf = tmp_path / "err.pdf"

    # 1. Invalid dtype
    float_arr = np.full((50, 50), 0.5, dtype=np.float32)
    with pytest.raises(PdfWriteError) as exc_info:
        export_single_page_pdf(float_arr, target_pdf)
    assert exc_info.value.code == ScannerErrorCode.PDF_WRITE_FAILED

    # 2. Unsupported type
    with pytest.raises(PdfWriteError) as exc_info:
        export_single_page_pdf(12345, target_pdf)  # type: ignore
    assert exc_info.value.code == ScannerErrorCode.PDF_WRITE_FAILED
