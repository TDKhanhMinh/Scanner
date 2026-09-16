"""One-image Quick Scan workflow with bounded previews and temporary PDF output."""

from __future__ import annotations

import base64
import hashlib
import logging
import time
from pathlib import Path
from typing import Literal, Optional, Union

import cv2
import numpy as np

from attendance_scanner.contracts import (
    ScanMode,
    ScannerError,
    ScannerErrorCode,
)
from attendance_scanner.events import QuickScanCompletedEvent
from attendance_scanner.pdf_export import PdfExportConfig, export_single_page_pdf
from attendance_scanner.pipeline.orchestrator import scan_one

QUICK_SCAN_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
QuickScanOrientation = Literal["auto", "landscape", "portrait"]
LOGGER = logging.getLogger(__name__)

_SAFE_ERROR_MESSAGES = {
    ScannerErrorCode.INVALID_REQUEST: "Yêu cầu Quick Scan không hợp lệ.",
    ScannerErrorCode.IMAGE_DECODE_FAILED: "Không thể đọc ảnh đầu vào.",
    ScannerErrorCode.OUTPUT_NOT_WRITABLE: "Không thể ghi file PDF tạm.",
    ScannerErrorCode.PDF_WRITE_FAILED: "Không thể tạo file PDF tạm.",
    ScannerErrorCode.UNEXPECTED_ERROR: "Quick Scan gặp lỗi nội bộ.",
}


def _encode_processed_preview(image: np.ndarray) -> str:
    """Encode a processed image under the bounded preview wire contract."""
    if image.dtype != np.uint8 or image.ndim not in {2, 3}:
        raise ScannerError(
            ScannerErrorCode.UNEXPECTED_ERROR,
            "Processed preview must be an 8-bit grayscale or BGR image",
        )
    height, width = image.shape[:2]
    target_dimension = 1280
    for _ in range(5):
        scale = min(1.0, target_dimension / float(max(height, width)))
        preview = image
        if scale < 1.0:
            preview = cv2.resize(
                image,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        for quality in (80, 72, 60, 50):
            success, encoded = cv2.imencode(
                ".jpg",
                preview,
                [int(cv2.IMWRITE_JPEG_QUALITY), quality],
            )
            if not success:
                continue
            if len(encoded) > 1_500_000:
                continue
            payload = base64.b64encode(encoded.tobytes()).decode("ascii")
            data_url = f"data:image/jpeg;base64,{payload}"
            if len(data_url) <= 2_000_000:
                return data_url
        target_dimension = max(256, round(target_dimension * 0.75))
    raise ScannerError(
        ScannerErrorCode.UNEXPECTED_ERROR,
        "Processed preview exceeded the bounded IPC size limit",
    )


def _orientation_for_pipeline(
    orientation: QuickScanOrientation,
) -> Literal["natural", "landscape", "portrait"]:
    return "landscape" if orientation == "auto" else orientation


def _pdf_orientation(image: np.ndarray) -> Literal["LANDSCAPE", "PORTRAIT"]:
    height, width = image.shape[:2]
    return "PORTRAIT" if height > width else "LANDSCAPE"


def _temp_pdf_path(source: Path, temp_root: Path) -> Path:
    stat = source.stat()
    identity = f"{source.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()[:24]
    return temp_root / f"quick_{digest}.pdf"


def execute_quick_scan(
    input_path: Union[str, Path],
    temp_root: Union[str, Path],
    *,
    mode: Union[ScanMode, str] = ScanMode.GRAY,
    detector_mode: Optional[str] = None,
    orientation: QuickScanOrientation = "auto",
    debug_diagnostics: bool = False,
) -> QuickScanCompletedEvent:
    """Process one image, build previews, and export a temporary PDF."""
    started_at = time.perf_counter()
    source = Path(input_path).expanduser()
    temporary_pdf: Optional[Path] = None
    try:
        if not source.is_file():
            raise ScannerError(
                ScannerErrorCode.IMAGE_DECODE_FAILED,
                f"Input image does not exist: {source}",
            )
        if source.suffix.lower() not in QUICK_SCAN_SUPPORTED_EXTENSIONS:
            raise ScannerError(
                ScannerErrorCode.INVALID_REQUEST,
                f"Unsupported quick-scan image extension: {source.suffix}",
            )
        root = Path(temp_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        temporary_pdf = _temp_pdf_path(source, root)
        result = scan_one(
            source,
            mode=mode,
            detector_mode=detector_mode,
            debug_diagnostics=debug_diagnostics,
            force_preview=True,
            preferred_orientation=_orientation_for_pipeline(orientation),
        )
        processed_preview = _encode_processed_preview(result.image)
        export_single_page_pdf(
            result.image,
            temporary_pdf,
            config=PdfExportConfig(page_orientation=_pdf_orientation(result.image)),
        )
        return QuickScanCompletedEvent(
            success=True,
            input_path=str(source.resolve()),
            temp_pdf_path=str(temporary_pdf),
            document_detected=result.document_detected,
            duration_ms=round((time.perf_counter() - started_at) * 1000.0),
            detection_preview=result.detection_preview,
            processed_preview_data_url=processed_preview,
            error_code=None,
            message=None,
            warning=result.warning,
            occlusion_risk=result.occlusion_risk,
        )
    except ScannerError as exc:
        if temporary_pdf is not None:
            temporary_pdf.unlink(missing_ok=True)
        return QuickScanCompletedEvent(
            success=False,
            input_path=str(source),
            temp_pdf_path=None,
            document_detected=False,
            duration_ms=round((time.perf_counter() - started_at) * 1000.0),
            detection_preview=None,
            processed_preview_data_url=None,
            error_code=exc.code,
            message=_SAFE_ERROR_MESSAGES.get(exc.code, "Không thể hoàn tất Quick Scan."),
            warning=None,
            occlusion_risk=False,
        )
    except Exception:  # noqa: BLE001 - quick scan must preserve a wire-safe error
        LOGGER.exception("Quick scan failed")
        if temporary_pdf is not None:
            temporary_pdf.unlink(missing_ok=True)
        return QuickScanCompletedEvent(
            success=False,
            input_path=str(source),
            temp_pdf_path=None,
            document_detected=False,
            duration_ms=round((time.perf_counter() - started_at) * 1000.0),
            detection_preview=None,
            processed_preview_data_url=None,
            error_code=ScannerErrorCode.UNEXPECTED_ERROR,
            message="Quick Scan gặp lỗi nội bộ.",
            warning=None,
            occlusion_risk=False,
        )


__all__ = [
    "QUICK_SCAN_SUPPORTED_EXTENSIONS",
    "execute_quick_scan",
]
