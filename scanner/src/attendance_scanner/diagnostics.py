"""Structured diagnostics and user-safe scanner error mapping."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Literal, Optional

from attendance_scanner.contracts import ScannerError, ScannerErrorCode

LOGGER_NAME = "attendance_scanner"
ScannerOperation = Literal["request", "plan", "file", "output", "state", "unknown"]

_USER_MESSAGES: Dict[ScannerErrorCode, str] = {
    ScannerErrorCode.INVALID_REQUEST: (
        "Yêu cầu quét chưa hợp lệ. Hãy kiểm tra lại các tùy chọn và đường dẫn."
    ),
    ScannerErrorCode.INVALID_INPUT_ROOT: (
        "Không tìm thấy hoặc không thể truy cập thư mục ảnh gốc. "
        "Hãy kiểm tra đường dẫn và quyền truy cập."
    ),
    ScannerErrorCode.OUTPUT_NOT_WRITABLE: (
        "Không thể ghi vào thư mục xuất PDF. "
        "Hãy chọn thư mục khác hoặc kiểm tra quyền truy cập."
    ),
    ScannerErrorCode.IMAGE_DECODE_FAILED: (
        "Không thể đọc ảnh này. Hãy kiểm tra file có bị hỏng "
        "và thuộc định dạng được hỗ trợ."
    ),
    ScannerErrorCode.PDF_WRITE_FAILED: (
        "Không thể tạo file PDF. Hãy kiểm tra dung lượng "
        "và quyền ghi của thư mục xuất."
    ),
    ScannerErrorCode.STATE_READ_FAILED: (
        "Không thể đọc trạng thái quét trước đó. "
        "Hãy thử lại hoặc chọn lại thư mục."
    ),
    ScannerErrorCode.STATE_WRITE_FAILED: (
        "Không thể lưu trạng thái quét. "
        "Hãy kiểm tra quyền ghi của thư mục ứng dụng."
    ),
    ScannerErrorCode.OUTPUT_COLLISION: (
        "Tên file PDF bị trùng. Hãy đổi tên ảnh hoặc chọn thư mục xuất khác."
    ),
    ScannerErrorCode.UNEXPECTED_ERROR: (
        "Đã xảy ra lỗi không xác định khi quét. Vui lòng thử lại; "
        "nếu lỗi lặp lại, gửi mã cho bộ phận hỗ trợ."
    ),
}


@dataclass(frozen=True)
class ScannerErrorInfo:
    """Stable error identity plus separate UI and diagnostic messages."""

    code: ScannerErrorCode
    user_message: str
    technical_message: str
    relative_path: Optional[str] = None

    def log_extra(self, operation: ScannerOperation) -> Dict[str, object]:
        """Return JSON-log fields without exposing absolute paths to the UI stream."""
        extra: Dict[str, object] = {
            "event_name": "scanner_error",
            "error_code": self.code.value,
            "operation": operation,
            "technical_message": self.technical_message,
        }
        if self.relative_path:
            extra["relative_path"] = self.relative_path.replace("\\", "/")
        return extra


def scanner_error_user_message(code: ScannerErrorCode) -> str:
    """Return the actionable Vietnamese message for a stable error code."""
    return _USER_MESSAGES[code]


def _technical_message(exc: BaseException) -> str:
    """Keep diagnostics bounded and avoid dumping binary payloads into logs."""
    if isinstance(exc, (bytes, bytearray, memoryview)):
        return f"{type(exc).__name__} payload omitted from diagnostics"
    message = str(exc) or type(exc).__name__
    return message[:2000] + ("…" if len(message) > 2000 else "")


def describe_scanner_error(
    exc: BaseException,
    *,
    operation: ScannerOperation = "unknown",
    relative_path: Optional[str] = None,
) -> ScannerErrorInfo:
    """Map exceptions to a stable code and a safe, actionable UI message."""
    if isinstance(exc, ScannerError):
        code = exc.code
    elif isinstance(exc, PermissionError) and operation in {"output", "state"}:
        code = (
            ScannerErrorCode.STATE_WRITE_FAILED
            if operation == "state"
            else ScannerErrorCode.OUTPUT_NOT_WRITABLE
        )
    elif (
        isinstance(exc, (FileNotFoundError, IsADirectoryError, NotADirectoryError))
        and operation == "file"
    ):
        code = ScannerErrorCode.IMAGE_DECODE_FAILED
    elif isinstance(exc, ValueError) and operation == "request":
        code = ScannerErrorCode.INVALID_REQUEST
    else:
        code = ScannerErrorCode.UNEXPECTED_ERROR

    return ScannerErrorInfo(
        code=code,
        user_message=scanner_error_user_message(code),
        technical_message=_technical_message(exc),
        relative_path=relative_path,
    )


class StructuredLogFormatter(logging.Formatter):
    """Emit one JSON object per line for sidecar and file diagnostics."""

    def __init__(self, *, include_diagnostics: bool) -> None:
        super().__init__()
        self.include_diagnostics = include_diagnostics

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, object] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "event": getattr(record, "event_name", "log"),
            "message": record.getMessage(),
        }
        for record_key, json_key in (
            ("error_code", "errorCode"),
            ("operation", "operation"),
            ("relative_path", "relativePath"),
        ):
            value = getattr(record, record_key, None)
            if value is not None:
                payload[json_key] = value

        if self.include_diagnostics:
            technical_message = getattr(record, "technical_message", None)
            if technical_message:
                payload["technicalMessage"] = technical_message
            if record.exc_info:
                payload["traceback"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_logging(log_path: Optional[Path] = None) -> None:
    """Configure safe JSON diagnostics on stderr and detailed diagnostics in a file."""
    package_logger = logging.getLogger(LOGGER_NAME)
    package_logger.setLevel(logging.INFO)
    package_logger.propagate = False
    for handler in package_logger.handlers[:]:
        package_logger.removeHandler(handler)
        handler.close()

    stderr_handler = logging.StreamHandler()
    stderr_handler.setFormatter(StructuredLogFormatter(include_diagnostics=False))
    package_logger.addHandler(stderr_handler)

    if log_path is None:
        return

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
            delay=False,
        )
    except OSError:
        return
    file_handler.setFormatter(StructuredLogFormatter(include_diagnostics=True))
    package_logger.addHandler(file_handler)


def log_scanner_error(
    info: ScannerErrorInfo,
    exc: BaseException,
    *,
    operation: ScannerOperation,
) -> None:
    """Write a safe UI-facing error and a detailed traceback to configured diagnostics."""
    logging.getLogger(LOGGER_NAME).error(
        info.user_message,
        extra=info.log_extra(operation),
        exc_info=(type(exc), exc, exc.__traceback__),
    )


if not logging.getLogger(LOGGER_NAME).handlers:
    logging.getLogger(LOGGER_NAME).addHandler(logging.NullHandler())
logging.getLogger(LOGGER_NAME).propagate = False
