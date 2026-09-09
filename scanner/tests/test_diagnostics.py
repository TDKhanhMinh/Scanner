"""Tests for structured diagnostics and safe error mapping (AS-17)."""

import json
import logging
from pathlib import Path

from attendance_scanner.contracts import (
    ImageDecodeError,
    OutputNotWritableError,
    ScannerErrorCode,
)
from attendance_scanner.diagnostics import (
    configure_logging,
    describe_scanner_error,
    log_scanner_error,
)


def _restore_default_logger() -> None:
    logger = logging.getLogger("attendance_scanner")
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False


def test_known_errors_map_to_actionable_messages_without_raw_details():
    info = describe_scanner_error(
        ImageDecodeError(r"C:\private\NV01\corrupt.jpg", "corrupt header"),
        operation="file",
        relative_path="NV01/corrupt.jpg",
    )

    assert info.code == ScannerErrorCode.IMAGE_DECODE_FAILED
    assert "Không thể đọc ảnh" in info.user_message
    assert "corrupt header" not in info.user_message
    assert info.relative_path == "NV01/corrupt.jpg"

    output_info = describe_scanner_error(
        OutputNotWritableError(r"C:\private\output", "permission denied"),
        operation="output",
    )
    assert output_info.code == ScannerErrorCode.OUTPUT_NOT_WRITABLE
    assert "quyền" in output_info.user_message


def test_unknown_errors_use_fallback_code_and_safe_message():
    info = describe_scanner_error(
        RuntimeError("internal implementation detail"),
        operation="file",
        relative_path="NV01/card.jpg",
    )

    assert info.code == ScannerErrorCode.UNEXPECTED_ERROR
    assert "internal implementation detail" not in info.user_message


def test_stderr_log_is_safe_while_file_log_keeps_traceback_diagnostics(tmp_path: Path, capsys):
    log_path = tmp_path / "logs" / "attendance-scanner.log"
    configure_logging(log_path)

    try:
        try:
            raise ImageDecodeError(r"C:\private\NV01\corrupt.jpg", "corrupt header")
        except ImageDecodeError as exc:
            info = describe_scanner_error(
                exc,
                operation="file",
                relative_path="NV01/corrupt.jpg",
            )
            log_scanner_error(info, exc, operation="file")

        stderr_payload = json.loads(capsys.readouterr().err.strip())
        assert stderr_payload["errorCode"] == "IMAGE_DECODE_FAILED"
        assert "traceback" not in stderr_payload
        assert "C:\\private\\NV01\\corrupt.jpg" not in json.dumps(stderr_payload)

        file_payload = json.loads(log_path.read_text(encoding="utf-8").strip())
        assert file_payload["errorCode"] == "IMAGE_DECODE_FAILED"
        assert file_payload["relativePath"] == "NV01/corrupt.jpg"
        assert "traceback" in file_payload
        assert "corrupt header" in file_payload["technicalMessage"]
    finally:
        _restore_default_logger()
