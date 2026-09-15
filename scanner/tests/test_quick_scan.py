"""Quick Scan workflow and CLI contract tests."""

import base64
import json
from pathlib import Path

from PIL import Image, ImageDraw

import attendance_scanner.quick_scan as quick_scan_module
from attendance_scanner.cli import main
from attendance_scanner.quick_scan import execute_quick_scan


def _write_document(path: Path) -> Path:
    image = Image.new("RGB", (320, 240), color=(40, 40, 40))
    ImageDraw.Draw(image).rectangle(
        (20, 20, 300, 220), fill=(245, 245, 245), outline=(0, 0, 0), width=4
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def test_execute_quick_scan_returns_bounded_preview_and_temporary_pdf(tmp_path: Path):
    source = _write_document(tmp_path / "document.png")
    temp_root = tmp_path / "quick-temp"

    result = execute_quick_scan(
        source,
        temp_root,
        mode="gray",
        detector_mode="classic",
        orientation="auto",
    )

    assert result.success is True
    assert result.error_code is None
    assert result.temp_pdf_path is not None
    assert Path(result.temp_pdf_path).is_file()
    assert result.detection_preview is not None
    assert result.processed_preview_data_url is not None
    assert result.processed_preview_data_url.startswith("data:image/jpeg;base64,")
    decoded = base64.b64decode(result.processed_preview_data_url.split(",", 1)[1])
    assert len(decoded) <= 1_500_000
    assert len(result.processed_preview_data_url) <= 2_000_000


def test_execute_quick_scan_auto_normalizes_to_landscape(tmp_path: Path, monkeypatch):
    source = _write_document(tmp_path / "document.png")
    captured = {}
    original_scan_one = quick_scan_module.scan_one

    def capture_orientation(*args, **kwargs):
        captured["preferred_orientation"] = kwargs.get("preferred_orientation")
        return original_scan_one(*args, **kwargs)

    monkeypatch.setattr(quick_scan_module, "scan_one", capture_orientation)

    execute_quick_scan(
        source,
        tmp_path / "quick-temp",
        detector_mode="classic",
        orientation="auto",
    )

    assert captured["preferred_orientation"] == "landscape"


def test_execute_quick_scan_returns_structured_failure_for_unsupported_input(tmp_path: Path):
    source = tmp_path / "document.txt"
    source.write_text("not an image", encoding="utf-8")

    result = execute_quick_scan(source, tmp_path / "quick-temp")

    assert result.success is False
    assert result.error_code is not None
    assert result.temp_pdf_path is None
    assert result.message


def test_execute_quick_scan_hides_unexpected_technical_details(
    tmp_path: Path,
    monkeypatch,
):
    source = _write_document(tmp_path / "document.png")

    def fail_scan(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("D:/private/path/internal OpenCV detail")

    monkeypatch.setattr(quick_scan_module, "scan_one", fail_scan)

    result = execute_quick_scan(source, tmp_path / "quick-temp")

    assert result.success is False
    assert result.message == "Quick Scan gặp lỗi nội bộ."
    assert "D:/private/path" not in (result.message or "")


def test_scan_one_cli_emits_one_jsonl_event(tmp_path: Path, capsys):
    source = _write_document(tmp_path / "document.jpg")
    temp_root = tmp_path / "quick-temp"

    exit_code = main(
        [
            "scan-one",
            "--input",
            str(source),
            "--temp-root",
            str(temp_root),
            "--detector-mode",
            "classic",
        ]
    )

    captured = capsys.readouterr()
    events = [json.loads(line) for line in captured.out.splitlines() if line.strip()]
    assert exit_code == 0
    assert len(events) == 1
    assert events[0]["type"] == "quick_scan_completed"
    assert events[0]["success"] is True
    assert events[0]["processedPreviewDataUrl"].startswith("data:image/jpeg;base64,")
