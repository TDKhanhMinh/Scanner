"""Integration tests for the resumable bounded-concurrency batch engine (AS-12)."""

import threading
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

import attendance_scanner.cli as cli
from attendance_scanner.batch import clamp_worker_count, default_worker_count, run_batch
from attendance_scanner.contracts import (
    FileProcessingStatus,
    ImageDecodeError,
    ScanMode,
    ScannerErrorCode,
    StateError,
)
from attendance_scanner.discovery import build_incremental_scan_plan, discover_employee_folders
from attendance_scanner.events import (
    FileCompletedEvent,
    FileFailedEvent,
    FileStartedEvent,
    ScanCompletedEvent,
    deserialize_event,
)
from attendance_scanner.pdf_export import PdfExportResult
from attendance_scanner.pipeline.orchestrator import SingleScanResult
from attendance_scanner.state import ManifestStore


def test_worker_count_uses_default_and_clamps_to_supported_range():
    with patch("attendance_scanner.batch.os.cpu_count", return_value=8):
        assert default_worker_count() == 3
    with patch("attendance_scanner.batch.os.cpu_count", return_value=1):
        assert default_worker_count() == 1

    assert clamp_worker_count(None) >= 1
    assert clamp_worker_count(0) == 1
    assert clamp_worker_count(2) == 2
    assert clamp_worker_count(99) == 4


def _create_batch_context(tmp_path: Path, count: int = 4):
    input_root = tmp_path / "employees"
    output_root = tmp_path / "output"
    employee = input_root / "NV01"
    employee.mkdir(parents=True)

    for index in range(1, count + 1):
        Image.new("RGB", (32, 24), color=(index * 20, 80, 120)).save(
            employee / f"{index}.png"
        )

    discovery = discover_employee_folders(input_root)
    store = ManifestStore(state_dir=tmp_path / "state")
    manifest = store.load_manifest(input_root, output_root=output_root)
    return input_root, output_root, discovery, manifest, store


def _scan_result(mode: ScanMode, warning: bool = False) -> SingleScanResult:
    warning_codes = ["DOCUMENT_NOT_DETECTED"] if warning else []
    return SingleScanResult(
        image=np.full((24, 32), 180, dtype=np.uint8),
        document_detected=not warning,
        original_width=32,
        original_height=24,
        output_width=32,
        output_height=24,
        mode=mode,
        warning_codes=warning_codes,
    )


def _fake_export(image, target_path, *, config=None) -> PdfExportResult:
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"%PDF-1.4\n%%EOF")
    return PdfExportResult(
        pdf_path=str(target),
        file_size_bytes=target.stat().st_size,
        width_px=32,
        height_px=24,
        mode="L",
    )


def test_batch_workers_are_bounded_and_manifest_is_checkpointed(tmp_path: Path):
    _, output_root, discovery, manifest, store = _create_batch_context(tmp_path)
    active = 0
    peak_active = 0
    active_lock = threading.Lock()

    def fake_scan(source, mode, *, config=None):
        nonlocal active, peak_active
        with active_lock:
            active += 1
            peak_active = max(peak_active, active)
        time.sleep(0.03)
        with active_lock:
            active -= 1
        return _scan_result(mode)

    with (
        patch("attendance_scanner.batch.scan_one", side_effect=fake_scan),
        patch("attendance_scanner.batch.export_single_page_pdf", side_effect=_fake_export),
        patch.object(store, "save_manifest", wraps=store.save_manifest) as save_manifest,
    ):
        result = run_batch(
            discovery=discovery,
            manifest=manifest,
            output_root=output_root,
            workers=2,
            manifest_store=store,
        )

    assert peak_active <= 2
    assert result.exit_code == 0
    assert result.summary.total_images == 4
    assert result.summary.success == 4
    assert result.summary.failed == 0
    assert save_manifest.call_count == 4
    assert len(list(output_root.rglob("*.pdf"))) == 4
    assert [item.relative_path for item in result.file_results] == [
        "NV01/1.png",
        "NV01/2.png",
        "NV01/3.png",
        "NV01/4.png",
    ]

    started = [event for event in result.events if isinstance(event, FileStartedEvent)]
    completed = [event for event in result.events if isinstance(event, FileCompletedEvent)]
    finished = [event for event in result.events if isinstance(event, ScanCompletedEvent)]
    assert {event.index for event in started} == {1, 2, 3, 4}
    assert {event.total for event in started} == {4}
    assert len(completed) == 4
    assert len(finished) == 1
    assert all(
        manifest.get_entry(item.relative_path).status == FileProcessingStatus.SUCCESS
        for item in discovery.files
    )


def test_corrupt_file_isolated_and_batch_returns_exit_code_2(tmp_path: Path):
    _, output_root, discovery, manifest, store = _create_batch_context(tmp_path, count=3)

    def fake_scan(source, mode, *, config=None):
        if Path(source).name == "2.png":
            raise ImageDecodeError(str(source), "injected corrupt image")
        return _scan_result(mode)

    with (
        patch("attendance_scanner.batch.scan_one", side_effect=fake_scan),
        patch("attendance_scanner.batch.export_single_page_pdf", side_effect=_fake_export),
    ):
        result = run_batch(
            discovery=discovery,
            manifest=manifest,
            output_root=output_root,
            workers=2,
            manifest_store=store,
        )

    assert result.exit_code == 2
    assert result.summary.total_images == 3
    assert result.summary.success == 2
    assert result.summary.failed == 1
    assert len(list(output_root.rglob("*.pdf"))) == 2
    failed_events = [event for event in result.events if isinstance(event, FileFailedEvent)]
    assert len(failed_events) == 1
    assert failed_events[0].relative_path == "NV01/2.png"
    assert failed_events[0].error_code.value == "IMAGE_DECODE_FAILED"
    failed_entry = manifest.get_entry("NV01/2.png")
    assert failed_entry is not None
    assert failed_entry.status == FileProcessingStatus.FAILED


def test_warning_is_successful_terminal_state_and_is_resumable(tmp_path: Path):
    _, output_root, discovery, manifest, store = _create_batch_context(tmp_path, count=1)

    with (
        patch(
            "attendance_scanner.batch.scan_one",
            return_value=_scan_result(ScanMode.GRAY, warning=True),
        ),
        patch("attendance_scanner.batch.export_single_page_pdf", side_effect=_fake_export),
    ):
        first_result = run_batch(
            discovery=discovery,
            manifest=manifest,
            output_root=output_root,
            workers=1,
            manifest_store=store,
        )

    assert first_result.exit_code == 0
    assert first_result.summary.warning == 1
    assert first_result.summary.success == 0
    entry = manifest.get_entry("NV01/1.png")
    assert entry is not None
    assert entry.status == FileProcessingStatus.WARNING

    second_discovery = discover_employee_folders(tmp_path / "employees")
    second_plan = build_incremental_scan_plan(second_discovery, manifest, output_root)
    assert second_plan.files_to_process == 0
    assert second_discovery.files_to_process == []

    with patch("attendance_scanner.batch.scan_one") as scan_one_mock:
        second_result = run_batch(
            discovery=second_discovery,
            manifest=manifest,
            output_root=output_root,
            workers=1,
            manifest_store=store,
        )

    scan_one_mock.assert_not_called()
    assert second_result.summary.total_images == 0
    assert second_result.summary.skipped == 1
    assert second_result.exit_code == 0


def test_invalid_batch_configuration_fails_before_processing(tmp_path: Path):
    _, output_root, discovery, manifest, store = _create_batch_context(tmp_path, count=1)

    with pytest.raises(ValueError, match="Unsupported scan mode"):
        run_batch(
            discovery=discovery,
            manifest=manifest,
            output_root=output_root,
            mode="invalid",
            manifest_store=store,
        )


def test_cli_scan_batch_emits_events_and_commits_output(tmp_path: Path, monkeypatch, capsys):
    _, output_root, _, _, _store = _create_batch_context(tmp_path, count=1)
    state_store = ManifestStore(state_dir=tmp_path / "state")
    monkeypatch.setattr(cli, "ManifestStore", lambda: state_store)

    exit_code = cli.main(
        [
            "scan-batch",
            "--input",
            str(tmp_path / "employees"),
            "--output",
            str(output_root),
            "--workers",
            "1",
        ]
    )

    assert exit_code == 0
    events = [
        deserialize_event(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    assert [event.type for event in events] == [
        "scan_plan",
        "file_started",
        "file_completed",
        "scan_completed",
    ]
    assert len(list(output_root.rglob("*.pdf"))) == 1


def test_cli_state_read_failure_returns_fatal_exit_code_1(tmp_path: Path, monkeypatch, capsys):
    def raise_state_error(*args, **kwargs):
        raise StateError(
            ScannerErrorCode.STATE_READ_FAILED,
            "injected state read failure",
        )

    class FailingStore:
        def load_manifest(self, *args, **kwargs):
            return raise_state_error(*args, **kwargs)

    monkeypatch.setattr(cli, "ManifestStore", FailingStore)
    exit_code = cli.main(["scan-batch", "--input", str(tmp_path)])

    assert exit_code == 1
    assert "injected state read failure" in capsys.readouterr().err
