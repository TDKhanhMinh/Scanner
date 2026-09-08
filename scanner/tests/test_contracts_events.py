"""Unit tests for scanner contracts and JSONL event protocol."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from attendance_scanner.contracts import (
    PROTOCOL_VERSION,
    BatchSummary,
    DiscoveredFile,
    FileClassification,
    FileProcessingStatus,
    FileResult,
    ScanBatchRequest,
    ScanMode,
    ScannerErrorCode,
    ScanPlan,
)
from attendance_scanner.events import (
    FileCompletedEvent,
    FileFailedEvent,
    FileStartedEvent,
    ScanCompletedEvent,
    ScanPlanEvent,
    deserialize_event,
    serialize_event,
)


def test_contracts_models_instantiation():
    """Verify that domain contract models instantiate and validate fields."""
    file_meta = DiscoveredFile(
        employee_name="NguyenVanA",
        file_name="page1.jpg",
        relative_path="NguyenVanA/page1.jpg",
        absolute_path="/abs/NguyenVanA/page1.jpg",
        size=1024,
        mtime_ns=1700000000,
        classification=FileClassification.NEW,
        target_relative_pdf="NguyenVanA/page1.pdf",
    )
    assert file_meta.employee_name == "NguyenVanA"
    assert file_meta.classification == FileClassification.NEW

    plan = ScanPlan(
        input_root="/input",
        output_root="/output",
        employees=5,
        total_images=20,
        new=18,
        modified=2,
        unchanged=0,
        files_to_process=20,
        unsupported_count=2,
        collisions=[],
    )
    assert plan.employees == 5
    assert plan.total_employees == 5
    assert plan.new == 18
    assert plan.new_count == 18
    assert plan.files_to_process == 20
    assert plan.unsupported_count == 2

    req = ScanBatchRequest(
        input_root="/input",
        output_root="/output",
        mode=ScanMode.GRAY,
        workers=3,
    )
    assert req.mode == ScanMode.GRAY

    result = FileResult(
        relative_path="NguyenVanA/page1.jpg",
        employee_name="NguyenVanA",
        target_relative_pdf="NguyenVanA/page1.pdf",
        status=FileProcessingStatus.SUCCESS,
        document_detected=True,
        duration_ms=350,
    )
    assert result.status == FileProcessingStatus.SUCCESS

    summary = BatchSummary(
        total_images=20,
        success=19,
        failed=1,
        warning=0,
        skipped=0,
        duration_ms=5400,
    )
    assert summary.total_images == 20


def test_serialize_event_single_line():
    """Verify serialize_event produces a single line with camelCase keys."""
    event = FileStartedEvent(
        relative_path="A/img.jpg",
        employee_name="A",
        index=1,
        total=10,
    )
    line = serialize_event(event)
    assert "\n" not in line
    assert "\r" not in line
    data = json.loads(line)
    assert data["type"] == "file_started"
    assert data["protocolVersion"] == PROTOCOL_VERSION
    assert data["relativePath"] == "A/img.jpg"
    assert data["employeeName"] == "A"
    assert data["index"] == 1
    assert data["total"] == 10


def test_deserialize_all_event_types():
    """Verify deserialization of all 5 event types into concrete models."""
    # 1. scan_plan with System Design canonical fields
    plan_json = json.dumps(
        {
            "protocolVersion": 1,
            "type": "scan_plan",
            "timestamp": "2026-09-07T00:00:00Z",
            "inputRoot": "/input",
            "outputRoot": "/output",
            "employees": 3,
            "totalImages": 12,
            "new": 10,
            "modified": 2,
            "unchanged": 0,
            "filesToProcess": 12,
            "collisions": [],
        }
    )
    plan_event = deserialize_event(plan_json)
    assert isinstance(plan_event, ScanPlanEvent)
    assert plan_event.employees == 3
    assert plan_event.total_employees == 3
    assert plan_event.new == 10
    assert plan_event.total_images == 12

    # 2. file_started
    started_json = json.dumps(
        {
            "protocolVersion": 1,
            "type": "file_started",
            "timestamp": "2026-09-07T00:00:01Z",
            "relativePath": "B/pic.png",
            "employeeName": "B",
            "index": 2,
            "total": 12,
        }
    )
    started_event = deserialize_event(started_json)
    assert isinstance(started_event, FileStartedEvent)
    assert started_event.index == 2

    # 3. file_completed
    completed_json = json.dumps(
        {
            "protocolVersion": 1,
            "type": "file_completed",
            "timestamp": "2026-09-07T00:00:02Z",
            "relativePath": "B/pic.png",
            "employeeName": "B",
            "outputRelativePath": "B/pic.pdf",
            "documentDetected": True,
            "warning": None,
            "durationMs": 400,
        }
    )
    completed_event = deserialize_event(completed_json)
    assert isinstance(completed_event, FileCompletedEvent)
    assert completed_event.document_detected is True
    assert completed_event.duration_ms == 400

    # 4. file_failed
    failed_json = json.dumps(
        {
            "protocolVersion": 1,
            "type": "file_failed",
            "timestamp": "2026-09-07T00:00:03Z",
            "relativePath": "C/bad.jpg",
            "employeeName": "C",
            "errorCode": "IMAGE_DECODE_FAILED",
            "message": "Corrupted image file",
        }
    )
    failed_event = deserialize_event(failed_json)
    assert isinstance(failed_event, FileFailedEvent)
    assert failed_event.error_code == ScannerErrorCode.IMAGE_DECODE_FAILED

    # 5. scan_completed
    summary_json = json.dumps(
        {
            "protocolVersion": 1,
            "type": "scan_completed",
            "timestamp": "2026-09-07T00:00:04Z",
            "totalProcessed": 12,
            "success": 11,
            "failed": 1,
            "warning": 0,
            "skipped": 0,
            "durationMs": 3500,
        }
    )
    summary_event = deserialize_event(summary_json)
    assert isinstance(summary_event, ScanCompletedEvent)
    assert summary_event.total_processed == 12


def test_deserialize_forward_compatibility_extra_fields():
    """Verify that unknown extra fields do not cause deserialization errors."""
    json_with_extra = json.dumps(
        {
            "protocolVersion": 1,
            "type": "file_completed",
            "timestamp": "2026-09-07T00:00:02Z",
            "relativePath": "B/pic.png",
            "employeeName": "B",
            "outputRelativePath": "B/pic.pdf",
            "documentDetected": True,
            "durationMs": 400,
            "futureTelemetryField": "v2_experimental",
            "nestedFutureData": {"someKey": 123},
        }
    )
    event = deserialize_event(json_with_extra)
    assert isinstance(event, FileCompletedEvent)
    assert getattr(event, "futureTelemetryField", None) == "v2_experimental"


def test_deserialize_validation_error():
    """Verify that invalid event type or missing required fields raise error."""
    with pytest.raises(ValueError):
        deserialize_event("   ")

    # Missing protocolVersion on wire
    with pytest.raises(ValidationError):
        deserialize_event(
            json.dumps(
                {
                    "type": "file_started",
                    "timestamp": "2026-09-07T00:00:00Z",
                    "relativePath": "A/1.jpg",
                    "employeeName": "A",
                    "index": 1,
                    "total": 1,
                }
            )
        )

    # Invalid protocolVersion (e.g. 2)
    with pytest.raises(ValidationError):
        deserialize_event(
            json.dumps(
                {
                    "protocolVersion": 2,
                    "type": "file_started",
                    "timestamp": "2026-09-07T00:00:00Z",
                    "relativePath": "A/1.jpg",
                    "employeeName": "A",
                    "index": 1,
                    "total": 1,
                }
            )
        )

    # Missing timestamp on wire
    with pytest.raises(ValidationError):
        deserialize_event(
            json.dumps(
                {
                    "protocolVersion": 1,
                    "type": "file_started",
                    "relativePath": "A/1.jpg",
                    "employeeName": "A",
                    "index": 1,
                    "total": 1,
                }
            )
        )

    # Empty timestamp
    with pytest.raises(ValidationError):
        deserialize_event(
            json.dumps(
                {
                    "protocolVersion": 1,
                    "timestamp": "   ",
                    "type": "file_started",
                    "relativePath": "A/1.jpg",
                    "employeeName": "A",
                    "index": 1,
                    "total": 1,
                }
            )
        )

    # Invalid errorCode
    with pytest.raises(ValidationError):
        deserialize_event(
            json.dumps(
                {
                    "protocolVersion": 1,
                    "timestamp": "2026-09-07T00:00:00Z",
                    "type": "file_failed",
                    "relativePath": "A/1.jpg",
                    "employeeName": "A",
                    "errorCode": "INVALID_ERROR_CODE",
                    "message": "err",
                }
            )
        )

    # Unknown event type
    with pytest.raises(ValidationError):
        deserialize_event(
            json.dumps(
                {
                    "protocolVersion": 1,
                    "timestamp": "2026-09-07T00:00:00Z",
                    "type": "unknown_event_type",
                }
            )
        )


def test_load_all_fixture_files():
    """Verify that all JSON fixture files in fixtures/scanner/events parse properly."""
    fixtures_dir = Path(__file__).resolve().parents[2] / "fixtures" / "scanner" / "events"
    assert fixtures_dir.is_dir(), f"Fixtures directory not found: {fixtures_dir}"

    fixture_files = list(fixtures_dir.glob("*.json"))
    assert len(fixture_files) >= 5

    for fpath in fixture_files:
        content = fpath.read_text(encoding="utf-8")
        event = deserialize_event(content)
        assert event.type in [
            "scan_plan",
            "file_started",
            "file_completed",
            "file_failed",
            "scan_completed",
        ]
        assert event.protocol_version == PROTOCOL_VERSION


def test_cli_emits_valid_jsonl(capsys, tmp_path):
    """Verify CLI subcommands output valid JSONL parseable by deserialize_event."""
    from attendance_scanner.cli import main

    # Plan
    ret = main(["plan", "--input", str(tmp_path), "--output", "test_output"])
    assert ret == 0
    captured = capsys.readouterr()
    lines = [line for line in captured.out.strip().split("\n") if line.strip()]
    assert len(lines) == 1
    plan_event = deserialize_event(lines[0])
    assert isinstance(plan_event, ScanPlanEvent)
    assert plan_event.input_root == str(tmp_path.resolve())

    # Scan-batch
    ret = main(["scan-batch", "--input", str(tmp_path), "--output", "test_output"])
    assert ret == 0
    captured = capsys.readouterr()
    lines = [line for line in captured.out.strip().split("\n") if line.strip()]
    assert len(lines) == 2
    ev1 = deserialize_event(lines[0])
    ev2 = deserialize_event(lines[1])
    assert isinstance(ev1, ScanPlanEvent)
    assert isinstance(ev2, ScanCompletedEvent)
