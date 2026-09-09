"""Acceptance tests for manifest-backed incremental planning (Task AS-11)."""

import os
from pathlib import Path

import attendance_scanner.cli as cli
from attendance_scanner.contracts import (
    FileClassification,
    FileProcessingStatus,
    ScanPlan,
)
from attendance_scanner.discovery import (
    build_incremental_scan_plan,
    classify_discovered_files,
    discover_employee_folders,
)
from attendance_scanner.events import ScanPlanEvent, deserialize_event
from attendance_scanner.fingerprint import compute_fast_fingerprint, compute_sha256
from attendance_scanner.state import ManifestEntry, ManifestStore


def _make_manifest(
    input_root: Path,
    output_root: Path,
    source: Path,
    *,
    status: FileProcessingStatus = FileProcessingStatus.SUCCESS,
    pipeline_version: str = "0.2.0",
    output_relative_path: str | None = None,
):
    store = ManifestStore(state_dir=input_root.parent / "state")
    manifest = store.load_manifest(input_root, output_root=output_root)
    size, mtime_ns = compute_fast_fingerprint(source)
    relative_path = source.relative_to(input_root).as_posix()
    if output_relative_path is None:
        output_relative_path = relative_path.rsplit(".", 1)[0] + ".pdf"
    manifest.set_entry(
        ManifestEntry(
            relative_path=relative_path,
            size=size,
            mtime_ns=mtime_ns,
            sha256=compute_sha256(source),
            output_relative_path=output_relative_path,
            status=status,
            processed_at="2026-09-07T00:00:00Z",
            pipeline_version=pipeline_version,
        )
    )
    return manifest


def _discover_and_plan(input_root: Path, manifest, output_root: Path):
    discovery = discover_employee_folders(input_root)
    plan = build_incremental_scan_plan(
        discovery=discovery,
        manifest=manifest,
        output_root=output_root,
    )
    return discovery, plan


def test_second_run_without_changes_has_no_process_items(tmp_path: Path):
    input_root = tmp_path / "employees"
    employee = input_root / "NV01"
    output_root = tmp_path / "output"
    employee.mkdir(parents=True)
    (output_root / "NV01").mkdir(parents=True)

    source = employee / "card.jpg"
    source.write_bytes(b"card-content")
    (output_root / "NV01" / "card.pdf").write_bytes(b"%PDF")
    manifest = _make_manifest(input_root, output_root, source)

    discovery, plan = _discover_and_plan(input_root, manifest, output_root)

    assert discovery.files[0].classification == FileClassification.UNCHANGED
    assert discovery.files_to_process == []
    assert plan.new == 0
    assert plan.modified == 0
    assert plan.rebuild == 0
    assert plan.unchanged == 1
    assert plan.files_to_process == 0


def test_new_and_modified_sources_are_selected(tmp_path: Path):
    input_root = tmp_path / "employees"
    employee = input_root / "NV01"
    output_root = tmp_path / "output"
    employee.mkdir(parents=True)
    (output_root / "NV01").mkdir(parents=True)

    existing = employee / "existing.jpg"
    existing.write_bytes(b"old-content")
    (output_root / "NV01" / "existing.pdf").write_bytes(b"%PDF")
    manifest = _make_manifest(input_root, output_root, existing)

    existing.write_bytes(b"new-content")
    new_source = employee / "new.jpg"
    new_source.write_bytes(b"new-file")

    discovery, plan = _discover_and_plan(input_root, manifest, output_root)
    classifications = {file.relative_path: file.classification for file in discovery.files}

    assert classifications == {
        "NV01/existing.jpg": FileClassification.MODIFIED,
        "NV01/new.jpg": FileClassification.NEW,
    }
    assert plan.new == 1
    assert plan.modified == 1
    assert plan.files_to_process == 2


def test_missing_pdf_forces_rebuild_even_when_source_is_unchanged(tmp_path: Path):
    input_root = tmp_path / "employees"
    employee = input_root / "NV01"
    output_root = tmp_path / "output"
    employee.mkdir(parents=True)
    source = employee / "card.jpg"
    source.write_bytes(b"card-content")

    manifest = _make_manifest(input_root, output_root, source)
    discovery, plan = _discover_and_plan(input_root, manifest, output_root)

    assert discovery.files[0].classification == FileClassification.REBUILD
    assert plan.rebuild == 1
    assert plan.files_to_process == 1


def test_mtime_change_with_same_bytes_updates_manifest_without_reprocessing(tmp_path: Path):
    input_root = tmp_path / "employees"
    employee = input_root / "NV01"
    output_root = tmp_path / "output"
    employee.mkdir(parents=True)
    (output_root / "NV01").mkdir(parents=True)

    source = employee / "card.jpg"
    source.write_bytes(b"same-bytes")
    (output_root / "NV01" / "card.pdf").write_bytes(b"%PDF")
    manifest = _make_manifest(input_root, output_root, source)
    old_size, old_mtime_ns = compute_fast_fingerprint(source)
    old_updated_at = manifest.updated_at

    os.utime(source, ns=(old_mtime_ns + 1_000_000, old_mtime_ns + 1_000_000))
    discovery, plan = _discover_and_plan(input_root, manifest, output_root)

    entry = manifest.get_entry("NV01/card.jpg")
    assert entry is not None
    assert discovery.files[0].classification == FileClassification.UNCHANGED
    assert discovery.files[0].sha256 == entry.sha256
    assert entry.size == old_size
    assert entry.mtime_ns != old_mtime_ns
    assert manifest.updated_at != old_updated_at
    assert plan.files_to_process == 0


def test_pipeline_mismatch_forces_rebuild_of_existing_output(tmp_path: Path):
    input_root = tmp_path / "employees"
    employee = input_root / "NV01"
    output_root = tmp_path / "output"
    employee.mkdir(parents=True)
    (output_root / "NV01").mkdir(parents=True)

    source = employee / "card.jpg"
    source.write_bytes(b"card-content")
    (output_root / "NV01" / "card.pdf").write_bytes(b"%PDF")
    manifest = _make_manifest(
        input_root,
        output_root,
        source,
        pipeline_version="0.0.9",
    )

    discovery, plan = _discover_and_plan(input_root, manifest, output_root)

    assert discovery.files[0].classification == FileClassification.REBUILD
    assert plan.files_to_process == 1
    assert plan.rebuild == 1
    assert plan.outdated_pipeline_count == 1


def test_failed_entry_is_retryable_and_uses_rebuild_when_output_is_missing(tmp_path: Path):
    input_root = tmp_path / "employees"
    employee = input_root / "NV01"
    output_root = tmp_path / "output"
    employee.mkdir(parents=True)
    source = employee / "card.jpg"
    source.write_bytes(b"card-content")

    failed_with_output = output_root / "NV01" / "retry.pdf"
    failed_with_output.parent.mkdir(parents=True)
    failed_with_output.write_bytes(b"%PDF")
    manifest = _make_manifest(
        input_root,
        output_root,
        source,
        status=FileProcessingStatus.FAILED,
        output_relative_path="NV01/retry.pdf",
    )

    discovery, plan = _discover_and_plan(input_root, manifest, output_root)
    assert discovery.files[0].classification == FileClassification.MODIFIED
    assert plan.files_to_process == 1

    failed_with_output.unlink()
    discovery, plan = _discover_and_plan(input_root, manifest, output_root)
    assert discovery.files[0].classification == FileClassification.REBUILD
    assert plan.rebuild == 1


def test_deleted_source_keeps_historical_manifest_entry(tmp_path: Path):
    input_root = tmp_path / "employees"
    employee = input_root / "NV01"
    output_root = tmp_path / "output"
    employee.mkdir(parents=True)

    source = employee / "card.jpg"
    source.write_bytes(b"card-content")
    manifest = _make_manifest(input_root, output_root, source)
    source.unlink()

    discovery = discover_employee_folders(input_root)
    classify_discovered_files(discovery, manifest, output_root=output_root)

    assert discovery.files == []
    assert manifest.get_entry("NV01/card.jpg") is not None


def test_cli_plan_reads_manifest_and_emits_incremental_counters(
    tmp_path: Path, monkeypatch, capsys
):
    input_root = tmp_path / "employees"
    employee = input_root / "NV01"
    output_root = tmp_path / "output"
    employee.mkdir(parents=True)
    (output_root / "NV01").mkdir(parents=True)

    source = employee / "card.jpg"
    source.write_bytes(b"card-content")
    (output_root / "NV01" / "card.pdf").write_bytes(b"%PDF")
    state_dir = tmp_path / "state"
    manifest = _make_manifest(input_root, output_root, source)
    persisted_store = ManifestStore(state_dir=state_dir)
    persisted_store.save_manifest(manifest)

    monkeypatch.setattr(cli, "ManifestStore", lambda: ManifestStore(state_dir=state_dir))
    assert cli.main(["plan", "--input", str(input_root), "--output", str(output_root)]) == 0

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    event = deserialize_event(lines[0])
    assert isinstance(event, ScanPlanEvent)
    assert event.new == 0
    assert event.modified == 0
    assert event.rebuild == 0
    assert event.unchanged == 1
    assert event.files_to_process == 0


def test_scan_plan_rejects_inconsistent_process_counter():
    try:
        ScanPlan(
            input_root="/input",
            output_root="/output",
            new=1,
            files_to_process=0,
        )
    except ValueError as exc:
        assert "new + modified + rebuild" in str(exc)
    else:
        raise AssertionError("Expected inconsistent ScanPlan counters to be rejected")


def test_scan_plan_event_derives_missing_process_counter_for_compatibility():
    event = deserialize_event(
        '{"protocolVersion":1,"type":"scan_plan",'
        '"timestamp":"2026-09-07T00:00:00Z","inputRoot":"/input",'
        '"outputRoot":"/output","employees":1,"totalImages":2,'
        '"new":1,"modified":0,"rebuild":1,"unchanged":0,"collisions":[]}'
    )

    assert isinstance(event, ScanPlanEvent)
    assert event.files_to_process == 2
