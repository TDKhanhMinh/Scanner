"""Flat-folder discovery, manifest and export workflow tests."""

import re
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

import attendance_scanner.flat_batch as flat_batch_module
from attendance_scanner.cli import main
from attendance_scanner.contracts import (
    DetectionFailureReason,
    FlatExportMode,
    ScannerError,
    ScannerErrorCode,
    StateError,
)
from attendance_scanner.discovery import discover_flat_folder
from attendance_scanner.flat_batch import run_flat_scan
from attendance_scanner.flat_state import FLAT_MANIFEST_FILENAME, FlatManifestStore


@pytest.fixture(autouse=True)
def isolate_flat_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Keep internal flat manifests out of the developer's real AppData."""
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))


def _write_image(path: Path, *, color: tuple[int, int, int] = (245, 245, 245)) -> Path:
    image = Image.new("RGB", (240, 180), color=(40, 40, 40))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 220, 160), fill=color, outline=(0, 0, 0), width=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _pdf_page_count(path: Path) -> int:
    payload = path.read_bytes()
    assert payload.startswith(b"%PDF")
    return len(re.findall(rb"/Type /Page\b", payload))


def test_flat_discovery_is_non_recursive_and_hashes_collisions(tmp_path: Path):
    root = tmp_path / "input"
    output = root / "out"
    _write_image(root / "img.jpg")
    _write_image(root / "img.png", color=(230, 240, 250))
    _write_image(root / "page2.webp")
    _write_image(output / "ignored.jpg")
    (root / "notes.txt").write_text("ignored", encoding="utf-8")

    result = discover_flat_folder(root, output)

    assert result.image_count == 3
    assert result.unsupported_count == 1
    assert result.collisions == ["img"]
    assert [file.file_name for file in result.files] == ["img.jpg", "img.png", "page2.webp"]
    assert all(len(file.content_fingerprint) == 64 for file in result.files)
    assert {file.target_relative_pdf for file in result.files} >= {"img__jpg.pdf", "img__png.pdf"}
    assert all(file.file_name != "ignored.jpg" for file in result.files)


def test_flat_per_image_scan_is_incremental_and_manifest_isolated(tmp_path: Path):
    root = tmp_path / "input"
    output = tmp_path / "output"
    first = _write_image(root / "page1.jpg")
    _write_image(root / "page2.jpg", color=(230, 240, 250))

    first_run = run_flat_scan(
        root,
        output,
        export_mode=FlatExportMode.PER_IMAGE,
        detector_mode="classic",
        workers=1,
    )
    second_run = run_flat_scan(
        root,
        output,
        export_mode=FlatExportMode.PER_IMAGE,
        detector_mode="classic",
        workers=1,
    )

    assert first_run.exit_code == 0
    assert first_run.total_processed == 2
    assert second_run.total_processed == 0
    assert second_run.skipped == 2
    assert (output / "page1.pdf").is_file()
    assert (output / "page2.pdf").is_file()
    manifest = FlatManifestStore(output).load(root)
    state_path = FlatManifestStore(output).get_manifest_path(root)
    assert state_path.is_file()
    assert not (output / FLAT_MANIFEST_FILENAME).exists()
    assert set(manifest.entries) == {"page1.jpg", "page2.jpg"}
    assert all(entry.output_fingerprint for entry in manifest.entries.values())

    _write_image(first, color=(210, 230, 240))
    changed_run = run_flat_scan(
        root,
        output,
        export_mode=FlatExportMode.PER_IMAGE,
        detector_mode="classic",
        workers=1,
    )
    assert changed_run.total_processed == 1
    assert changed_run.skipped == 1


def test_flat_manifest_migrates_legacy_output_state_to_appdata(tmp_path: Path):
    root = tmp_path / "input"
    output = tmp_path / "output"
    store = FlatManifestStore(output)
    legacy = store.empty_manifest(root)
    output.mkdir()
    (output / FLAT_MANIFEST_FILENAME).write_text(
        legacy.model_dump_json(by_alias=True),
        encoding="utf-8",
    )

    loaded = store.load(root)

    assert loaded.input_root == str(root.resolve())
    assert store.get_manifest_path(root).is_file()
    assert not (output / FLAT_MANIFEST_FILENAME).exists()


def test_flat_manifest_save_maps_state_root_conflict_to_typed_error(tmp_path: Path):
    state_root = tmp_path / "state-root"
    state_root.write_text("conflicting file", encoding="utf-8")
    store = FlatManifestStore(tmp_path / "output", state_root=state_root)

    with pytest.raises(StateError) as error:
        store.save(store.empty_manifest(tmp_path / "input"))

    assert error.value.code == ScannerErrorCode.STATE_WRITE_FAILED


def test_flat_manifest_persists_detector_provenance_and_reprocesses_model_change(
    tmp_path: Path,
):
    root = tmp_path / "input"
    output = tmp_path / "output"
    _write_image(root / "page.jpg")

    first = run_flat_scan(root, output, detector_mode="classic", workers=1)
    assert first.exit_code == 0
    manifest_store = FlatManifestStore(output)
    manifest = manifest_store.load(root)
    entry = manifest.entries["page.jpg"]
    assert entry.detector_name == "v1_cv"
    assert entry.detector_model_version == "opencv-classical"
    assert entry.detection_status in {"detected", "fallback"}
    assert isinstance(entry.detection_quality_summary, dict)

    entry.detector_model_version = "old-model"
    manifest_store.save(manifest)
    rebuilt = run_flat_scan(root, output, detector_mode="classic", workers=1)
    assert rebuilt.total_processed == 1
    assert (
        manifest_store.load(root).entries["page.jpg"].detector_model_version == "opencv-classical"
    )


def test_flat_manifest_preserves_failed_detection_status_without_fallback(
    tmp_path: Path,
    monkeypatch,
):
    root = tmp_path / "input"
    output = tmp_path / "output"
    _write_image(root / "page.jpg")
    original_scan_one = flat_batch_module.scan_one

    def ambiguous_scan(*args, **kwargs):
        result = original_scan_one(*args, **kwargs)
        result.document_detected = False
        result.detection_fallback_used = False
        result.detection_reason_codes = [DetectionFailureReason.HYBRID_AMBIGUOUS]
        return result

    monkeypatch.setattr(flat_batch_module, "scan_one", ambiguous_scan)
    run_flat_scan(root, output, detector_mode="classic", workers=1)

    entry = FlatManifestStore(output).load(root).entries["page.jpg"]
    assert entry.detection_status == "failed"
    assert entry.detection_fallback_used is False
    assert entry.detection_reason_codes == [DetectionFailureReason.HYBRID_AMBIGUOUS]


def test_flat_per_image_scan_failure_emits_terminal_file_event(
    tmp_path: Path,
    monkeypatch,
):
    root = tmp_path / "input"
    output = tmp_path / "output"
    _write_image(root / "page.jpg")

    def fail_scan(*args, **kwargs):
        raise ScannerError(code=ScannerErrorCode.IMAGE_DECODE_FAILED, message="injected failure")

    monkeypatch.setattr(flat_batch_module, "scan_one", fail_scan)
    events = []
    result = run_flat_scan(root, output, detector_mode="classic", workers=1, emit=events.append)

    file_events = [event for event in events if event.type == "flat_file_completed"]
    assert result.exit_code == 2
    assert result.failed == 1
    assert len(file_events) == 1
    assert file_events[0].status == "failed"
    assert file_events[0].error_code == ScannerErrorCode.IMAGE_DECODE_FAILED


def test_flat_merged_export_failure_keeps_previous_artifact_and_marks_all_files_failed(
    tmp_path: Path,
    monkeypatch,
):
    root = tmp_path / "input"
    output = tmp_path / "output"
    first = _write_image(root / "page1.jpg")
    _write_image(root / "page2.jpg", color=(230, 240, 250))
    run_flat_scan(
        root,
        output,
        export_mode=FlatExportMode.MERGED,
        detector_mode="classic",
        workers=1,
    )
    merged = output / "input_merged.pdf"
    previous_bytes = merged.read_bytes()
    previous_manifest = FlatManifestStore(output).load(root)

    _write_image(first, color=(210, 230, 240))

    def fail_export(*args, **kwargs):
        raise ScannerError(code=ScannerErrorCode.PDF_WRITE_FAILED, message="injected failure")

    monkeypatch.setattr(flat_batch_module, "export_pdf_pages", fail_export)
    events = []
    result = run_flat_scan(
        root,
        output,
        export_mode=FlatExportMode.MERGED,
        detector_mode="classic",
        workers=1,
        emit=events.append,
    )

    completed = events[-1]
    file_events = [event for event in events if event.type == "flat_file_completed"]
    assert result.exit_code == 2
    assert result.success == 0
    assert result.failed == 2
    assert completed.artifact_status == "not_committed"
    assert completed.artifact_error_code == ScannerErrorCode.PDF_WRITE_FAILED
    assert len(file_events) == 2
    assert all(event.status == "failed" for event in file_events)
    assert merged.read_bytes() == previous_bytes
    assert FlatManifestStore(output).load(root).model_dump() == previous_manifest.model_dump()


def test_flat_merged_scan_emits_source_progress_before_terminal_results(tmp_path: Path):
    root = tmp_path / "input"
    output = tmp_path / "output"
    _write_image(root / "page1.jpg")
    _write_image(root / "page2.jpg", color=(230, 240, 250))
    events = []

    run_flat_scan(
        root,
        output,
        export_mode=FlatExportMode.MERGED,
        detector_mode="classic",
        workers=1,
        emit=events.append,
    )

    progress_events = [event for event in events if event.type == "flat_scan_progress"]
    completed_index = next(
        index for index, event in enumerate(events) if event.type == "flat_scan_completed"
    )
    assert [event.completed_sources for event in progress_events] == [1, 2]
    assert all(events.index(event) < completed_index for event in progress_events)


def test_flat_empty_or_unsupported_folder_creates_output_and_explains_state(tmp_path: Path):
    root = tmp_path / "input"
    output = tmp_path / "output"
    root.mkdir()
    (root / "notes.txt").write_text("ignored", encoding="utf-8")
    events = []

    result = run_flat_scan(root, output, emit=events.append)

    completed = events[-1]
    plan = events[0]
    assert result.exit_code == 0
    assert output.is_dir()
    assert plan.unsupported_count == 1
    assert completed.unsupported_count == 1
    assert completed.artifact_status == "not_required"
    assert completed.artifact_message


def test_flat_manifest_rebuilds_when_orientation_changes(tmp_path: Path):
    root = tmp_path / "input"
    output = tmp_path / "output"
    _write_image(root / "page.jpg")

    run_flat_scan(root, output, detector_mode="classic", orientation="auto", workers=1)
    assert (
        FlatManifestStore(output).load(root).entries["page.jpg"].preferred_orientation
        == "landscape"
    )
    changed = run_flat_scan(
        root,
        output,
        detector_mode="classic",
        orientation="portrait",
        workers=1,
    )

    assert changed.total_processed == 1
    manifest = FlatManifestStore(output).load(root)
    assert manifest.entries["page.jpg"].preferred_orientation == "portrait"


def test_flat_manifest_switches_export_modes_without_reusing_output_paths(tmp_path: Path):
    root = tmp_path / "input"
    output = tmp_path / "output"
    _write_image(root / "page1.jpg")
    _write_image(root / "page2.jpg", color=(230, 240, 250))

    run_flat_scan(
        root,
        output,
        export_mode=FlatExportMode.MERGED,
        detector_mode="classic",
        workers=1,
    )
    merged = output / "input_merged.pdf"
    assert merged.is_file()

    switched = run_flat_scan(
        root,
        output,
        export_mode=FlatExportMode.PER_IMAGE,
        detector_mode="classic",
        workers=1,
    )

    assert switched.total_processed == 2
    assert (output / "page1.pdf").is_file()
    assert (output / "page2.pdf").is_file()
    assert not merged.exists()


def test_flat_merged_scan_rebuilds_after_source_deletion(tmp_path: Path):
    root = tmp_path / "input"
    output = tmp_path / "output"
    _write_image(root / "page1.jpg")
    second = _write_image(root / "page2.jpg", color=(230, 240, 250))

    first_run = run_flat_scan(
        root,
        output,
        export_mode=FlatExportMode.MERGED,
        detector_mode="classic",
        workers=1,
    )
    merged = output / "input_merged.pdf"
    assert first_run.exit_code == 0
    assert _pdf_page_count(merged) == 2

    second.unlink()
    rebuilt = run_flat_scan(
        root,
        output,
        export_mode=FlatExportMode.MERGED,
        detector_mode="classic",
        workers=1,
    )

    assert rebuilt.total_processed == 1
    assert _pdf_page_count(merged) == 1
    assert set(FlatManifestStore(output).load(root).entries) == {"page1.jpg"}


def test_scan_flat_cli_streams_plan_file_and_summary_events(tmp_path: Path, capsys):
    root = tmp_path / "input"
    output = tmp_path / "output"
    _write_image(root / "page.jpg")

    exit_code = main(
        [
            "scan-flat",
            "--input",
            str(root),
            "--output",
            str(output),
            "--export-mode",
            "per-image",
            "--detector-mode",
            "classic",
            "--workers",
            "1",
        ]
    )

    events = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert exit_code == 0
    assert events[0].find('"type":"flat_scan_plan"') >= 0
    assert events[-1].find('"type":"flat_scan_completed"') >= 0
