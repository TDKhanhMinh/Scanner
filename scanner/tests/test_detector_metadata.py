"""AS-51 detector/model metadata persistence tests."""

from pathlib import Path

from PIL import Image

from attendance_scanner.batch import _aggregate_detection_metadata, _make_failure_entry, run_batch
from attendance_scanner.contracts import BatchPeriod, ExportMode
from attendance_scanner.discovery import (
    build_group_aware_scan_plan,
    discover_employee_folders,
)
from attendance_scanner.state import ManifestArtifact, ManifestEntry, ManifestStore


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (160, 120), color=(128, 128, 128)).save(path)


def test_per_image_scan_persists_detector_metadata_without_changing_identity(tmp_path: Path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _write_image(input_root / "NV01" / "sheet.png")
    store = ManifestStore(state_dir=tmp_path / "state")
    discovery = discover_employee_folders(input_root)
    manifest = store.load_manifest(input_root, output_root=output_root)

    result = run_batch(
        discovery,
        manifest,
        output_root=output_root,
        workers=1,
        manifest_store=store,
        mode="gray",
        pipeline_version="v2-test-pipeline",
    )

    assert result.summary.failed == 0
    persisted = store.load_manifest(input_root, output_root=output_root)
    entry = persisted.entries["NV01/sheet.png"]
    assert entry.pipeline_version == "v2-test-pipeline"
    assert entry.detector_name == "v1_cv"
    assert entry.detector_mode == "gray"
    assert entry.detector_model_version == "opencv-classical"
    assert entry.detection_status in {"detected", "fallback"}
    assert entry.detection_fallback_used is True
    assert entry.relative_path == "NV01/sheet.png"
    assert entry.artifact_dependencies == ["NV01/sheet.pdf"]


def test_grouped_artifact_persists_detector_metadata_and_source_dependencies(tmp_path: Path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "grouped-output"
    _write_image(input_root / "NV01" / "sheet.png")
    store = ManifestStore(state_dir=tmp_path / "grouped-state")
    discovery = discover_employee_folders(input_root)
    manifest = store.load_manifest(input_root, output_root=output_root)
    period = BatchPeriod(year=2026, month=9)
    plan = build_group_aware_scan_plan(
        discovery,
        manifest,
        output_root,
        period,
        export_mode=ExportMode.GROUPED,
    )

    result = run_batch(
        discovery,
        manifest,
        output_root=output_root,
        workers=1,
        manifest_store=store,
        mode="gray",
        pipeline_version="v2-grouped-pipeline",
        batch_period=period,
        export_mode=ExportMode.GROUPED,
        group_plan=plan,
        manual_orders={"NV01:2026-09": ["NV01/sheet.png"]},
    )

    assert result.summary.failed == 0
    persisted = store.load_manifest(input_root, output_root=output_root)
    assert len(persisted.artifacts) == 1
    artifact = next(iter(persisted.artifacts.values()))
    assert artifact.export_mode == ExportMode.GROUPED
    assert artifact.pipeline_version == "v2-grouped-pipeline"
    assert artifact.detector_name == "v1_cv"
    assert artifact.detector_mode == "gray"
    assert artifact.detector_model_version == "opencv-classical"
    assert artifact.source_relative_paths == ["NV01/sheet.png"]


def test_legacy_manifest_entry_and_artifact_round_trip_with_optional_metadata():
    entry_payload = {
        "relativePath": "NV01/sheet.png",
        "size": 10,
        "mtimeNs": 20,
        "status": "success",
        "processedAt": "2026-09-10T00:00:00+00:00",
    }
    artifact_payload = {
        "exportMode": "PER_IMAGE",
        "outputRelativePath": "NV01/sheet.pdf",
    }

    from attendance_scanner.state import ManifestEntry

    entry = ManifestEntry.model_validate(entry_payload)
    artifact = ManifestArtifact.model_validate(artifact_payload)
    assert entry.detector_name is None
    assert artifact.detector_model_version is None
    assert entry.model_dump(by_alias=True)["relativePath"] == "NV01/sheet.png"
    assert artifact.model_dump(by_alias=True)["outputRelativePath"] == "NV01/sheet.pdf"


def test_failure_attempt_uses_current_provenance_and_unknown_fallback_state(tmp_path: Path):
    store = ManifestStore(state_dir=tmp_path / "state")
    manifest = store.load_manifest(tmp_path / "input")
    manifest.set_entry(
        ManifestEntry(
            relative_path="NV01/sheet.png",
            size=1,
            mtime_ns=1,
            status="success",
            processed_at="2026-09-10T00:00:00+00:00",
            pipeline_version="old-pipeline",
            detector_name="old-detector",
            detector_model_version="old-model",
            detection_status="detected",
            detection_fallback_used=False,
        )
    )

    failure = _make_failure_entry(
        "NV01/sheet.png",
        2,
        2,
        "new-hash",
        "NV01/sheet.pdf",
        manifest,
        "new-pipeline",
        detector_name="hybrid",
        detector_mode="hybrid",
        detector_model_version="new-model",
    )

    assert failure.pipeline_version == "new-pipeline"
    assert failure.detector_name == "hybrid"
    assert failure.detector_model_version == "new-model"
    assert failure.detection_status == "failed"
    assert failure.detection_fallback_used is None
    assert failure.output_relative_paths == ["NV01/sheet.pdf"]


def test_grouped_metadata_aggregates_mixed_page_provenance():
    entries = [
        ManifestEntry(
            relative_path="a.png",
            size=1,
            mtime_ns=1,
            status="success",
            processed_at="2026-09-10T00:00:00+00:00",
            pipeline_version="p1",
            detector_name="hybrid",
            detector_mode="hybrid",
            detector_model_version="m1",
            detection_status="detected",
            detection_fallback_used=False,
        ),
        ManifestEntry(
            relative_path="b.png",
            size=1,
            mtime_ns=1,
            status="warning",
            processed_at="2026-09-10T00:00:00+00:00",
            pipeline_version="p2",
            detector_name="v1_cv",
            detector_mode="gray",
            detector_model_version="m2",
            detection_status="fallback",
            detection_fallback_used=True,
        ),
    ]

    aggregate = _aggregate_detection_metadata(entries)

    assert aggregate["pipeline_version"] is None
    assert aggregate["detector_name"] is None
    assert aggregate["detector_model_version"] is None
    assert aggregate["detection_status"] == "fallback"
    assert aggregate["detection_fallback_used"] is True
    assert aggregate["detection_quality_summary"]["detectorVariantCount"] == 2
