"""AS-26 manifest v2 migration, context, and artifact dependency tests."""

import json
from pathlib import Path

import pytest

from attendance_scanner.contracts import (
    BatchPeriod,
    CompletenessStatus,
    DocumentGroupKey,
    ExportMode,
    FileProcessingStatus,
    PageIdentity,
    PageType,
    ScannerErrorCode,
)
from attendance_scanner.state import (
    CURRENT_SCHEMA_VERSION,
    LEGACY_SCHEMA_VERSION,
    ManifestArtifact,
    ManifestEntry,
    ManifestGroup,
    ManifestStore,
    assign_entry_context,
    compute_root_id,
)


def _legacy_manifest(input_root: Path, output_root: Path) -> dict:
    return {
        "schemaVersion": LEGACY_SCHEMA_VERSION,
        "rootId": compute_root_id(input_root),
        "inputRoot": str(input_root.resolve()),
        "outputRoot": str(output_root.resolve()),
        "createdAt": "2026-09-01T00:00:00Z",
        "updatedAt": "2026-09-01T00:00:00Z",
        "entries": {
            "NV01/random-source.png": {
                "relativePath": "NV01/random-source.png",
                "size": 123,
                "mtimeNs": 456,
                "sha256": "abc123",
                "outputRelativePath": "NV01/random-source.pdf",
                "status": "success",
                "processedAt": "2026-09-01T00:00:00Z",
                "pipelineVersion": "0.1.0",
            }
        },
    }


def test_v1_manifest_migrates_losslessly_to_v2_and_persists_atomically(tmp_path: Path):
    input_root = tmp_path / "employees"
    output_root = tmp_path / "output"
    state_root = tmp_path / "state"
    input_root.mkdir()
    output_root.mkdir()
    store = ManifestStore(state_dir=state_root)
    manifest_path = store.get_manifest_path(input_root)
    state_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(_legacy_manifest(input_root, output_root)), encoding="utf-8"
    )

    migrated = store.load_manifest(input_root, output_root=output_root)
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = migrated.get_entry("NV01/random-source.png")

    assert migrated.schema_version == CURRENT_SCHEMA_VERSION
    assert entry is not None
    assert entry.period is None
    assert entry.page_identity.page_type == PageType.UNKNOWN
    assert entry.artifact_dependencies == ["NV01/random-source.pdf"]
    assert migrated.artifacts["NV01/random-source.pdf"].export_mode == ExportMode.PER_IMAGE
    assert migrated.artifacts["NV01/random-source.pdf"].source_relative_paths == [
        "NV01/random-source.png"
    ]
    assert persisted["schemaVersion"] == CURRENT_SCHEMA_VERSION
    assert not list(state_root.glob("*.corrupt.*"))


def test_new_entry_gets_selected_period_but_modified_entry_keeps_old_period():
    new_entry = ManifestEntry(
        relative_path="NV01/new.png",
        size=10,
        mtime_ns=20,
        status=FileProcessingStatus.PENDING,
        processed_at="2026-09-01T00:00:00Z",
    )
    requested = BatchPeriod(year=2026, month=9)
    assign_entry_context(new_entry, employee_relative_dir="NV01", batch_period=requested)
    assert new_entry.period == requested
    assert new_entry.group_key == DocumentGroupKey(
        employee_relative_dir="NV01", year=2026, month=9
    )

    old_period = BatchPeriod(year=2026, month=8)
    old_entry = ManifestEntry(
        relative_path="NV01/old.png",
        size=10,
        mtime_ns=20,
        period=old_period,
        group_key=DocumentGroupKey(employee_relative_dir="NV01", year=2026, month=8),
        processed_at="2026-08-01T00:00:00Z",
    )
    assign_entry_context(
        old_entry,
        employee_relative_dir="NV01",
        batch_period=requested,
        page_identity=PageIdentity(page_type=PageType.FIRST_HALF, page_order=1),
    )
    assert old_entry.period == old_period
    assert old_entry.group_key.month == 8
    assert old_entry.page_identity.page_order == 1


def test_v2_round_trip_preserves_group_and_ordered_artifact_dependencies(tmp_path: Path):
    input_root = tmp_path / "employees"
    input_root.mkdir()
    store = ManifestStore(state_dir=tmp_path / "state")
    manifest = store.load_manifest(input_root, output_root=tmp_path / "output")
    key = DocumentGroupKey(employee_relative_dir="Nguyễn Văn A", year=2026, month=9)
    group = ManifestGroup(
        key=key,
        source_relative_paths=["Nguyễn Văn A/page-1.png", "Nguyễn Văn A/page-2.png"],
        artifact_relative_paths=["Nguyễn Văn A/2026-09_Nguyễn Văn A.pdf"],
        completeness_status=CompletenessStatus.COMPLETE,
        review_required=False,
    )
    artifact = ManifestArtifact(
        export_mode=ExportMode.GROUPED,
        output_relative_path="Nguyễn Văn A/2026-09_Nguyễn Văn A.pdf",
        source_relative_paths=["Nguyễn Văn A/page-1.png", "Nguyễn Văn A/page-2.png"],
        artifact_version="0.1.0",
        artifact_hash="deadbeef",
    )
    manifest.set_group(group)
    manifest.set_artifact(artifact)
    store.save_manifest(manifest)

    restored = store.load_manifest(input_root, output_root=tmp_path / "output")

    assert restored.groups == {"Nguyễn Văn A:2026-09": group}
    assert restored.artifacts[artifact.output_relative_path] == artifact


def test_migration_failure_keeps_original_v1_file_and_returns_typed_state_error(tmp_path: Path):
    input_root = tmp_path / "employees"
    output_root = tmp_path / "output"
    input_root.mkdir()
    output_root.mkdir()
    store = ManifestStore(state_dir=tmp_path / "state")
    manifest_path = store.get_manifest_path(input_root)
    store.state_dir.mkdir(parents=True, exist_ok=True)
    invalid = _legacy_manifest(input_root, output_root)
    del invalid["entries"]["NV01/random-source.png"]["processedAt"]
    original = json.dumps(invalid)
    manifest_path.write_text(original, encoding="utf-8")

    with pytest.raises(Exception) as exc_info:
        store.load_manifest(input_root, output_root=output_root)

    assert getattr(exc_info.value, "code", None) == ScannerErrorCode.STATE_READ_FAILED
    assert manifest_path.read_text(encoding="utf-8") == original
    assert not list(store.state_dir.glob("*.corrupt.*"))
