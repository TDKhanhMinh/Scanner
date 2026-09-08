"""AS-27 affected document-group planning tests."""

from pathlib import Path

from PIL import Image

from attendance_scanner.contracts import (
    BatchPeriod,
    CompletenessStatus,
    DocumentGroupKey,
    ExportMode,
    FileProcessingStatus,
)
from attendance_scanner.discovery import build_group_aware_scan_plan, discover_employee_folders
from attendance_scanner.state import (
    ManifestArtifact,
    ManifestEntry,
    ManifestGroup,
    ManifestStore,
)


def _write_image(path: Path, value: int = 160) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 30), color=(value, value, value)).save(path)


def _seed_grouped_context(tmp_path: Path):
    input_root = tmp_path / "employees"
    output_root = tmp_path / "output"
    state_root = tmp_path / "state"
    files = {
        "A/first.png": 160,
        "A/second.png": 170,
        "B/only.png": 180,
        "A_old/only.png": 190,
    }
    for relative_path, value in files.items():
        _write_image(input_root / relative_path, value)

    store = ManifestStore(state_dir=state_root)
    manifest = store.load_manifest(input_root, output_root=output_root)
    period = BatchPeriod(year=2026, month=9)
    groups = {
        "A:2026-09": ["A/first.png", "A/second.png"],
        "B:2026-09": ["B/only.png"],
        "A_old:2026-08": ["A_old/only.png"],
    }
    for storage_key, source_paths in groups.items():
        employee_name, month = storage_key.split(":")
        year, month_number = month.split("-")
        key = DocumentGroupKey(
            employee_relative_dir=employee_name,
            year=int(year),
            month=int(month_number),
        )
        output_relative_path = f"{employee_name}/{year}-{month_number}_{employee_name}.pdf"
        (output_root / output_relative_path).parent.mkdir(parents=True, exist_ok=True)
        (output_root / output_relative_path).write_bytes(b"%PDF-1.7\n%%EOF")
        manifest.set_group(
            ManifestGroup(
                key=key,
                source_relative_paths=source_paths,
                artifact_relative_paths=[output_relative_path],
                completeness_status=CompletenessStatus.COMPLETE,
                review_required=False,
            )
        )
        manifest.set_artifact(
            ManifestArtifact(
                export_mode=ExportMode.GROUPED,
                output_relative_path=output_relative_path,
                source_relative_paths=source_paths,
                artifact_version="0.1.0",
            )
        )
        for source_path in source_paths:
            manifest.set_entry(
                ManifestEntry(
                    relative_path=source_path,
                    size=(input_root / source_path).stat().st_size,
                    mtime_ns=(input_root / source_path).stat().st_mtime_ns,
                    output_relative_path=output_relative_path,
                    output_relative_paths=[output_relative_path],
                    status=FileProcessingStatus.SUCCESS,
                    processed_at="2026-09-01T00:00:00Z",
                    period=period if employee_name != "A_old" else BatchPeriod(year=2026, month=8),
                    group_key=key,
                )
            )
    store.save_manifest(manifest)
    return input_root, output_root, manifest, store, period


def test_modified_page_affects_only_matching_employee_period_group(tmp_path: Path):
    input_root, output_root, manifest, _, period = _seed_grouped_context(tmp_path)
    _write_image(input_root / "A/second.png", 240)

    discovery = discover_employee_folders(input_root)
    plan = build_group_aware_scan_plan(
        discovery, manifest, output_root, period, export_mode=ExportMode.GROUPED
    )

    assert plan.affected_group_count == 1
    affected = plan.affected_groups[0]
    assert affected.key == DocumentGroupKey(employee_relative_dir="A", year=2026, month=9)
    assert affected.process_relative_paths == ["A/first.png", "A/second.png"]
    assert affected.reasons == ["source_modified"]
    assert plan.process_relative_paths == ["A/first.png", "A/second.png"]


def test_per_image_mode_keeps_affected_process_set_at_file_level(tmp_path: Path):
    input_root, output_root, manifest, _, period = _seed_grouped_context(tmp_path)
    _write_image(input_root / "A/second.png", 240)

    plan = build_group_aware_scan_plan(
        discover_employee_folders(input_root),
        manifest,
        output_root,
        period,
        export_mode=ExportMode.PER_IMAGE,
    )

    assert plan.process_relative_paths == ["A/second.png"]
    assert plan.affected_groups[0].process_relative_paths == ["A/second.png"]


def test_missing_grouped_output_rebuilds_only_that_group(tmp_path: Path):
    input_root, output_root, manifest, _, period = _seed_grouped_context(tmp_path)
    (output_root / "A/2026-09_A.pdf").unlink()

    plan = build_group_aware_scan_plan(
        discover_employee_folders(input_root),
        manifest,
        output_root,
        period,
        export_mode=ExportMode.GROUPED,
    )

    assert plan.affected_group_count == 1
    assert plan.affected_groups[0].key.employee_relative_dir == "A"
    assert "missing_grouped_output" in plan.affected_groups[0].reasons
    assert plan.affected_groups[0].reasons.count("missing_grouped_output") == 1
    assert plan.process_relative_paths == ["A/first.png", "A/second.png"]


def test_removed_source_marks_group_incomplete_without_deleting_artifact(tmp_path: Path):
    input_root, output_root, manifest, _, period = _seed_grouped_context(tmp_path)
    (input_root / "A/second.png").unlink()

    plan = build_group_aware_scan_plan(
        discover_employee_folders(input_root),
        manifest,
        output_root,
        period,
        export_mode=ExportMode.GROUPED,
    )

    affected = plan.affected_groups[0]
    assert affected.key.employee_relative_dir == "A"
    assert affected.completeness_status == CompletenessStatus.INCOMPLETE
    assert affected.review_required is True
    assert "source_removed" in affected.reasons
    assert (output_root / "A/2026-09_A.pdf").is_file()
