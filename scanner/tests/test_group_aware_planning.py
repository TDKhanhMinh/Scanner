"""AS-27 affected document-group planning tests."""

from pathlib import Path

from PIL import Image

from attendance_scanner.contracts import (
    BatchPeriod,
    CompletenessStatus,
    DocumentGroupKey,
    ExportMode,
    FileProcessingStatus,
    PageIdentity,
    PageType,
)
from attendance_scanner.discovery import (
    DEFAULT_PIPELINE_VERSION,
    build_group_aware_scan_plan,
    discover_employee_folders,
)
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
        "B/first.png": 180,
        "B/second.png": 181,
        "A_old/first.png": 190,
        "A_old/second.png": 191,
    }
    for relative_path, value in files.items():
        _write_image(input_root / relative_path, value)

    store = ManifestStore(state_dir=state_root)
    manifest = store.load_manifest(input_root, output_root=output_root)
    period = BatchPeriod(year=2026, month=9)
    groups = {
        "A:2026-09": ["A/first.png", "A/second.png"],
        "B:2026-09": ["B/first.png", "B/second.png"],
        "A_old:2026-08": ["A_old/first.png", "A_old/second.png"],
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
                artifact_version=DEFAULT_PIPELINE_VERSION,
            )
        )
        for source_path in source_paths:
            page_identity = PageIdentity()
            if source_path.endswith("first.png"):
                page_identity = PageIdentity(
                    page_type=PageType.FIRST_HALF,
                    page_order=1,
                    confidence=0.95,
                )
            elif source_path.endswith("second.png"):
                page_identity = PageIdentity(
                    page_type=PageType.SECOND_HALF,
                    page_order=2,
                    confidence=0.95,
                )
            else:
                page_identity = PageIdentity(
                    page_type=PageType.FIRST_HALF,
                    page_order=1,
                    confidence=0.95,
                )
            manifest.set_entry(
                ManifestEntry(
                    relative_path=source_path,
                    size=(input_root / source_path).stat().st_size,
                    mtime_ns=(input_root / source_path).stat().st_mtime_ns,
                    output_relative_path=output_relative_path,
                    output_relative_paths=[output_relative_path],
                    status=FileProcessingStatus.SUCCESS,
                    processed_at="2026-09-01T00:00:00Z",
                    pipeline_version=DEFAULT_PIPELINE_VERSION,
                    period=period if employee_name != "A_old" else BatchPeriod(year=2026, month=8),
                    group_key=key,
                    page_identity=page_identity,
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


def test_persisted_manual_order_suppresses_review_until_source_changes(tmp_path: Path):
    input_root, output_root, manifest, store, period = _seed_grouped_context(tmp_path)
    key = DocumentGroupKey(employee_relative_dir="A", year=2026, month=9)
    entries = [manifest.get_entry(path) for path in ["A/first.png", "A/second.png"]]
    assert all(entry is not None for entry in entries)

    from attendance_scanner.state import set_manual_group_order

    set_manual_group_order(
        manifest,
        group_key=key,
        ordered_source_paths=["A/second.png", "A/first.png"],
        source_entries=[entry for entry in entries if entry is not None],
    )
    store.save_manifest(manifest)

    plan = build_group_aware_scan_plan(
        discover_employee_folders(input_root),
        manifest,
        output_root,
        period,
        export_mode=ExportMode.GROUPED,
    )
    assert plan.affected_group_count == 0
    assert plan.review_groups == []

    _write_image(input_root / "A/first.png", 241)
    changed_plan = build_group_aware_scan_plan(
        discover_employee_folders(input_root),
        manifest,
        output_root,
        period,
        export_mode=ExportMode.GROUPED,
    )
    assert changed_plan.affected_group_count == 1
    assert "manual_order_invalidated" in changed_plan.affected_groups[0].reasons
    assert changed_plan.review_groups[0].review_required is True


def test_entirely_removed_employee_group_is_pruned_and_not_in_review(tmp_path: Path):
    import shutil

    input_root, output_root, manifest, store, period = _seed_grouped_context(tmp_path)
    # Remove employee B completely from input and output
    shutil.rmtree(input_root / "B")
    (output_root / "B/2026-09_B.pdf").unlink()

    plan = build_group_aware_scan_plan(
        discover_employee_folders(input_root),
        manifest,
        output_root,
        period,
        export_mode=ExportMode.GROUPED,
    )

    # Employee B should not require review or appear in review_groups
    assert not any(group.key.employee_relative_dir == "B" for group in plan.review_groups)
    assert not any(group.key.employee_relative_dir == "B" for group in plan.affected_groups)
    assert "B:2026-09" not in manifest.groups
    assert not any("B/" in path for path in manifest.entries)


def test_entirely_removed_sources_with_persisted_artifact_does_not_require_review(tmp_path: Path):
    import shutil

    input_root, output_root, manifest, store, period = _seed_grouped_context(tmp_path)
    # Remove employee B input images, but keep output artifact
    shutil.rmtree(input_root / "B")

    plan = build_group_aware_scan_plan(
        discover_employee_folders(input_root),
        manifest,
        output_root,
        period,
        export_mode=ExportMode.GROUPED,
    )

    # Should not appear in review_groups since there are no images on disk
    assert not any(group.key.employee_relative_dir == "B" for group in plan.review_groups)
    b_group = next(g for g in plan.affected_groups if g.key.employee_relative_dir == "B")
    assert b_group.completeness_status == CompletenessStatus.INCOMPLETE
    assert b_group.review_required is False
