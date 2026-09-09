"""AS-21 200-image reliability and incremental acceptance tests."""

from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw, PdfParser

from attendance_scanner.batch import run_batch
from attendance_scanner.contracts import BatchPeriod, ExportMode, ScannerErrorCode
from attendance_scanner.discovery import (
    build_group_aware_scan_plan,
    build_incremental_scan_plan,
    discover_employee_folders,
)
from attendance_scanner.events import FileCompletedEvent, FileFailedEvent
from attendance_scanner.state import ManifestStore


def _write_valid_image(path: Path, marker: int) -> None:
    """Create a deterministic dark-desk/light-paper image with no PII."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (160, 120), (32, 32, 32))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 15, 140, 105), fill=(242, 242, 242), outline=(16, 16, 16), width=2)
    draw.line((30, 35, 125, 35), fill=(80, 80, 80), width=2)
    draw.line((30, 50, 110 + marker % 15, 50), fill=(80, 80, 80), width=2)
    image.save(path)


def _write_corrupt_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not-a-valid-image")


def _write_blank_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (160, 120), (128, 128, 128)).save(path)


def _create_200_image_root(root: Path) -> None:
    for employee_index in range(1, 5):
        for image_index in range(1, 51):
            _write_valid_image(
                root / f"NV{employee_index:02d}" / f"card-{image_index:03d}.png",
                image_index,
            )


def _run_and_count(
    input_root: Path,
    output_root: Path,
    store: ManifestStore,
    *,
    expected_processed: int,
) -> tuple[object, int, float, int]:
    discovery = discover_employee_folders(input_root)
    manifest = store.load_manifest(input_root, output_root=output_root)
    build_incremental_scan_plan(discovery, manifest, output_root)
    calls = 0

    from attendance_scanner.pipeline.orchestrator import scan_one as real_scan_one

    def counted_scan(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_scan_one(*args, **kwargs)

    tracemalloc.start()
    started_at = time.perf_counter()
    with patch("attendance_scanner.batch.scan_one", side_effect=counted_scan):
        result = run_batch(
            discovery=discovery,
            manifest=manifest,
            output_root=output_root,
            workers=2,
            manifest_store=store,
        )
    duration_seconds = time.perf_counter() - started_at
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert calls == expected_processed
    return result, calls, duration_seconds, peak_bytes


def test_200_image_incremental_acceptance_scenarios(tmp_path: Path) -> None:
    input_root = tmp_path / "employees"
    output_root = tmp_path / "output"
    state_root = tmp_path / "state"
    _create_200_image_root(input_root)
    store = ManifestStore(state_dir=state_root)

    first_result, _, first_duration, peak_bytes = _run_and_count(
        input_root, output_root, store, expected_processed=200
    )
    assert first_result.summary.failed == 0
    assert first_result.summary.total_images == 200
    assert len(list(output_root.rglob("*.pdf"))) == 200

    grouped_output_root = tmp_path / "grouped-output"
    grouped_store = ManifestStore(state_dir=tmp_path / "grouped-state")
    grouped_discovery = discover_employee_folders(input_root)
    grouped_manifest = grouped_store.load_manifest(input_root, output_root=grouped_output_root)
    grouped_period = BatchPeriod(year=2026, month=9)
    grouped_plan = build_group_aware_scan_plan(
        grouped_discovery,
        grouped_manifest,
        grouped_output_root,
        grouped_period,
        export_mode=ExportMode.GROUPED,
    )
    manual_orders = {
        f"NV{employee_index:02d}:2026-09": [
            f"NV{employee_index:02d}/card-{image_index:03d}.png" for image_index in range(1, 51)
        ]
        for employee_index in range(1, 5)
    }
    grouped_result = run_batch(
        discovery=grouped_discovery,
        manifest=grouped_manifest,
        output_root=grouped_output_root,
        workers=2,
        manifest_store=grouped_store,
        batch_period=grouped_period,
        export_mode=ExportMode.GROUPED,
        group_plan=grouped_plan,
        manual_orders=manual_orders,
    )
    expected_document_groups = len({file.employee_name for file in grouped_discovery.files})
    grouped_artifacts = list(grouped_output_root.rglob("*.pdf"))
    expected_grouped_artifacts = len(grouped_artifacts)
    auto_ordered_groups = sum(
        1 for group in grouped_plan.affected_groups if not group.review_required
    )
    review_required_groups = len(grouped_plan.review_groups)
    incorrect_auto_order_count = sum(
        1
        for group in grouped_plan.affected_groups
        if not group.review_required
        and grouped_manifest.groups[
            f"{group.key.employee_relative_dir}:{group.key.year:04d}-{group.key.month:02d}"
        ].source_relative_paths
        != sorted(group.source_relative_paths)
    )
    assert grouped_result.summary.failed == 0
    assert grouped_plan.affected_group_count == expected_document_groups
    assert expected_grouped_artifacts == expected_document_groups
    assert auto_ordered_groups == 0
    assert review_required_groups == expected_document_groups
    assert incorrect_auto_order_count == 0
    assert all(
        group.key.year == 2026 and group.key.month == 9 for group in grouped_plan.affected_groups
    )

    second_discovery = discover_employee_folders(input_root)
    second_manifest = store.load_manifest(input_root, output_root=output_root)
    second_plan = build_incremental_scan_plan(second_discovery, second_manifest, output_root)
    assert second_plan.files_to_process == 0
    second_result, _, _, _ = _run_and_count(input_root, output_root, store, expected_processed=0)
    assert second_result.summary.skipped == 200

    for image_index in range(51, 61):
        _write_valid_image(input_root / "NV05" / f"card-{image_index:03d}.png", image_index)
    added_result, _, _, _ = _run_and_count(input_root, output_root, store, expected_processed=10)
    assert added_result.summary.failed == 0

    for image_index in range(1, 4):
        _write_valid_image(input_root / "NV01" / f"card-{image_index:03d}.png", 1000 + image_index)
    modified_result, _, _, _ = _run_and_count(input_root, output_root, store, expected_processed=3)
    assert modified_result.summary.failed == 0

    (output_root / "NV01" / "card-001.pdf").unlink()
    (output_root / "NV01" / "card-002.pdf").unlink()
    rebuilt_result, _, _, _ = _run_and_count(input_root, output_root, store, expected_processed=2)
    assert rebuilt_result.summary.failed == 0
    assert len(list(output_root.rglob("*.pdf"))) == 210

    sample_pdf = output_root / "NV01" / "card-001.pdf"
    assert sample_pdf.read_bytes().startswith(b"%PDF-")
    parser = PdfParser.PdfParser(filename=str(sample_pdf))
    assert len(parser.pages) == 1

    print(
        json.dumps(
            {
                "images": 200,
                "firstScanSeconds": round(first_duration, 3),
                "peakTracemallocBytes": peak_bytes,
                "secondScanProcessed": second_result.summary.total_images,
                "addedProcessed": added_result.summary.total_images,
                "modifiedProcessed": modified_result.summary.total_images,
                "rebuiltProcessed": rebuilt_result.summary.total_images,
                "finalPdfCount": len(list(output_root.rglob("*.pdf"))),
                "expectedDocumentGroups": expected_document_groups,
                "expectedGroupedArtifacts": expected_grouped_artifacts,
                "autoOrderedGroups": auto_ordered_groups,
                "reviewRequiredGroups": review_required_groups,
                "incorrectAutoOrderCount": incorrect_auto_order_count,
                "groupingScope": "employee + 2026-09",
                "templateCoverage": (
                    "synthetic document detector fixtures; page identity review remains required"
                ),
            },
            sort_keys=True,
        )
    )


def test_corrupt_and_warning_files_are_isolated_and_resumable(tmp_path: Path) -> None:
    input_root = tmp_path / "employees"
    output_root = tmp_path / "output"
    store = ManifestStore(state_dir=tmp_path / "state")
    _write_valid_image(input_root / "NV01" / "good.png", 1)
    _write_corrupt_image(input_root / "NV02" / "corrupt.jpg")
    _write_blank_image(input_root / "NV03" / "blank.png")

    discovery = discover_employee_folders(input_root)
    manifest = store.load_manifest(input_root, output_root=output_root)
    result = run_batch(
        discovery=discovery,
        manifest=manifest,
        output_root=output_root,
        workers=2,
        manifest_store=store,
    )

    assert result.exit_code == 2
    assert result.summary.failed == 1
    assert result.summary.warning == 1
    assert result.summary.success == 1
    failed_events = [event for event in result.events if isinstance(event, FileFailedEvent)]
    warning_events = [
        event for event in result.events if isinstance(event, FileCompletedEvent) and event.warning
    ]
    assert failed_events[0].error_code == ScannerErrorCode.IMAGE_DECODE_FAILED
    assert failed_events[0].message
    assert len(warning_events) == 1
    assert (output_root / "NV03" / "blank.pdf").is_file()

    (input_root / "NV02" / "corrupt.jpg").unlink()
    _write_valid_image(input_root / "NV02" / "repaired.png", 2)
    repaired_discovery = discover_employee_folders(input_root)
    repaired_manifest = store.load_manifest(input_root, output_root=output_root)
    repaired_result = run_batch(
        discovery=repaired_discovery,
        manifest=repaired_manifest,
        output_root=output_root,
        workers=1,
        manifest_store=store,
    )
    assert repaired_result.exit_code == 0
    assert repaired_result.summary.failed == 0
    assert (output_root / "NV02" / "repaired.pdf").is_file()
