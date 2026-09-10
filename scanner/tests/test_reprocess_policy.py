"""AS-52 controlled detector/model reprocess policy tests."""

import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from attendance_scanner import cli
from attendance_scanner.contracts import FileClassification, FileProcessingStatus
from attendance_scanner.discovery import (
    ReprocessPolicy,
    build_incremental_scan_plan,
    discover_employee_folders,
    pipeline_version_for_mode,
)
from attendance_scanner.fingerprint import compute_fast_fingerprint, compute_sha256
from attendance_scanner.state import ManifestEntry, ManifestStore


def _write_image(path: Path, color: int = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (160, 120), color=(color, color, color)).save(path)


def _manifest_with_entries(root: Path, output: Path, store: ManifestStore) -> object:
    manifest = store.load_manifest(root, output_root=output)
    for employee, version, model, mode in (
        ("NV01", "old-pipeline", "old-model", "gray"),
        ("NV02", pipeline_version_for_mode("0.3.0", "gray"), "opencv-classical", "gray"),
    ):
        source = root / employee / "sheet.png"
        size, mtime_ns = compute_fast_fingerprint(source)
        manifest.set_entry(
            ManifestEntry(
                relative_path=f"{employee}/sheet.png",
                size=size,
                mtime_ns=mtime_ns,
                sha256=compute_sha256(source),
                output_relative_path=f"{employee}/sheet.pdf",
                output_relative_paths=[f"{employee}/sheet.pdf"],
                status=FileProcessingStatus.SUCCESS,
                processed_at=datetime.now(timezone.utc).isoformat(),
                pipeline_version=version,
                detector_name="v1_cv",
                detector_mode=mode,
                detector_model_version=model,
                detection_status="detected",
                detection_fallback_used=False,
            )
        )
        (output / employee / "sheet.pdf").parent.mkdir(parents=True, exist_ok=True)
        (output / employee / "sheet.pdf").write_bytes(b"%PDF-test")
    return manifest


def _policy(*, opt_in: bool) -> ReprocessPolicy:
    return ReprocessPolicy(
        pipeline_version="0.3.0",
        detector_name="v1_cv",
        detector_mode="gray",
        detector_model_version="opencv-classical",
        opt_in=opt_in,
    )


def test_same_version_is_unchanged_and_version_mismatch_is_reported_separately(tmp_path: Path):
    root = tmp_path / "input"
    output = tmp_path / "output"
    _write_image(root / "NV01" / "sheet.png")
    _write_image(root / "NV02" / "sheet.png")
    store = ManifestStore(state_dir=tmp_path / "state")
    manifest = _manifest_with_entries(root, output, store)
    discovery = discover_employee_folders(root)

    plan = build_incremental_scan_plan(
        discovery,
        manifest,
        output_root=output,
        reprocess_policy=ReprocessPolicy(
            pipeline_version="0.3.0",
            detector_name="v1_cv",
            detector_mode="gray",
            detector_model_version="opencv-classical",
            opt_in=False,
        ),
    )

    assert discovery.files[0].classification == FileClassification.NEEDS_REPROCESS
    assert discovery.files[1].classification == FileClassification.UNCHANGED
    assert plan.needs_reprocess == 1
    assert plan.files_to_process == 0
    assert plan.reprocess_reasons == {
        "PIPELINE_VERSION_CHANGED": 1,
        "MODEL_VERSION_CHANGED": 1,
    }


def test_opt_in_rebuilds_only_version_affected_file_and_preserves_unrelated_group(tmp_path: Path):
    root = tmp_path / "input"
    output = tmp_path / "output"
    _write_image(root / "NV01" / "sheet.png")
    _write_image(root / "NV02" / "sheet.png")
    store = ManifestStore(state_dir=tmp_path / "state")
    manifest = _manifest_with_entries(root, output, store)
    discovery = discover_employee_folders(root)

    plan = build_incremental_scan_plan(
        discovery,
        manifest,
        output_root=output,
        reprocess_policy=_policy(opt_in=True),
    )

    assert discovery.files[0].classification == FileClassification.REBUILD
    assert discovery.files[1].classification == FileClassification.UNCHANGED
    assert plan.files_to_process == 1
    assert plan.needs_reprocess == 0


def test_source_modified_and_missing_output_keep_existing_rebuild_rules(tmp_path: Path):
    root = tmp_path / "input"
    output = tmp_path / "output"
    _write_image(root / "NV01" / "sheet.png")
    _write_image(root / "NV02" / "sheet.png")
    store = ManifestStore(state_dir=tmp_path / "state")
    manifest = _manifest_with_entries(root, output, store)
    discovery = discover_employee_folders(root)
    source_bytes = (root / "NV01" / "sheet.png").read_bytes()
    (root / "NV01" / "sheet.png").write_bytes(source_bytes + b"changed")
    (output / "NV02" / "sheet.pdf").unlink()
    discovery = discover_employee_folders(root)

    build_incremental_scan_plan(
        discovery,
        manifest,
        output_root=output,
        reprocess_policy=_policy(opt_in=False),
    )

    assert discovery.files[0].classification == FileClassification.MODIFIED
    assert discovery.files[1].classification == FileClassification.REBUILD
    assert discovery.needs_reprocess_count == 0


def test_cli_plan_reports_needs_reprocess_and_reprocess_opt_in(monkeypatch, tmp_path: Path, capsys):
    root = tmp_path / "input"
    output = tmp_path / "output"
    _write_image(root / "NV01" / "sheet.png")
    store = ManifestStore(state_dir=tmp_path / "state")
    manifest = store.load_manifest(root, output_root=output)
    source = root / "NV01" / "sheet.png"
    size, mtime_ns = compute_fast_fingerprint(source)
    manifest.set_entry(
        ManifestEntry(
            relative_path="NV01/sheet.png",
            size=size,
            mtime_ns=mtime_ns,
            sha256=compute_sha256(source),
            output_relative_path="NV01/sheet.pdf",
            output_relative_paths=["NV01/sheet.pdf"],
            status="success",
            processed_at=datetime.now(timezone.utc).isoformat(),
            pipeline_version="old-pipeline",
            detector_name="v1_cv",
            detector_mode="gray",
            detector_model_version="old-model",
            detection_status="detected",
            detection_fallback_used=False,
        )
    )
    (output / "NV01").mkdir(parents=True)
    (output / "NV01" / "sheet.pdf").write_bytes(b"%PDF-test")
    store.save_manifest(manifest)
    monkeypatch.setattr(cli, "ManifestStore", lambda: store)

    assert cli.main(["plan", "--input", str(root), "--output", str(output), "--mode", "gray"]) == 0
    event = json.loads(capsys.readouterr().out)
    assert event["needsReprocess"] == 1
    assert event["filesToProcess"] == 0

    assert (
        cli.main(
            [
                "plan",
                "--input",
                str(root),
                "--output",
                str(output),
                "--mode",
                "gray",
                "--reprocess",
            ]
        )
        == 0
    )
    opt_in_event = json.loads(capsys.readouterr().out)
    assert opt_in_event["needsReprocess"] == 0
    assert opt_in_event["filesToProcess"] == 1
