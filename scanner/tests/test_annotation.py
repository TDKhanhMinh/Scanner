"""AS-34 timesheet annotation workflow tests."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import attendance_scanner.annotation as annotation_module
from attendance_scanner.annotation import (
    AnnotationSession,
    discover_annotation_session,
    finalize_annotation_session,
    render_annotation_overlay,
    resolve_annotation_image_path,
    summarize_annotation_session,
    update_annotation,
)
from attendance_scanner.benchmark import BenchmarkSource
from attendance_scanner.pipeline.load import load_image


def _source() -> BenchmarkSource:
    return BenchmarkSource(
        name="Synthetic anonymized timesheet fixture",
        license="Test-only synthetic data",
        attribution="Attendance Scanner tests",
    )


def _create_images(root: Path, count: int = 3) -> None:
    root.mkdir(parents=True)
    for index in range(count):
        Image.new("RGB", (120, 100), color=(220 + index, 220, 220)).save(
            root / f"sheet-{index + 1:03d}.png"
        )


def _corners(record: object) -> list[tuple[float, float]]:
    return [(10.0, 10.0), (110.0, 10.0), (110.0, 90.0), (10.0, 90.0)]


def _annotate_all(session: AnnotationSession) -> None:
    for index, record in enumerate(session.records):
        update_annotation(
            session,
            record.sample_id,
            corners=_corners(record),
            scenario_tags=("clean", "grid-heavy" if index == 0 else "shadow"),
            sequence_id=f"capture-{index + 1}",
        )


def _load_cli_module():
    script_path = Path(__file__).parents[2] / "scripts" / "annotate-timesheet-benchmark.py"
    spec = importlib.util.spec_from_file_location("annotate_timesheet_benchmark", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_discovery_is_deterministic_and_uses_exif_normalized_dimensions(tmp_path: Path):
    root = tmp_path / "images"
    _create_images(root, count=2)
    rotated = Image.new("RGB", (40, 60), color=(230, 230, 230))
    exif = rotated.getexif()
    exif[0x0112] = 6
    rotated.save(root / "sheet-000.jpg", exif=exif)

    session = discover_annotation_session(root, dataset="timesheet-v1", source=_source())

    assert [record.image_path for record in session.records] == [
        "sheet-000.jpg",
        "sheet-001.png",
        "sheet-002.png",
    ]
    assert session.records[0].status == "pending"
    assert (session.records[0].width, session.records[0].height) == (60, 40)
    assert session.records[0].sample_id == "timesheet-v1:sheet-000.jpg"


def test_session_round_trip_overlay_and_coverage_summary(tmp_path: Path):
    root = tmp_path / "images"
    _create_images(root)
    session = discover_annotation_session(root, dataset="timesheet-v1", source=_source())
    record = update_annotation(
        session,
        session.records[0].sample_id,
        corners=_corners(session.records[0]),
        scenario_tags=("overlap", "shadow"),
        sequence_id="sequence-01",
    )

    source_before = load_image(root / record.image_path).image.copy()
    overlay = render_annotation_overlay(root / record.image_path, record)
    assert overlay.dtype == np.uint8
    assert overlay.shape == source_before.shape
    assert np.array_equal(load_image(root / record.image_path).image, source_before)

    session_path = tmp_path / "session.json"
    session.save(session_path)
    loaded = AnnotationSession.load(session_path)
    summary = summarize_annotation_session(loaded)
    assert summary["sampleCount"] == 3
    assert summary["status"] == {"annotated": 1, "pending": 2}
    assert summary["scenarioTags"] == {"overlap": 1, "shadow": 1}


def test_finalize_builds_as32_manifest_with_explicit_sequence_ids(tmp_path: Path):
    root = tmp_path / "images"
    _create_images(root)
    session = discover_annotation_session(root, dataset="timesheet-v1", source=_source())
    _annotate_all(session)

    manifest = finalize_annotation_session(session, minimum_samples=1, maximum_samples=3)
    manifest.validate_references(root)

    assert len(manifest.samples) == 3
    assert [sample.sequence_id for sample in manifest.samples] == [
        "capture-1",
        "capture-2",
        "capture-3",
    ]
    assert manifest.samples[0].scenario_tags == ["clean", "grid-heavy"]


def test_finalize_accepts_explicit_out_of_frame_policy(tmp_path: Path):
    root = tmp_path / "images"
    _create_images(root, count=1)
    session = discover_annotation_session(root, dataset="timesheet-v1", source=_source())
    update_annotation(
        session,
        session.records[0].sample_id,
        corners=((-10.0, -10.0), (110.0, -10.0), (110.0, 110.0), (-10.0, 110.0)),
        document_visible=False,
        visibility_reason="out_of_frame",
    )

    manifest = finalize_annotation_session(session, minimum_samples=1, maximum_samples=1)

    assert manifest.samples[0].document_visible is False
    assert manifest.samples[0].visibility_reason == "out_of_frame"


def test_finalize_rejects_pending_and_visible_out_of_bounds_records(tmp_path: Path):
    root = tmp_path / "images"
    _create_images(root, count=2)
    session = discover_annotation_session(root, dataset="timesheet-v1", source=_source())
    update_annotation(session, session.records[0].sample_id, corners=_corners(session.records[0]))
    with pytest.raises(ValueError, match="Unannotated records remain"):
        finalize_annotation_session(session, minimum_samples=1, maximum_samples=2)

    update_annotation(
        session,
        session.records[1].sample_id,
        corners=((-1.0, 10.0), (110.0, 10.0), (110.0, 90.0), (10.0, 90.0)),
    )
    with pytest.raises(ValueError, match="out-of-bounds"):
        finalize_annotation_session(session, minimum_samples=1, maximum_samples=2)


def test_annotation_paths_reject_traversal_and_symlink_escape(tmp_path: Path):
    root = tmp_path / "images"
    _create_images(root, count=1)
    outside = tmp_path / "outside.png"
    Image.new("RGB", (120, 100), color=(100, 100, 100)).save(outside)
    session = discover_annotation_session(root, dataset="timesheet-v1", source=_source())

    traversal_record = session.records[0].model_copy(update={"image_path": "../outside.png"})
    with pytest.raises(ValueError, match="escapes root"):
        resolve_annotation_image_path(session, traversal_record)

    link = root / "link.png"
    try:
        link.symlink_to(outside)
    except OSError:
        return
    symlink_record = session.records[0].model_copy(update={"image_path": "link.png"})
    with pytest.raises(ValueError, match="escapes root"):
        resolve_annotation_image_path(session, symlink_record)


def test_annotation_session_save_preserves_previous_file_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = tmp_path / "session.json"
    target.write_text("previous", encoding="utf-8")

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(annotation_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        annotation_module.atomic_write_text(target, "new content")

    assert target.read_text(encoding="utf-8") == "previous"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_annotation_cli_init_set_preview_and_finalize(tmp_path: Path):
    module = _load_cli_module()

    root = tmp_path / "images"
    _create_images(root, count=1)
    session_path = tmp_path / "session.json"
    manifest_path = tmp_path / "manifest.json"
    sample_id = "timesheet-v1:sheet-001.png"

    assert (
        module.main(
            [
                "init",
                "--image-root",
                str(root),
                "--session",
                str(session_path),
                "--dataset",
                "timesheet-v1",
                "--source-name",
                "Synthetic fixture",
                "--license",
                "Test-only",
                "--attribution",
                "Tests",
            ]
        )
        == 0
    )
    assert (
        module.main(
            [
                "set",
                "--session",
                str(session_path),
                "--sample-id",
                sample_id,
                "--corners-json",
                "[[10,10],[110,10],[110,90],[10,90]]",
                "--visible",
            ]
        )
        == 0
    )
    overlay_path = tmp_path / "review" / "sheet.png"
    assert (
        module.main(
            [
                "preview",
                "--session",
                str(session_path),
                "--sample-id",
                sample_id,
                "--output",
                str(overlay_path),
            ]
        )
        == 0
    )
    assert overlay_path.is_file()
    assert (
        module.main(
            [
                "finalize",
                "--session",
                str(session_path),
                "--output-manifest",
                str(manifest_path),
                "--min-samples",
                "1",
                "--max-samples",
                "1",
            ]
        )
        == 0
    )
    assert manifest_path.is_file()


def test_annotation_cli_preview_rejects_root_escape(tmp_path: Path):
    module = _load_cli_module()
    root = tmp_path / "images"
    _create_images(root, count=1)
    session = discover_annotation_session(root, dataset="timesheet-v1", source=_source())
    session.records[0] = session.records[0].model_copy(update={"image_path": "../outside.png"})
    session_path = tmp_path / "session.json"
    session.save(session_path)
    output = tmp_path / "escape.png"

    assert (
        module.main(
            [
                "preview",
                "--session",
                str(session_path),
                "--sample-id",
                session.records[0].sample_id,
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert not output.exists()


def test_annotation_cli_default_preview_names_avoid_duplicate_stems(tmp_path: Path):
    module = _load_cli_module()
    root = tmp_path / "images"
    root.mkdir()
    Image.new("RGB", (120, 100), color=(220, 220, 220)).save(root / "sheet.jpg")
    Image.new("RGB", (120, 100), color=(210, 210, 210)).save(root / "sheet.png")
    session_path = tmp_path / "session.json"

    assert (
        module.main(
            [
                "init",
                "--image-root",
                str(root),
                "--session",
                str(session_path),
                "--dataset",
                "timesheet-v1",
                "--source-name",
                "Synthetic fixture",
                "--license",
                "Test-only",
                "--attribution",
                "Tests",
            ]
        )
        == 0
    )
    session = AnnotationSession.load(session_path)
    for record in session.records:
        assert (
            module.main(
                [
                    "preview",
                    "--session",
                    str(session_path),
                    "--sample-id",
                    record.sample_id,
                ]
            )
            == 0
        )

    overlays = sorted((tmp_path / "overlays").glob("*_overlay.png"))
    assert len(overlays) == 2
    assert overlays[0].name != overlays[1].name
