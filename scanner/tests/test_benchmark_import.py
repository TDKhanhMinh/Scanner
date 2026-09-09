"""AS-33 deterministic SmartDoc import tests."""

import csv
import gzip
import shutil
from pathlib import Path

import pytest
from PIL import Image

from attendance_scanner.benchmark_import import (
    SmartDocImportConfig,
    import_smartdoc_subset,
    summarize_manifest,
)


def _create_smartdoc_source(root: Path) -> Path:
    metadata_path = root / "metadata.csv"
    root.mkdir(parents=True)
    rows = []
    for index, (model_type, background) in enumerate(
        (
            ("datasheet", "background01"),
            ("datasheet", "background01"),
            ("datasheet", "background02"),
            ("tax", "background01"),
            ("tax", "background02"),
            ("letter", "background01"),
        ),
        start=1,
    ):
        relative_path = f"{background}/{model_type}001/frame_{index:04d}.jpeg"
        image_path = root / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (320, 240), color=(220, 220, 220)).save(image_path)
        rows.append(
            {
                "image_path": relative_path,
                "modeltype_name": model_type,
                "model_name": f"{model_type}001",
                "bg_name": background,
                "tl_x": "40",
                "tl_y": "30",
                "tr_x": "280",
                "tr_y": "32",
                "br_x": "278",
                "br_y": "210",
                "bl_x": "42",
                "bl_y": "208",
            }
        )
    with metadata_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return metadata_path


def test_import_smartdoc_subset_is_deterministic_and_validates_references(tmp_path: Path):
    source_root = tmp_path / "source"
    output_root = tmp_path / "benchmark"
    metadata_path = _create_smartdoc_source(source_root)
    compressed_metadata = source_root / "metadata.csv.gz"
    with metadata_path.open("rb") as source, gzip.open(compressed_metadata, "wb") as target:
        shutil.copyfileobj(source, target)
    config = SmartDocImportConfig(
        target_count=4,
        seed=7,
        model_types=("datasheet", "tax"),
    )

    first = import_smartdoc_subset(
        source_root,
        output_root,
        metadata_path=compressed_metadata,
        config=config,
    )
    first_json = first.dump_canonical_json()
    second = import_smartdoc_subset(
        source_root,
        output_root,
        metadata_path=compressed_metadata,
        config=config,
    )

    assert len(first.samples) == 4
    assert first_json == second.dump_canonical_json()
    first.validate_references(output_root)
    summary = summarize_manifest(first)
    assert summary["sampleCount"] == 4
    assert sum(summary["bySplit"].values()) == 4
    assert (output_root / "manifest.json").is_file()


def test_import_smartdoc_subset_rejects_insufficient_eligible_rows(tmp_path: Path):
    source_root = tmp_path / "source"
    metadata_path = _create_smartdoc_source(source_root)

    with pytest.raises(ValueError, match="eligible SmartDoc rows"):
        import_smartdoc_subset(
            source_root,
            tmp_path / "benchmark",
            metadata_path=metadata_path,
            config=SmartDocImportConfig(target_count=10, model_types=("datasheet",)),
        )


def test_import_smartdoc_subset_reorders_exif_rotated_ground_truth(tmp_path: Path):
    source_root = tmp_path / "source"
    source_root.mkdir(parents=True)
    image_path = source_root / "background01" / "datasheet001" / "frame_0001.jpg"
    image_path.parent.mkdir(parents=True)
    image = Image.new("RGB", (480, 640), color=(220, 220, 220))
    exif = image.getexif()
    exif[0x0112] = 6
    image.save(image_path, "JPEG", exif=exif)

    metadata_path = source_root / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "image_path",
                "modeltype_name",
                "model_name",
                "bg_name",
                "tl_x",
                "tl_y",
                "tr_x",
                "tr_y",
                "br_x",
                "br_y",
                "bl_x",
                "bl_y",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_path": "background01/datasheet001/frame_0001.jpg",
                "modeltype_name": "datasheet",
                "model_name": "datasheet001",
                "bg_name": "background01",
                "tl_x": "60",
                "tl_y": "80",
                "tr_x": "420",
                "tr_y": "80",
                "br_x": "420",
                "br_y": "560",
                "bl_x": "60",
                "bl_y": "560",
            }
        )

    manifest = import_smartdoc_subset(
        source_root,
        tmp_path / "benchmark",
        metadata_path=metadata_path,
        config=SmartDocImportConfig(target_count=1, model_types=("datasheet",)),
    )

    sample = manifest.samples[0]
    assert (sample.width, sample.height) == (640, 480)
    assert sample.tl[0] < sample.tr[0]
    assert sample.tl[1] < sample.bl[1]
    manifest.validate_references(tmp_path / "benchmark")


def test_import_smartdoc_subset_validates_all_frames_before_materializing(tmp_path: Path):
    source_root = tmp_path / "source"
    metadata_path = _create_smartdoc_source(source_root)
    missing = source_root / "background01" / "datasheet001" / "frame_0001.jpeg"
    missing.unlink()
    output_root = tmp_path / "benchmark"

    with pytest.raises(ValueError, match="does not exist"):
        import_smartdoc_subset(
            source_root,
            output_root,
            metadata_path=metadata_path,
            config=SmartDocImportConfig(
                target_count=5,
                model_types=("datasheet", "tax"),
            ),
        )

    assert not output_root.exists()
