"""Deterministic SmartDoc subset import into the AS-32 benchmark manifest."""

from __future__ import annotations

import csv
import gzip
import hashlib
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from PIL import Image, ImageOps

from attendance_scanner.benchmark import (
    BenchmarkManifest,
    BenchmarkSample,
    BenchmarkSource,
    BenchmarkSplit,
)

SMARTDOC_DATASET_NAME = "smartdoc2015-ch1"
SMARTDOC_LICENSE = "CC-BY-4.0"
SMARTDOC_SOURCE_URL = "https://sites.google.com/site/icdar15smartdoc/challenge-1/dataset"
SMARTDOC_ATTRIBUTION = (
    "Burie et al., ICDAR2015 Competition on Smartphone Document Capture and OCR (SmartDoc)"
)


@dataclass(frozen=True)
class SmartDocImportConfig:
    """Reproducible selection policy for a local SmartDoc archive."""

    target_count: int = 250
    seed: int = 20260909
    model_types: Tuple[str, ...] = ("datasheet", "tax")
    backgrounds: Optional[Tuple[str, ...]] = None


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _read_metadata(metadata_path: Path) -> List[Dict[str, str]]:
    """Read SmartDoc metadata.csv or metadata.csv.gz without changing source files."""
    if metadata_path.suffix.lower() == ".gz":
        stream = gzip.open(metadata_path, "rt", encoding="utf-8", newline="")
    else:
        stream = metadata_path.open("r", encoding="utf-8", newline="")
    with stream:
        return list(csv.DictReader(stream))


def _resolve_source_path(source_root: Path, relative_path: str) -> Path:
    """Resolve a metadata image path while preventing source-root escape."""
    candidate = (source_root / relative_path.replace("/", "\\")).resolve()
    try:
        candidate.relative_to(source_root.resolve())
    except ValueError as exc:
        raise ValueError(f"SmartDoc image path escapes source root: {relative_path}") from exc
    return candidate


def _transform_exif_point(
    point: Tuple[float, float],
    raw_width: int,
    raw_height: int,
    orientation: int,
) -> Tuple[float, float]:
    """Map a raw-image point into the ImageOps.exif_transpose coordinate space."""
    x, y = point
    if orientation == 2:
        return raw_width - 1 - x, y
    if orientation == 3:
        return raw_width - 1 - x, raw_height - 1 - y
    if orientation == 4:
        return x, raw_height - 1 - y
    if orientation == 5:
        return y, x
    if orientation == 6:
        return raw_height - 1 - y, x
    if orientation == 7:
        return raw_height - 1 - y, raw_width - 1 - x
    if orientation == 8:
        return y, raw_width - 1 - x
    return x, y


def _normalized_image_geometry(
    image_path: Path,
    raw_corners: Sequence[Tuple[float, float]],
) -> Tuple[int, int, List[Tuple[float, float]]]:
    """Return post-EXIF dimensions and transformed ground-truth corners."""
    with Image.open(image_path) as image:
        image.load()
        raw_width, raw_height = image.size
        exif = image.getexif()
        orientation = int(exif.get(0x0112, 1) or 1)
        normalized = ImageOps.exif_transpose(image)
        if normalized is None:
            normalized_width, normalized_height = raw_width, raw_height
        else:
            normalized_width, normalized_height = normalized.size
    transformed_corners = [
        _transform_exif_point(point, raw_width, raw_height, orientation) for point in raw_corners
    ]
    return normalized_width, normalized_height, _canonicalize_corners(transformed_corners)


def _canonicalize_corners(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Reassign transformed points to semantic TL/TR/BR/BL labels."""
    if len(points) != 4:
        raise ValueError("SmartDoc ground truth must contain exactly four corners")
    sums = [point[0] + point[1] for point in points]
    differences = [point[1] - point[0] for point in points]
    indices = [
        min(range(4), key=lambda index: (sums[index], index)),
        min(range(4), key=lambda index: (differences[index], index)),
        max(range(4), key=lambda index: (sums[index], -index)),
        max(range(4), key=lambda index: (differences[index], -index)),
    ]
    if len(set(indices)) != 4:
        raise ValueError("SmartDoc ground truth corners are ambiguous after EXIF normalization")
    return [points[index] for index in indices]


def _split_for_sequence(sequence_id: str, seed: int) -> BenchmarkSplit:
    """Assign an entire capture sequence to one reproducible split."""
    bucket = int(_stable_key(seed, sequence_id)[:8], 16) % 10
    if bucket < 6:
        return "train"
    if bucket < 8:
        return "tune"
    return "acceptance"


def _select_rows(
    rows: Iterable[Dict[str, str]],
    config: SmartDocImportConfig,
) -> List[Dict[str, str]]:
    """Select rows round-robin across model/background buckets deterministically."""
    if config.target_count < 1:
        raise ValueError("target_count must be positive")
    allowed_types = set(config.model_types)
    allowed_backgrounds = set(config.backgrounds) if config.backgrounds else None
    buckets: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    seen_paths: set[str] = set()
    for row in rows:
        model_type = row.get("modeltype_name", "").strip()
        background = row.get("bg_name", "").strip()
        image_path = row.get("image_path", "").replace("\\", "/").strip()
        if model_type not in allowed_types or not background or not image_path:
            continue
        if allowed_backgrounds is not None and background not in allowed_backgrounds:
            continue
        if image_path in seen_paths:
            continue
        seen_paths.add(image_path)
        bucket_key = f"{model_type}:{background}"
        buckets[bucket_key].append(row)

    for bucket_key, bucket_rows in buckets.items():
        bucket_rows.sort(key=lambda row: _stable_key(config.seed, row["image_path"]))
        buckets[bucket_key] = bucket_rows
    if sum(len(bucket_rows) for bucket_rows in buckets.values()) < config.target_count:
        raise ValueError(
            f"Only {sum(len(bucket_rows) for bucket_rows in buckets.values())} eligible SmartDoc "
            f"rows are available; cannot select {config.target_count}"
        )

    selected: List[Dict[str, str]] = []
    cursors = dict.fromkeys(buckets, 0)
    bucket_keys = sorted(buckets)
    while len(selected) < config.target_count:
        progressed = False
        for bucket_key in bucket_keys:
            cursor = cursors[bucket_key]
            bucket_rows = buckets[bucket_key]
            if cursor >= len(bucket_rows):
                continue
            selected.append(bucket_rows[cursor])
            cursors[bucket_key] = cursor + 1
            progressed = True
            if len(selected) == config.target_count:
                break
        if not progressed:
            break
    return selected


def import_smartdoc_subset(
    source_root: Union[str, Path],
    output_root: Union[str, Path],
    *,
    metadata_path: Optional[Union[str, Path]] = None,
    config: Optional[SmartDocImportConfig] = None,
) -> BenchmarkManifest:
    """Copy a deterministic SmartDoc subset and write an AS-32 manifest."""
    cfg = config or SmartDocImportConfig()
    source_root_path = Path(source_root).resolve()
    output_root_path = Path(output_root).resolve()
    metadata = (
        Path(metadata_path).resolve()
        if metadata_path is not None
        else source_root_path / "metadata.csv.gz"
    )
    if not metadata.is_file():
        raise ValueError(f"SmartDoc metadata file does not exist: {metadata}")

    rows = _select_rows(_read_metadata(metadata), cfg)
    prepared_samples: List[Tuple[BenchmarkSample, Path]] = []
    source = BenchmarkSource(
        name="SmartDoc 2015 Challenge 1",
        license=SMARTDOC_LICENSE,
        attribution=SMARTDOC_ATTRIBUTION,
        url=SMARTDOC_SOURCE_URL,
    )
    for row in rows:
        relative_source = row["image_path"].replace("\\", "/")
        source_path = _resolve_source_path(source_root_path, relative_source)
        if not source_path.is_file():
            raise ValueError(f"SmartDoc image does not exist: {relative_source}")
        raw_corners = [
            (float(row["tl_x"]), float(row["tl_y"])),
            (float(row["tr_x"]), float(row["tr_y"])),
            (float(row["br_x"]), float(row["br_y"])),
            (float(row["bl_x"]), float(row["bl_y"])),
        ]
        width, height, corners = _normalized_image_geometry(source_path, raw_corners)
        visible = all(0.0 <= x < width and 0.0 <= y < height for x, y in corners)
        sequence_id = f"{row.get('bg_name', '')}/{row.get('model_name', '')}"
        model_type = row.get("modeltype_name", "unknown")
        background = row.get("bg_name", "unknown")
        split = _split_for_sequence(sequence_id, cfg.seed)
        destination_relative = f"images/{relative_source}"
        prepared_samples.append(
            (
                BenchmarkSample(
                    sample_id=f"{SMARTDOC_DATASET_NAME}:{relative_source}",
                    image_path=destination_relative,
                    dataset=SMARTDOC_DATASET_NAME,
                    split=split,
                    width=width,
                    height=height,
                    tl=corners[0],
                    tr=corners[1],
                    br=corners[2],
                    bl=corners[3],
                    document_visible=visible,
                    visibility_reason=None if visible else "out_of_frame",
                    scenario_tags=[
                        "smartdoc",
                        f"model_type:{model_type}",
                        f"background:{background}",
                    ],
                    source=source,
                    sequence_id=sequence_id,
                ),
                source_path,
            )
        )

    samples = [sample for sample, _ in prepared_samples]
    manifest = BenchmarkManifest(samples=samples)
    output_root_path.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root_path.name}.staging-", dir=output_root_path.parent)
    )
    try:
        for sample, source_path in prepared_samples:
            staging_path = staging_root / sample.image_path
            staging_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, staging_path)
        (staging_root / "manifest.json").write_text(
            manifest.dump_canonical_json(),
            encoding="utf-8",
        )

        for sample, _ in prepared_samples:
            destination_path = output_root_path / sample.image_path
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staging_root / sample.image_path, destination_path)
        shutil.copy2(staging_root / "manifest.json", output_root_path / "manifest.json")
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return manifest


def summarize_manifest(manifest: BenchmarkManifest) -> Dict[str, object]:
    """Return stable counts for a human/machine-readable corpus summary."""
    split_counts = Counter(sample.split for sample in manifest.samples)
    dataset_counts = Counter(sample.dataset for sample in manifest.samples)
    tag_counts = Counter(tag for sample in manifest.samples for tag in sample.scenario_tags)
    return {
        "schemaVersion": manifest.schema_version,
        "sampleCount": len(manifest.samples),
        "bySplit": dict(sorted(split_counts.items())),
        "byDataset": dict(sorted(dataset_counts.items())),
        "scenarioTags": dict(sorted(tag_counts.items())),
    }


__all__ = [
    "SMARTDOC_ATTRIBUTION",
    "SMARTDOC_DATASET_NAME",
    "SMARTDOC_LICENSE",
    "SMARTDOC_SOURCE_URL",
    "SmartDocImportConfig",
    "import_smartdoc_subset",
    "summarize_manifest",
]
