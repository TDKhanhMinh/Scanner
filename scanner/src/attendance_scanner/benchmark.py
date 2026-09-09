"""Versioned benchmark manifest and ground-truth validation contracts.

The benchmark contract is intentionally independent from the runtime JSONL
protocol. Coordinates are always pixel coordinates after EXIF normalization and
are stored in canonical ``TL -> TR -> BR -> BL`` order.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

BENCHMARK_SCHEMA_VERSION: Literal["1.0"] = "1.0"
BenchmarkPoint = Tuple[float, float]
BenchmarkSplit = Literal["train", "tune", "acceptance"]
VisibilityReason = Literal["fully_visible", "border_touching", "out_of_frame", "occluded"]


class BenchmarkSource(BaseModel):
    """Dataset provenance and license metadata for one benchmark sample."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    license: str = Field(min_length=1)
    attribution: str = Field(min_length=1)
    url: Optional[str] = None


def _cross(a: BenchmarkPoint, b: BenchmarkPoint, c: BenchmarkPoint) -> float:
    """Return the signed cross product of AB and AC."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _polygon_area(points: List[BenchmarkPoint]) -> float:
    """Return the absolute shoelace area of a polygon."""
    return abs(
        sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
        / 2.0
    )


def _orientation(a: BenchmarkPoint, b: BenchmarkPoint, c: BenchmarkPoint) -> int:
    """Return the orientation sign for three points."""
    value = _cross(a, b, c)
    if abs(value) < 1e-6:
        return 0
    return 1 if value > 0 else -1


def _segments_intersect(
    first_start: BenchmarkPoint,
    first_end: BenchmarkPoint,
    second_start: BenchmarkPoint,
    second_end: BenchmarkPoint,
) -> bool:
    """Return whether two closed line segments intersect."""
    first_orientations = (
        _orientation(first_start, first_end, second_start),
        _orientation(first_start, first_end, second_end),
    )
    second_orientations = (
        _orientation(second_start, second_end, first_start),
        _orientation(second_start, second_end, first_end),
    )
    return (
        first_orientations[0] != first_orientations[1]
        and second_orientations[0] != second_orientations[1]
    )


def _validate_polygon_geometry(points: List[BenchmarkPoint], *, label: str) -> None:
    """Validate a non-degenerate polygon without self-intersections."""
    if len(points) < 3:
        raise ValueError(f"{label} must contain at least three points")
    if any(not math.isfinite(value) for point in points for value in point):
        raise ValueError(f"{label} contains non-finite coordinates")
    if _polygon_area(points) < 1.0:
        raise ValueError(f"{label} area is too small")
    for first_index in range(len(points)):
        first_end_index = (first_index + 1) % len(points)
        for second_index in range(first_index + 1, len(points)):
            second_end_index = (second_index + 1) % len(points)
            if first_index in (second_index, second_end_index) or first_end_index in (
                second_index,
                second_end_index,
            ):
                continue
            if _segments_intersect(
                points[first_index],
                points[first_end_index],
                points[second_index],
                points[second_end_index],
            ):
                raise ValueError(f"{label} must not self-intersect")


def _validate_quad_geometry(points: List[BenchmarkPoint], *, label: str) -> None:
    """Validate a convex, non-degenerate quadrilateral in canonical order."""
    if len(points) != 4:
        raise ValueError(f"{label} must contain exactly four points")
    if any(not math.isfinite(value) for point in points for value in point):
        raise ValueError(f"{label} contains non-finite coordinates")
    if len({(round(point[0], 6), round(point[1], 6)) for point in points}) != 4:
        raise ValueError(f"{label} contains duplicate points")

    crosses = [
        _cross(points[index], points[(index + 1) % 4], points[(index + 2) % 4])
        for index in range(4)
    ]
    # With image coordinates (x right, y down), TL -> TR -> BR -> BL is
    # clockwise and therefore has positive cross products.
    if any(value <= 1e-6 for value in crosses):
        raise ValueError(
            f"{label} must be convex, non-self-intersecting, and ordered TL -> TR -> BR -> BL"
        )
    sums = [point[0] + point[1] for point in points]
    differences = [point[1] - point[0] for point in points]
    expected_indices = [
        min(range(4), key=lambda index: (sums[index], index)),
        min(range(4), key=lambda index: (differences[index], index)),
        max(range(4), key=lambda index: (sums[index], -index)),
        max(range(4), key=lambda index: (differences[index], -index)),
    ]
    if len(set(expected_indices)) != 4 or expected_indices != [0, 1, 2, 3]:
        raise ValueError(f"{label} does not use semantic TL -> TR -> BR -> BL labels")
    if _polygon_area(points) < 1.0:
        raise ValueError(f"{label} area is too small")


def _validate_relative_path(value: str, *, label: str) -> None:
    """Reject absolute or escaping paths in a benchmark manifest."""
    candidate = Path(value)
    if not value or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{label} must be a non-empty relative path inside the dataset root")


class BenchmarkSample(BaseModel):
    """One image and its canonical document ground truth."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    image_path: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    split: BenchmarkSplit
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    tl: BenchmarkPoint
    tr: BenchmarkPoint
    br: BenchmarkPoint
    bl: BenchmarkPoint
    document_polygon: Optional[List[BenchmarkPoint]] = Field(default=None, min_length=3)
    mask_path: Optional[str] = None
    document_visible: bool = True
    visibility_reason: Optional[VisibilityReason] = None
    scenario_tags: List[str] = Field(default_factory=list)
    source: BenchmarkSource
    sequence_id: Optional[str] = None

    @model_validator(mode="after")
    def validate_ground_truth(self) -> "BenchmarkSample":
        """Validate paths, corner order, bounds, and visibility policy."""
        _validate_relative_path(self.image_path, label="image_path")
        if self.mask_path is not None:
            _validate_relative_path(self.mask_path, label="mask_path")
        if any(not tag.strip() for tag in self.scenario_tags):
            raise ValueError("scenario_tags must not contain empty values")

        corners = [self.tl, self.tr, self.br, self.bl]
        _validate_quad_geometry(corners, label="document corners")
        if self.document_polygon is not None:
            _validate_polygon_geometry(self.document_polygon, label="document_polygon")
            if self.document_visible and any(
                not (0.0 <= point[0] < self.width and 0.0 <= point[1] < self.height)
                for point in self.document_polygon
            ):
                raise ValueError(
                    "document_polygon must be inside image bounds for a visible document"
                )

        if self.document_visible:
            if self.visibility_reason not in (None, "fully_visible"):
                raise ValueError("visible samples may only use visibility_reason=fully_visible")
            for label, point in zip(("tl", "tr", "br", "bl"), corners, strict=True):
                if not (0.0 <= point[0] < self.width and 0.0 <= point[1] < self.height):
                    raise ValueError(f"{label} must be inside image bounds for a visible document")
        elif self.visibility_reason in (None, "fully_visible"):
            raise ValueError(
                "invisible samples require visibility_reason="
                "border_touching, out_of_frame, or occluded"
            )
        return self

    @property
    def corners(self) -> List[BenchmarkPoint]:
        """Return corners in canonical detector order."""
        return [self.tl, self.tr, self.br, self.bl]


class BenchmarkManifest(BaseModel):
    """Canonical JSON benchmark manifest for V2 detection metrics."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = BENCHMARK_SCHEMA_VERSION
    coordinate_space: Literal["exif_normalized_pixels"] = "exif_normalized_pixels"
    split_policy: Literal["sequence_group"] = "sequence_group"
    samples: List[BenchmarkSample] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_manifest_identity(self) -> "BenchmarkManifest":
        """Ensure sample IDs, image paths, and sequence split assignments are stable."""
        sample_ids = [sample.sample_id for sample in self.samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("sample_id values must be unique")
        image_paths = [sample.image_path.replace("\\", "/") for sample in self.samples]
        if len(image_paths) != len(set(image_paths)):
            raise ValueError("image_path values must be unique")

        split_by_sequence: Dict[str, BenchmarkSplit] = {}
        for sample in self.samples:
            if sample.sequence_id is None:
                continue
            previous_split = split_by_sequence.setdefault(sample.sequence_id, sample.split)
            if previous_split != sample.split:
                raise ValueError(
                    f"sequence_id {sample.sequence_id!r} appears in multiple splits: "
                    f"{previous_split!r} and {sample.split!r}"
                )
        return self

    def dump_canonical_json(self) -> str:
        """Serialize the manifest deterministically for review and benchmark hashing."""
        return (
            json.dumps(
                self.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    def validate_references(self, image_root: Union[str, Path]) -> None:
        """Validate referenced image/mask files and EXIF-normalized dimensions."""
        root = Path(image_root).resolve()
        for sample in self.samples:
            image_path = (root / sample.image_path).resolve()
            _assert_inside_root(image_path, root, label="image_path")
            if not image_path.is_file():
                raise ValueError(f"Referenced image does not exist: {sample.image_path}")

            # Import the project loader lazily so schema-only validation remains
            # usable without decoding every image.
            from attendance_scanner.pipeline.load import load_image

            loaded = load_image(image_path)
            if (loaded.width, loaded.height) != (sample.width, sample.height):
                raise ValueError(
                    f"{sample.sample_id}: manifest dimensions {sample.width}x{sample.height} "
                    f"do not match EXIF-normalized image {loaded.width}x{loaded.height}"
                )
            if sample.mask_path is not None:
                mask_path = (root / sample.mask_path).resolve()
                _assert_inside_root(mask_path, root, label="mask_path")
                if not mask_path.is_file():
                    raise ValueError(f"Referenced mask does not exist: {sample.mask_path}")
                from PIL import Image

                with Image.open(mask_path) as mask:
                    mask.load()
                    if mask.size != (sample.width, sample.height):
                        raise ValueError(
                            f"{sample.sample_id}: mask dimensions {mask.width}x{mask.height} "
                            f"do not match image {sample.width}x{sample.height}"
                        )


def _assert_inside_root(path: Path, root: Path, *, label: str) -> None:
    """Ensure a resolved manifest reference cannot escape its dataset root."""
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes dataset root: {path}") from exc


def load_benchmark_manifest(
    path: Union[str, Path],
    *,
    image_root: Optional[Union[str, Path]] = None,
) -> BenchmarkManifest:
    """Load and validate a canonical JSON manifest, optionally checking files."""
    manifest_path = Path(path)
    manifest = BenchmarkManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if image_root is not None:
        manifest.validate_references(image_root)
    return manifest


def main(argv: Optional[List[str]] = None) -> int:
    """Validate a benchmark manifest from the command line."""
    parser = argparse.ArgumentParser(
        description="Validate an Attendance Scanner V2 benchmark manifest"
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--image-root", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        manifest = load_benchmark_manifest(args.manifest, image_root=args.image_root)
    except Exception as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=False))
        return 1
    print(
        json.dumps(
            {
                "valid": True,
                "schemaVersion": manifest.schema_version,
                "sampleCount": len(manifest.samples),
                "splits": sorted({sample.split for sample in manifest.samples}),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "BenchmarkManifest",
    "BenchmarkSample",
    "BenchmarkSource",
    "load_benchmark_manifest",
    "main",
]
