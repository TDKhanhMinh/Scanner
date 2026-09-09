"""AS-32 benchmark manifest and ground-truth contract tests."""

import json
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from attendance_scanner.benchmark import (
    BENCHMARK_SCHEMA_VERSION,
    BenchmarkManifest,
    BenchmarkSample,
    load_benchmark_manifest,
)


def _sample(index: int, **overrides: object) -> dict[str, object]:
    sample: dict[str, object] = {
        "sample_id": f"synthetic-{index:02d}",
        "image_path": f"images/sample-{index:02d}.png",
        "dataset": "synthetic",
        "split": "acceptance" if index >= 4 else "train",
        "width": 640,
        "height": 480,
        "tl": [80.0, 60.0],
        "tr": [560.0, 70.0],
        "br": [550.0, 410.0],
        "bl": [90.0, 400.0],
        "document_visible": True,
        "scenario_tags": ["perspective"],
        "source": {
            "name": "Attendance Scanner synthetic fixtures",
            "license": "internal-synthetic",
            "attribution": "Attendance Scanner test suite",
        },
        "sequence_id": f"sequence-{index:02d}",
    }
    sample.update(overrides)
    return sample


def _manifest(*samples: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "coordinate_space": "exif_normalized_pixels",
        "split_policy": "sequence_group",
        "samples": list(samples),
    }


def test_manifest_accepts_five_valid_ground_truth_samples_and_dumps_canonical_json():
    manifest = BenchmarkManifest.model_validate(
        _manifest(*[_sample(index) for index in range(1, 6)])
    )

    assert len(manifest.samples) == 5
    assert manifest.samples[0].corners[0] == (80.0, 60.0)
    dumped = json.loads(manifest.dump_canonical_json())
    assert dumped["schema_version"] == "1.0"
    assert dumped["samples"][0]["tl"] == [80.0, 60.0]


@pytest.mark.parametrize(
    "payload",
    [
        _sample(1, br=[80.0, 60.0]),
        _sample(2, tl=[-1.0, 60.0]),
        _sample(
            3,
            tl=[80.0, 60.0],
            tr=[550.0, 410.0],
            br=[560.0, 70.0],
            bl=[90.0, 400.0],
        ),
        _sample(4, image_path="../outside.png"),
        _sample(5, document_visible=False, visibility_reason=None),
    ],
)
def test_invalid_ground_truth_samples_are_rejected(payload: dict[str, object]):
    with pytest.raises(ValidationError):
        BenchmarkSample.model_validate(payload)


def test_missing_required_corner_is_rejected():
    payload = _sample(1)
    del payload["br"]

    with pytest.raises(ValidationError, match="br"):
        BenchmarkSample.model_validate(payload)


def test_cyclic_corner_labels_are_rejected_even_when_winding_is_clockwise():
    with pytest.raises(ValidationError, match="semantic"):
        BenchmarkSample.model_validate(
            _sample(
                1,
                tl=[560.0, 70.0],
                tr=[550.0, 410.0],
                br=[90.0, 400.0],
                bl=[80.0, 60.0],
            )
        )


def test_manifest_rejects_duplicate_ids_paths_and_sequence_split_leakage():
    with pytest.raises(ValidationError, match="sample_id"):
        BenchmarkManifest.model_validate(_manifest(_sample(1), _sample(1)))

    duplicate_path = _sample(2, image_path="images/sample-01.png")
    with pytest.raises(ValidationError, match="image_path"):
        BenchmarkManifest.model_validate(_manifest(_sample(1), duplicate_path))

    leaked_sequence = _sample(2, sequence_id="sequence-01", split="tune")
    with pytest.raises(ValidationError, match="multiple splits"):
        BenchmarkManifest.model_validate(_manifest(_sample(1), leaked_sequence))


def test_manifest_rejects_self_intersecting_optional_polygon():
    with pytest.raises(ValidationError, match="self-intersect"):
        BenchmarkManifest.model_validate(
            _manifest(
                _sample(
                    1,
                    document_polygon=[
                        [80.0, 60.0],
                        [550.0, 410.0],
                        [560.0, 70.0],
                        [90.0, 400.0],
                    ],
                )
            )
        )


def test_visible_document_polygon_must_stay_inside_image_bounds():
    with pytest.raises(ValidationError, match="document_polygon"):
        BenchmarkManifest.model_validate(
            _manifest(
                _sample(
                    1,
                    document_polygon=[
                        [-1.0, 60.0],
                        [560.0, 70.0],
                        [550.0, 410.0],
                        [90.0, 400.0],
                    ],
                )
            )
        )


def test_validate_references_checks_exif_normalized_dimensions_and_mask(tmp_path: Path):
    image_path = tmp_path / "images" / "sample.png"
    mask_path = tmp_path / "masks" / "sample.png"
    image_path.parent.mkdir(parents=True)
    mask_path.parent.mkdir(parents=True)
    Image.new("RGB", (640, 480), color=(230, 230, 230)).save(image_path)
    Image.new("L", (640, 480), color=255).save(mask_path)

    payload = _sample(1, image_path="images/sample.png", mask_path="masks/sample.png")
    manifest = BenchmarkManifest.model_validate(_manifest(payload))
    manifest.validate_references(tmp_path)

    wrong_dimensions = manifest.model_copy(
        update={"samples": [manifest.samples[0].model_copy(update={"width": 320})]}
    )
    with pytest.raises(ValueError, match="dimensions"):
        wrong_dimensions.validate_references(tmp_path)

    wrong_mask_path = tmp_path / "masks" / "wrong.png"
    Image.new("L", (320, 240), color=255).save(wrong_mask_path)
    wrong_mask = manifest.model_copy(
        update={
            "samples": [manifest.samples[0].model_copy(update={"mask_path": "masks/wrong.png"})]
        }
    )
    with pytest.raises(ValueError, match="mask dimensions"):
        wrong_mask.validate_references(tmp_path)


def test_validate_references_uses_exif_normalized_dimensions(tmp_path: Path):
    image_path = tmp_path / "images" / "oriented.jpg"
    image_path.parent.mkdir(parents=True)
    image = Image.new("RGB", (480, 640), color=(230, 230, 230))
    exif = image.getexif()
    exif[0x0112] = 6
    image.save(image_path, "JPEG", exif=exif)

    manifest = BenchmarkManifest.model_validate(
        _manifest(
            _sample(
                1,
                image_path="images/oriented.jpg",
                width=640,
                height=480,
            )
        )
    )

    manifest.validate_references(tmp_path)


def test_load_benchmark_manifest_reads_json_and_validates_schema_only(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(_manifest(*[_sample(index) for index in range(1, 6)])),
        encoding="utf-8",
    )

    manifest = load_benchmark_manifest(manifest_path)

    assert manifest.schema_version == BENCHMARK_SCHEMA_VERSION
    assert len(manifest.samples) == 5
