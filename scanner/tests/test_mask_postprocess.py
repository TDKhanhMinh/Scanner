"""AS-39 mask cleanup, components, and transform tests."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from attendance_scanner.mask_postprocess import (
    MaskPostprocessConfig,
    postprocess_document_mask,
)
from attendance_scanner.pipeline.load import load_image
from attendance_scanner.segmentation import (
    SegmentationConfig,
    preprocess_segmentation_input,
)


def _write_image(path: Path, size: tuple[int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(230, 230, 230)).save(path)
    return path


def test_mask_threshold_and_light_morphology_are_configurable():
    probability = np.zeros((20, 20), dtype=np.float32)
    probability[5:15, 5:15] = 0.9
    probability[9, 9] = 0.1
    probability[1, 1] = 0.9

    raw = postprocess_document_mask(
        probability,
        MaskPostprocessConfig(
            threshold=0.8,
            open_kernel_size=0,
            close_kernel_size=0,
            morphology_iterations=0,
            min_component_area_ratio=0.0,
        ),
    )
    cleaned = postprocess_document_mask(
        probability,
        MaskPostprocessConfig(
            threshold=0.8,
            open_kernel_size=3,
            close_kernel_size=3,
            morphology_iterations=1,
            min_component_area_ratio=0.0,
        ),
    )

    assert raw.cleaned_mask[1, 1]
    assert 80 <= cleaned.selected_mask.sum() < 100
    assert cleaned.selected_mask[9, 9]


def test_multi_component_selection_preserves_ambiguity_and_border_diagnostics():
    probability = np.zeros((40, 60), dtype=np.float32)
    probability[10:20, 2:12] = 0.9
    probability[10:20, 48:58] = 0.9
    result = postprocess_document_mask(
        probability,
        MaskPostprocessConfig(
            threshold=0.5,
            close_kernel_size=0,
            morphology_iterations=0,
            min_component_area_ratio=0.0,
            ambiguity_margin=1.0,
        ),
    )

    assert result.component_count == 2
    assert result.ambiguous is True
    assert result.selected_label is not None
    assert result.components[0].border_contact == []
    assert result.components[1].border_contact == []

    border_probability = np.zeros((20, 20), dtype=np.float32)
    border_probability[:, :5] = 0.9
    border_result = postprocess_document_mask(
        border_probability,
        MaskPostprocessConfig(close_kernel_size=0, morphology_iterations=0),
    )
    assert "left" in border_result.border_contact


def test_postprocess_rejects_non_probability_mask_and_even_kernel():
    with pytest.raises(ValueError, match=r"finite values in \[0,1\]"):
        postprocess_document_mask(np.array([[2.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="must be odd"):
        MaskPostprocessConfig(close_kernel_size=2)


def test_letterbox_transform_round_trip_and_no_width_height_swap(tmp_path: Path):
    image = load_image(_write_image(tmp_path / "portrait.png", (300, 500)))
    config = SegmentationConfig(
        input_width=640,
        input_height=480,
        resize_mode="letterbox",
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )

    tensor, transform = preprocess_segmentation_input(image, config)
    source_points = ((0.0, 0.0), (299.0, 0.0), (299.0, 499.0), (0.0, 499.0))
    input_points = tuple(
        (x * transform.scale_x + transform.pad_x, y * transform.scale_y + transform.pad_y)
        for x, y in source_points
    )
    restored = transform.map_points_to_source(input_points)

    assert tensor.shape == (1, 3, 480, 640)
    assert transform.source_width == 300
    assert transform.source_height == 500
    assert transform.pad_x > 0
    assert transform.pad_y == 0
    assert np.allclose(restored, source_points, atol=1.0)


def test_exif_rotated_fixture_uses_normalized_source_dimensions(tmp_path: Path):
    path = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (30, 50), color=(230, 230, 230))
    exif = image.getexif()
    exif[0x0112] = 6
    image.save(path, exif=exif)
    loaded = load_image(path)

    _, transform = preprocess_segmentation_input(
        loaded,
        SegmentationConfig(input_width=64, input_height=64),
    )

    assert (loaded.width, loaded.height) == (50, 30)
    assert (transform.source_width, transform.source_height) == (50, 30)
