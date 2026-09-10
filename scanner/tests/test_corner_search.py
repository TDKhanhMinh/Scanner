"""AS-47 local corner ROI and edge-search tests."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from attendance_scanner.corner_search import CornerSearchConfig, search_corner_rois
from attendance_scanner.pipeline.load import load_image


def _write_document(path: Path, *, grid: bool = False) -> Path:
    image = Image.new("RGB", (320, 240), color=(45, 45, 45))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 300, 220), fill=(245, 245, 245), outline=(0, 0, 0), width=4)
    if grid:
        for x in range(70, 281, 35):
            draw.line((x, 70, x, 180), fill=(20, 20, 20), width=2)
        for y in range(70, 181, 25):
            draw.line((60, y, 280, y), fill=(20, 20, 20), width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def test_shifted_corner_finds_expected_edges_in_bounded_roi(tmp_path: Path):
    image = load_image(_write_document(tmp_path / "sheet.png"))
    predicted = [(25, 25), (295, 25), (295, 215), (25, 215)]

    result = search_corner_rois(
        image,
        predicted,
        config=CornerSearchConfig(
            minimum_radius_px=12,
            maximum_radius_px=24,
            samples_per_side=16,
        ),
    )

    top_left = result.corners[0]
    detected = [sample for sample in top_left.samples if sample.edge_detected]
    assert len(result.corners) == 4
    assert top_left.roi_xywh[2] <= 49
    assert top_left.roi_xywh[3] <= 49
    assert detected
    assert any(sample.point[0] <= 23 or sample.point[1] <= 23 for sample in detected)
    assert top_left.confidence > 0.0
    assert result.diagnostics["roiCount"] == 4


def test_border_corner_roi_is_clipped_and_grid_lines_do_not_dominate(tmp_path: Path):
    image = load_image(_write_document(tmp_path / "grid.png", grid=True))
    result = search_corner_rois(
        image,
        [(0, 0), (300, 20), (300, 220), (20, 220)],
        config=CornerSearchConfig(minimum_radius_px=20, maximum_radius_px=20),
    )

    top_left = result.corners[0]
    assert top_left.roi_xywh[0] == 0
    assert top_left.roi_xywh[1] == 0
    assert top_left.roi_xywh[2] <= 21
    assert top_left.roi_xywh[3] <= 21
    assert all(
        0 <= sample.point[0] < 320 and 0 <= sample.point[1] < 240 for sample in top_left.samples
    )


def test_debug_roi_artifacts_are_bounded_and_search_does_not_mutate_source(tmp_path: Path):
    path = _write_document(tmp_path / "sheet.png")
    image = load_image(path)
    original = image.image.copy()
    debug_directory = tmp_path / "roi-debug"

    result = search_corner_rois(
        image,
        [(25, 25), (295, 25), (295, 215), (25, 215)],
        config=CornerSearchConfig(debug_directory=debug_directory, samples_per_side=8),
    )

    assert len(result.debug_artifacts) == 4
    assert all(Path(path).is_file() for path in result.debug_artifacts)
    assert np.array_equal(image.image, original)
