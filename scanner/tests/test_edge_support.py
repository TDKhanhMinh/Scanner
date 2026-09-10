"""AS-42 edge-support evidence tests."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from attendance_scanner.edge_support import (
    EdgeSupportConfig,
    build_edge_map,
    score_quad_edge_support,
)
from attendance_scanner.pipeline.load import load_image


def _write_grid_document(path: Path) -> Path:
    image = Image.new("RGB", (320, 240), color=(45, 45, 45))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 300, 220), fill=(245, 245, 245), outline=(0, 0, 0), width=4)
    for x in range(70, 281, 35):
        draw.line((x, 70, x, 180), fill=(20, 20, 20), width=2)
    for y in range(70, 181, 25):
        draw.line((60, y, 280, y), fill=(20, 20, 20), width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def test_quad_aligned_to_outer_edge_scores_above_shifted_quad(tmp_path: Path):
    image = load_image(_write_grid_document(tmp_path / "grid.png"))
    config = EdgeSupportConfig(morph_close_iterations=1, morph_dilate_iterations=0)
    edge_map = build_edge_map(image, config=config)

    aligned = score_quad_edge_support(
        edge_map,
        [(20, 20), (300, 20), (300, 220), (20, 220)],
    )
    shifted = score_quad_edge_support(
        edge_map,
        [(35, 35), (285, 35), (285, 205), (35, 205)],
    )

    assert aligned.overall_score > shifted.overall_score
    assert aligned.weakest_edge_score > shifted.weakest_edge_score
    assert aligned.edge_scores["top"] > 0.8
    assert aligned.diagnostics["config"]["cannyThreshold1"] == 50


def test_grid_lines_do_not_make_inner_quad_outscore_outer_boundary(tmp_path: Path):
    image = load_image(_write_grid_document(tmp_path / "grid.png"))
    edge_map = build_edge_map(image)

    outer = score_quad_edge_support(edge_map, [(20, 20), (300, 20), (300, 220), (20, 220)])
    inner = score_quad_edge_support(edge_map, [(60, 70), (280, 70), (280, 180), (60, 180)])

    assert outer.overall_score >= inner.overall_score


def test_border_touching_quad_returns_evidence_and_no_source_mutation(tmp_path: Path):
    image = load_image(_write_grid_document(tmp_path / "grid.png"))
    original = image.image.copy()
    edge_map = build_edge_map(
        image,
        config=EdgeSupportConfig(max_dimension=160, band_ratio=0.02),
    )

    result = score_quad_edge_support(
        edge_map,
        [(0, 10), (319, 15), (319, 230), (0, 225)],
    )

    assert result.band_detection_px >= 1
    assert result.overall_score >= 0.0
    assert edge_map.detection_width == 160
    assert np.array_equal(image.image, original)
