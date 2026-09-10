"""AS-41 OpenCV candidate-generator tests."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import attendance_scanner.cv_candidates as cv_candidates_module
from attendance_scanner.cv_candidates import CvCandidateConfig, generate_cv_candidates
from attendance_scanner.pipeline.load import load_image


def _write_document(path: Path, *, nested: bool = False) -> Path:
    image = Image.new("RGB", (320, 240), color=(35, 35, 35))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 300, 220), fill=(245, 245, 245), outline=(10, 10, 10), width=4)
    if nested:
        draw.rectangle((80, 65, 240, 180), outline=(15, 15, 15), width=4)
        draw.line((85, 100, 235, 100), fill=(15, 15, 15), width=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def test_no_candidate_is_a_stable_empty_result(tmp_path: Path):
    path = tmp_path / "blank.png"
    Image.new("RGB", (320, 240), color=(128, 128, 128)).save(path)
    image = load_image(path)

    first = generate_cv_candidates(image)
    second = generate_cv_candidates(image)

    assert first.candidates == []
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_one_document_returns_candidate_with_diagnostics_and_no_warp(tmp_path: Path):
    path = _write_document(tmp_path / "single.png")
    image = load_image(path)
    source_before = image.image.copy()

    result = generate_cv_candidates(image, config=CvCandidateConfig(max_candidates=4))

    assert result.candidates
    candidate = result.candidates[0]
    assert len(candidate.corners.points) == 4
    assert candidate.corners.source == "cv_contour"
    assert candidate.area_ratio > 0.5
    assert candidate.perimeter_px > 0.0
    assert candidate.corners.diagnostics["convex"] is True
    assert np.array_equal(image.image, source_before)


def test_nested_document_keeps_multiple_candidates_in_deterministic_order(tmp_path: Path):
    path = _write_document(tmp_path / "nested.png", nested=True)
    image = load_image(path)
    config = CvCandidateConfig(max_candidates=8)

    first = generate_cv_candidates(image, config=config)
    second = generate_cv_candidates(image, config=config)

    assert len(first.candidates) >= 2
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert [candidate.candidate_id for candidate in first.candidates] == list(
        range(len(first.candidates))
    )
    assert first.candidates[0].area_ratio > first.candidates[-1].area_ratio


def test_top_n_is_configurable_and_generator_does_not_call_perspective(tmp_path: Path):
    path = _write_document(tmp_path / "nested.png", nested=True)
    image = load_image(path)

    result = generate_cv_candidates(image, config=CvCandidateConfig(max_candidates=1))

    assert len(result.candidates) <= 1
    assert result.preprocessing.max_candidates == 1
    assert result.detection_width == 320
    assert result.detection_height == 240


def test_each_contour_can_keep_distinct_valid_epsilon_approximations(tmp_path: Path, monkeypatch):
    path = _write_document(tmp_path / "epsilon.png")
    image = load_image(path)
    original_approx = cv_candidates_module.cv2.approxPolyDP

    def fake_approximation(contour, epsilon, closed):  # type: ignore[no-untyped-def]
        if epsilon < 20.0:
            return np.array([[[10, 10]], [[310, 10]], [[310, 210]], [[10, 210]]], dtype=np.int32)
        return np.array([[[20, 20]], [[300, 20]], [[300, 200]], [[20, 200]]], dtype=np.int32)

    monkeypatch.setattr(cv_candidates_module.cv2, "approxPolyDP", fake_approximation)
    result = generate_cv_candidates(
        image,
        config=CvCandidateConfig(approximation_epsilon_ratios=(0.01, 0.08), max_candidates=8),
    )
    monkeypatch.setattr(cv_candidates_module.cv2, "approxPolyDP", original_approx)

    by_contour = {}
    for candidate in result.candidates:
        by_contour.setdefault(candidate.source_contour_index, []).append(candidate)
    assert any(len(candidates) >= 2 for candidates in by_contour.values())
