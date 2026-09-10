"""AS-40 mask and quadrilateral geometry validator tests."""

import cv2
import numpy as np
import pytest

from attendance_scanner.geometry_validator import (
    GeometryReasonCode,
    GeometryValidationConfig,
    validate_candidate,
    validate_mask,
    validate_quadrilateral,
)


def _trapezoid_mask() -> np.ndarray:
    mask = np.zeros((100, 140), dtype=np.float32)
    cv2.fillPoly(
        mask,
        [np.array([[20, 10], [120, 14], [110, 88], [25, 92]], dtype=np.int32)],
        1.0,
    )
    return mask


def test_valid_trapezoid_mask_and_quad_return_typed_evidence():
    result = validate_candidate(
        mask=_trapezoid_mask(),
        corners=[(20, 10), (120, 14), (110, 88), (25, 92)],
        image_size=(140, 100),
    )

    assert result.valid is True
    assert result.validator_version == "1.0"
    assert result.mask.component_count == 1
    assert result.mask.dominant_component_ratio == 1.0
    assert result.quadrilateral.normalized_corners is not None
    assert result.quadrilateral.area_ratio > 0.5
    assert result.quality_score > 0.5


def test_rotated_quad_is_canonicalized_without_warping():
    result = validate_quadrilateral(
        [(40, 10), (110, 40), (100, 80), (30, 50)],
        image_size=(140, 100),
    )

    assert result.valid is True
    assert result.normalized_corners is not None
    assert result.normalized_corners.tl == (40.0, 10.0)
    assert result.normalized_corners.tr == (110.0, 40.0)


def test_border_touching_is_evidence_not_automatic_rejection():
    result = validate_quadrilateral(
        [(0, 10), (120, 10), (118, 90), (0, 90)],
        image_size=(120, 100),
    )

    assert result.valid is True
    assert GeometryReasonCode.BORDER_TOUCHING in result.reason_codes
    assert "left" in result.border_contact


def test_tiny_mask_is_rejected_and_has_lower_quality_score():
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[48:52, 48:52] = 1.0

    result = validate_mask(mask)

    assert result.valid is False
    assert GeometryReasonCode.MASK_AREA_TOO_SMALL in result.reason_codes
    assert result.quality_score < 0.2


def test_fragmented_mask_is_ambiguous_with_component_diagnostics():
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[10:50, 10:50] = 1.0
    mask[55:95, 55:95] = 1.0

    result = validate_mask(
        mask,
        config=GeometryValidationConfig(mask_min_dominance_ratio=0.7),
    )

    assert result.valid is False
    assert result.component_count == 2
    assert result.dominant_component_ratio == pytest.approx(0.5)
    assert GeometryReasonCode.MASK_FRAGMENTED in result.reason_codes
    assert GeometryReasonCode.MASK_AMBIGUOUS in result.reason_codes


def test_bow_tie_and_line_like_quad_are_rejected():
    bow_tie = validate_quadrilateral(
        [(10, 10), (90, 90), (90, 10), (10, 90)],
        image_size=(100, 100),
    )
    line_like = validate_quadrilateral(
        [(10, 10), (90, 10), (90, 11), (10, 11)],
        image_size=(100, 100),
    )

    assert bow_tie.valid is False
    assert GeometryReasonCode.QUAD_SELF_INTERSECTION in bow_tie.reason_codes
    assert line_like.valid is False
    assert GeometryReasonCode.QUAD_SIDE_TOO_SHORT in line_like.reason_codes
    assert GeometryReasonCode.QUAD_AREA_OUT_OF_RANGE in line_like.reason_codes


def test_nonfinite_and_shape_mismatch_are_explicit_reason_codes():
    mask = np.array([[np.nan]], dtype=np.float32)
    mask_result = validate_mask(mask, expected_shape=(2, 2))
    quad_result = validate_quadrilateral(
        [(float("nan"), 0), (10, 0), (10, 10), (0, 10)],
        image_size=(20, 20),
    )

    assert GeometryReasonCode.MASK_SHAPE_MISMATCH in mask_result.reason_codes
    assert GeometryReasonCode.MASK_NONFINITE in mask_result.reason_codes
    assert GeometryReasonCode.QUAD_NONFINITE in quad_result.reason_codes


def test_non_numeric_mask_returns_typed_dtype_reason_instead_of_raw_type_error():
    result = validate_mask(np.array([["not-a-mask"]], dtype=object))

    assert result.valid is False
    assert result.reason_codes == [GeometryReasonCode.MASK_DTYPE_UNSUPPORTED]


def test_validator_does_not_change_input_mask_or_perform_warp():
    mask = _trapezoid_mask()
    original = mask.copy()

    result = validate_candidate(
        mask=mask,
        corners=[(20, 10), (120, 14), (110, 88), (25, 92)],
        image_size=(140, 100),
    )

    assert np.array_equal(mask, original)
    assert result.quadrilateral.normalized_corners is not None
