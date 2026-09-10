"""AS-49 refined corner intersection and quality-gate tests."""

from attendance_scanner.corner_refinement import (
    CornerRefinementConfig,
    RefinementReasonCode,
    refine_document_corners,
)
from attendance_scanner.detector import CanonicalCorners
from attendance_scanner.line_fitting import FittedDocumentLine, FittedLineSet, LineFittingConfig


def _before() -> CanonicalCorners:
    return CanonicalCorners.from_sequence([(12, 12), (108, 12), (108, 88), (12, 88)])


def _line(
    label: str,
    p1: tuple[float, float],
    p2: tuple[float, float],
    *,
    confidence: float = 0.9,
    source: str = "local",
    fallback: bool = False,
) -> FittedDocumentLine:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = (dx * dx + dy * dy) ** 0.5
    nx, ny = -dy / length, dx / length
    return FittedDocumentLine(
        label=label,  # type: ignore[arg-type]
        anchor=p1,
        direction=(dx / length, dy / length),
        normal_a=nx,
        normal_b=ny,
        normal_c=-(nx * p1[0] + ny * p1[1]),
        angle_deg=__import__("math").degrees(__import__("math").atan2(dy, dx)) % 180.0,
        inlier_count=10,
        sample_count=10,
        inlier_ratio=1.0,
        residual_mean_px=0.1,
        residual_p95_px=0.2,
        support_length_px=length,
        confidence=confidence,
        fallback_used=fallback,
        source=source,  # type: ignore[arg-type]
    )


def _perfect_lines() -> FittedLineSet:
    return FittedLineSet(
        source_width=120,
        source_height=100,
        lines=[
            _line("top", (10, 10), (110, 10)),
            _line("right", (110, 10), (110, 90)),
            _line("bottom", (110, 90), (10, 90)),
            _line("left", (10, 90), (10, 10)),
        ],
        config=LineFittingConfig(),
    )


def test_perfect_lines_are_intersected_and_accepted():
    result = refine_document_corners(_before(), _perfect_lines(), image_size=(120, 100))

    assert result.accepted is True
    assert result.selected_corners == CanonicalCorners.from_sequence(
        [(10, 10), (110, 10), (110, 90), (10, 90)]
    )
    assert result.reason_codes == [RefinementReasonCode.ACCEPTED]
    assert result.max_displacement_px < 4.0


def test_near_parallel_or_large_outlier_rejects_and_preserves_original():
    lines = _perfect_lines().model_copy(
        update={
            "lines": [
                _line("top", (10, 10), (110, 10)),
                _line("right", (10, 11), (110, 11)),
                _line("bottom", (110, 90), (10, 90)),
                _line("left", (10, 90), (10, 10)),
            ]
        }
    )
    result = refine_document_corners(
        _before(),
        lines,
        image_size=(120, 100),
        config=CornerRefinementConfig(max_corner_displacement_ratio=0.01),
    )

    assert result.accepted is False
    assert result.selected_corners == _before()
    assert (
        RefinementReasonCode.NEAR_PARALLEL_LINES in result.reason_codes
        or RefinementReasonCode.DISPLACEMENT_TOO_LARGE in result.reason_codes
    )


def test_partial_edge_fit_is_explicit_and_support_drop_rejects():
    partial_lines = _perfect_lines().model_copy(
        update={
            "lines": [
                _line("top", (10, 10), (110, 10)),
                _line("right", (110, 10), (110, 90), source="none", confidence=0.0),
                _line("bottom", (110, 90), (10, 90)),
                _line("left", (10, 90), (10, 10)),
            ]
        }
    )
    partial = refine_document_corners(_before(), partial_lines, image_size=(120, 100))
    dropped = refine_document_corners(
        _before(),
        _perfect_lines(),
        image_size=(120, 100),
        before_edge_support=0.9,
        after_edge_support=0.7,
    )

    assert partial.accepted is True
    assert partial.reason_codes == [RefinementReasonCode.PARTIAL]
    assert "right" in partial.fallback_edge_labels
    assert dropped.accepted is False
    assert RefinementReasonCode.EDGE_SUPPORT_DROP in dropped.reason_codes

    strict = refine_document_corners(
        _before(),
        partial_lines,
        image_size=(120, 100),
        config=CornerRefinementConfig(allow_partial_refinement=False),
    )
    assert strict.accepted is False
    assert strict.selected_corners == _before()
    assert strict.reason_codes == [RefinementReasonCode.PARTIAL_NOT_ALLOWED]
