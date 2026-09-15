"""SLA benchmark contract tests for the v4 workflow boundary metrics."""

from attendance_scanner.performance_benchmark import LatencySummary
from attendance_scanner.workflow_sla import evaluate_sla


def _latency(*, p50: float | None, p95: float | None) -> LatencySummary:
    return LatencySummary(count=3, mean_ms=p50, p50_ms=p50, p95_ms=p95, p99_ms=p95)


def test_classical_sla_evaluation_requires_both_boundaries_to_pass():
    report = evaluate_sla(
        "classic",
        end_to_end=_latency(p50=1100.0, p95=1700.0),
        pipeline=_latency(p50=350.0, p95=650.0),
        image_width=4000,
        image_height=3000,
        repetitions=3,
    )

    assert report.within_sla is True
    assert report.end_to_end.within_sla is True
    assert report.pipeline.within_sla is True


def test_sla_evaluation_rejects_missing_or_exceeded_percentiles():
    missing = evaluate_sla(
        "classic",
        end_to_end=_latency(p50=None, p95=None),
        pipeline=_latency(p50=350.0, p95=650.0),
        image_width=4000,
        image_height=3000,
        repetitions=3,
    )
    exceeded = evaluate_sla(
        "classic",
        end_to_end=_latency(p50=1201.0, p95=1700.0),
        pipeline=_latency(p50=350.0, p95=650.0),
        image_width=4000,
        image_height=3000,
        repetitions=3,
    )

    assert missing.within_sla is False
    assert exceeded.within_sla is False
