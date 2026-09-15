"""Reproducible SLA contracts and evaluation for the multi-workflow scanner."""

from __future__ import annotations

from typing import Dict, Literal

from pydantic import Field

from attendance_scanner.contracts import BaseContract
from attendance_scanner.performance_benchmark import LatencySummary

SLA_BENCHMARK_VERSION = "1.0"
SlaDetectorMode = Literal["classic", "ai_enhanced"]


class SlaThreshold(BaseContract):
    """P50/P95 limits for one detector mode and one execution boundary."""

    p50_ms: float = Field(ge=0.0)
    p95_ms: float = Field(ge=0.0)


SLA_THRESHOLDS: Dict[SlaDetectorMode, Dict[str, SlaThreshold]] = {
    "classic": {
        "end_to_end": SlaThreshold(p50_ms=1200.0, p95_ms=1800.0),
        "pipeline": SlaThreshold(p50_ms=400.0, p95_ms=700.0),
    },
    "ai_enhanced": {
        "end_to_end": SlaThreshold(p50_ms=2800.0, p95_ms=3800.0),
        "pipeline": SlaThreshold(p50_ms=1200.0, p95_ms=1800.0),
    },
}


class SlaBoundaryResult(BaseContract):
    """Measured latency distribution and explicit threshold outcome."""

    latency: LatencySummary
    threshold: SlaThreshold
    within_sla: bool


class WorkflowSlaMeasurement(BaseContract):
    """One detector-mode measurement over a fixed-size fixture."""

    detector_mode: SlaDetectorMode
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    repetitions: int = Field(ge=1)
    model_available: bool = False
    end_to_end: SlaBoundaryResult
    pipeline: SlaBoundaryResult

    @property
    def within_sla(self) -> bool:
        """Return true only when the mode was measured and both boundaries pass."""
        return (
            (self.detector_mode == "classic" or self.model_available)
            and self.end_to_end.within_sla
            and self.pipeline.within_sla
        )


def evaluate_sla(
    detector_mode: SlaDetectorMode,
    *,
    end_to_end: LatencySummary,
    pipeline: LatencySummary,
    image_width: int,
    image_height: int,
    repetitions: int,
    model_available: bool = False,
) -> WorkflowSlaMeasurement:
    """Evaluate measured P50/P95 values without inventing missing percentiles."""
    thresholds = SLA_THRESHOLDS[detector_mode]

    def within(latency: LatencySummary, threshold: SlaThreshold) -> bool:
        return (
            latency.p50_ms is not None
            and latency.p95_ms is not None
            and latency.p50_ms <= threshold.p50_ms
            and latency.p95_ms <= threshold.p95_ms
        )

    return WorkflowSlaMeasurement(
        detector_mode=detector_mode,
        image_width=image_width,
        image_height=image_height,
        repetitions=repetitions,
        model_available=model_available,
        end_to_end=SlaBoundaryResult(
            latency=end_to_end,
            threshold=thresholds["end_to_end"],
            within_sla=within(end_to_end, thresholds["end_to_end"]),
        ),
        pipeline=SlaBoundaryResult(
            latency=pipeline,
            threshold=thresholds["pipeline"],
            within_sla=within(pipeline, thresholds["pipeline"]),
        ),
    )


__all__ = [
    "SLA_BENCHMARK_VERSION",
    "SLA_THRESHOLDS",
    "SlaBoundaryResult",
    "SlaDetectorMode",
    "SlaThreshold",
    "WorkflowSlaMeasurement",
    "evaluate_sla",
]
