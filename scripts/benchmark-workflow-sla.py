"""Measure the v4 scanner SLA at both process and in-process boundaries."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scanner" / "src"))

from attendance_scanner.performance_benchmark import (  # noqa: E402
    LatencySummary,
    measure_pipeline_performance,
)
from attendance_scanner.fingerprint import compute_sha256  # noqa: E402
from attendance_scanner.segmentation import get_default_segmentation_model_path  # noqa: E402
from attendance_scanner.workflow_sla import evaluate_sla  # noqa: E402

SUPPORTED_MODES = ("classic", "ai_enhanced")


def create_sla_fixture(path: Path, *, width: int = 4000, height: int = 3000) -> Path:
    """Create a deterministic 12MP document-like fixture without user data."""
    if width != 4000 or height != 3000:
        raise ValueError("The v4 SLA fixture must remain 4000x3000 pixels")
    image = Image.new("RGB", (width, height), (84, 84, 84))
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (220, 180, width - 220, height - 180),
        fill=(242, 242, 238),
        outline=(20, 20, 20),
        width=12,
    )
    for y in range(520, height - 360, 180):
        draw.line((360, y, width - 360, y), fill=(90, 90, 90), width=3)
    for x in range(720, width - 480, 420):
        draw.line((x, 420, x, height - 300), fill=(90, 90, 90), width=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=92, optimize=False, progressive=False)
    return path


def _latency(values: list[float]) -> LatencySummary:
    if not values:
        return LatencySummary(count=0)
    array = np.asarray(values, dtype=np.float64)
    return LatencySummary(
        count=len(values),
        mean_ms=float(np.mean(array)),
        p50_ms=float(np.percentile(array, 50)),
        p95_ms=float(np.percentile(array, 95)),
        p99_ms=float(np.percentile(array, 99)),
    )


def _fresh_process_latency(
    fixture: Path,
    *,
    detector_mode: str,
    repetitions: int,
    timeout_seconds: int,
    packaged_artifact: Path | None,
) -> LatencySummary:
    values: list[float] = []
    with tempfile.TemporaryDirectory(prefix="attendance-sla-") as temp_root:
        temp_path = Path(temp_root)
        for index in range(repetitions):
            quick_root = temp_path / f"quick-{index}"
            if packaged_artifact is None:
                command = [
                    sys.executable,
                    "-m",
                    "attendance_scanner.cli",
                    "scan-one",
                ]
            else:
                command = [str(packaged_artifact)]
            command.extend(
                [
                    "--input",
                    str(fixture),
                    "--temp-root",
                    str(quick_root),
                    "--detector-mode",
                    detector_mode,
                    "--orientation",
                    "auto",
                ]
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = os.pathsep.join(
                [str(ROOT / "scanner" / "src"), environment.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep)
            started_at = time.perf_counter()
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            values.append((time.perf_counter() - started_at) * 1000.0)
            if completed.returncode not in {0, 2}:
                raise RuntimeError(
                    f"scan-one failed for {detector_mode}: "
                    f"exit={completed.returncode}, stderr={completed.stderr.strip()}"
                )
            events = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
            if len(events) != 1 or events[0].get("type") != "quick_scan_completed":
                raise RuntimeError("scan-one did not emit exactly one completion event")
            if not events[0].get("success", False):
                raise RuntimeError(
                    f"scan-one returned an unsuccessful event: {events[0].get('message')}"
                )
    return _latency(values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=None)
    parser.add_argument("--detector-mode", choices=SUPPORTED_MODES, action="append", default=None)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--packaged-artifact", type=Path, default=None)
    parser.add_argument(
        "--packaged-model",
        type=Path,
        default=None,
        help="Explicit extracted/declared model file for packaged AI provenance",
    )
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args(argv)
    if args.repetitions < 1 or args.timeout_seconds < 1:
        raise SystemExit("--repetitions and --timeout-seconds must be at least 1")

    generated_fixture = args.fixture is None
    if args.fixture is None:
        fixture = args.output.resolve().with_name(f"{args.output.resolve().stem}-fixture-12mp.jpg")
        create_sla_fixture(fixture)
    else:
        fixture = args.fixture.resolve()
        with Image.open(fixture) as image:
            if image.size != (4000, 3000):
                raise SystemExit("--fixture must be exactly 4000x3000 pixels")

    modes = args.detector_mode or list(SUPPORTED_MODES)
    host_model_path = get_default_segmentation_model_path()
    host_model_available = host_model_path is not None
    packaged_model_path = (
        args.packaged_model.resolve()
        if args.packaged_model is not None and args.packaged_model.is_file()
        else None
    )
    packaged_model_available = (
        packaged_model_path is not None
        if args.packaged_artifact is not None
        else host_model_available
    )
    measurements = []
    for detector_mode in modes:
        if detector_mode == "ai_enhanced" and (
            not host_model_available or not packaged_model_available
        ):
            reason = (
                "ONNX model is not available in the host environment"
                if not host_model_available
                else "Packaged model provenance was not explicitly declared"
            )
            measurements.append(
                {
                    "detectorMode": detector_mode,
                    "skipped": True,
                    "skipReason": reason,
                }
            )
            continue
        end_to_end = _fresh_process_latency(
            fixture,
            detector_mode=detector_mode,
            repetitions=args.repetitions,
            timeout_seconds=args.timeout_seconds,
            packaged_artifact=args.packaged_artifact,
        )
        pipeline_report = measure_pipeline_performance(
            [fixture] * args.repetitions,
            detector_mode=detector_mode,
        )
        measurement = evaluate_sla(
            detector_mode,
            end_to_end=end_to_end,
            pipeline=pipeline_report.latency_ms,
            image_width=4000,
            image_height=3000,
            repetitions=args.repetitions,
            model_available=host_model_available,
        )
        measurement_payload = measurement.model_dump(by_alias=True)
        measurement_payload["withinSla"] = measurement.within_sla
        measurement_payload["pipelineStageLatencyMs"] = {
            stage: latency.model_dump(by_alias=True)
            for stage, latency in pipeline_report.stage_latency_ms.items()
        }
        measurements.append(measurement_payload)

    fixture_sha256 = compute_sha256(fixture)

    report = {
        "schemaVersion": "1.0",
        "fixture": {
            "width": 4000,
            "height": 3000,
            "path": str(fixture),
            "exists": fixture.is_file(),
            "sha256": fixture_sha256,
            "source": "generated" if generated_fixture else "provided",
            "contentIsSynthetic": generated_fixture,
            "recipe": "synthetic document geometry" if generated_fixture else None,
        },
        "measurements": measurements,
        "packagedArtifact": (
            str(args.packaged_artifact.resolve()) if args.packaged_artifact else None
        ),
        "packagedModel": (
            {
                "path": str(packaged_model_path),
                "exists": True,
                "sha256": compute_sha256(packaged_model_path),
            }
            if packaged_model_path is not None
            else None
        ),
        "hostModelAvailable": host_model_available,
        "packagedModelAvailable": packaged_model_available,
        "enforced": args.enforce,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    failed = [
        item
        for item in measurements
        if not item.get("skipped", False) and not item.get("withinSla", False)
    ]
    return 2 if args.enforce and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
