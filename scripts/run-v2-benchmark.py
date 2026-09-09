"""Run the reproducible Attendance Scanner V2 benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from attendance_scanner.benchmark_runner import (
    BenchmarkThresholds,
    load_benchmark_datasets,
    load_prediction_bundle,
    run_benchmark,
    write_report_files,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run V2 detector metrics against one or more AS-32 manifests"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        required=True,
        help="AS-32 manifest; repeat for combined datasets",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        action="append",
        default=None,
        help="image root paired with each manifest; defaults to the manifest directory",
    )
    parser.add_argument(
        "--detector",
        choices=("v1_cv", "segmentation_only", "cv_v2", "hybrid", "reference"),
        required=True,
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="prediction bundle for segmentation_only, cv_v2, hybrid, or an external adapter",
    )
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--pipeline-version", default=None)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument(
        "--gate",
        action="store_true",
        help="evaluate AS-35 release targets (98%% success, 0.95 IoU, etc.)",
    )
    parser.add_argument("--min-success-rate", type=float, default=None)
    parser.add_argument("--min-mean-iou", type=float, default=None)
    parser.add_argument("--max-p95-normalized-corner-error", type=float, default=None)
    parser.add_argument("--max-false-detection-rate", type=float, default=None)
    args = parser.parse_args(argv)

    try:
        datasets = load_benchmark_datasets(args.manifest, args.image_root)
        prediction_bundle = (
            load_prediction_bundle(args.predictions)
            if args.predictions is not None
            else None
        )
        custom_threshold = any(
            value is not None
            for value in (
                args.min_success_rate,
                args.min_mean_iou,
                args.max_p95_normalized_corner_error,
                args.max_false_detection_rate,
            )
        )
        if args.gate and custom_threshold:
            parser.error("Use --gate alone or provide custom thresholds, not both")
        thresholds = None
        if args.gate:
            thresholds = BenchmarkThresholds.release_targets()
        elif custom_threshold:
            thresholds = BenchmarkThresholds(
                min_detection_success_rate=args.min_success_rate,
                min_mean_iou=args.min_mean_iou,
                max_p95_normalized_corner_error=args.max_p95_normalized_corner_error,
                max_false_detection_rate=args.max_false_detection_rate,
            )
        report = run_benchmark(
            datasets,
            detector=args.detector,
            prediction_bundle=prediction_bundle,
            thresholds=thresholds,
            model_version=args.model_version,
            pipeline_version=args.pipeline_version,
            repo_root=args.repo_root,
        )
        write_report_files(
            report,
            json_path=args.json_output,
            markdown_path=args.markdown_output,
            csv_path=args.csv_output,
        )
        print(
            json.dumps(
                {
                    "detector": report.detector,
                    "sampleCount": report.metrics["sample_count"],
                    "detectionSuccessRate": report.metrics["detection_success_rate"],
                    "meanIoU": report.metrics["mean_iou"],
                    "thresholdPassed": report.threshold_gate["passed"],
                },
                ensure_ascii=False,
            )
        )
        return 0 if report.threshold_gate["passed"] is not False else 2
    except Exception as exc:  # noqa: BLE001 - CLI reports all runner/input errors as JSON
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
