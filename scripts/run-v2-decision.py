"""Run the full V2 detector comparison and emit an explicit decision report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scanner" / "src"))

from attendance_scanner.benchmark_decision import (
    REQUIRED_DETECTORS,
    render_decision_markdown,
    run_full_v2_decision,
)
from attendance_scanner.benchmark_runner import load_benchmark_datasets


def _keyed_paths(values: list[str] | None, *, label: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values or []:
        detector, separator, path = value.partition("=")
        if not separator or not detector or not path:
            raise ValueError(f"{label} must use DETECTOR=PATH syntax")
        if detector in parsed:
            raise ValueError(f"Duplicate {label} detector: {detector}")
        parsed[detector] = Path(path)
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--image-root", type=Path, action="append", default=None)
    parser.add_argument(
        "--predictions",
        action="append",
        default=None,
        help="Prediction bundle mapping in DETECTOR=PATH form; repeat for V2 adapters",
    )
    parser.add_argument(
        "--performance",
        action="append",
        default=None,
        help="Optional performance report mapping in DETECTOR=PATH form",
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        prediction_paths = _keyed_paths(args.predictions, label="--predictions")
        performance_paths = _keyed_paths(args.performance, label="--performance")
        unknown_predictions = set(prediction_paths) - set(REQUIRED_DETECTORS)
        if unknown_predictions:
            raise ValueError(
                f"Unsupported detector in --predictions: {min(unknown_predictions)}"
            )
        datasets = load_benchmark_datasets(args.manifest, args.image_root)
        report = run_full_v2_decision(
            datasets,
            prediction_bundle_paths=prediction_paths,
            performance_reports=performance_paths,
            repo_root=args.repo_root,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report.dump_canonical_json(), encoding="utf-8")
        if args.markdown_output is not None:
            args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
            args.markdown_output.write_text(
                render_decision_markdown(report), encoding="utf-8"
            )
        print(
            json.dumps(
                report.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if report.decision_status == "READY" else 2
    except Exception as exc:  # noqa: BLE001 - report invalid input with stable exit code
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
