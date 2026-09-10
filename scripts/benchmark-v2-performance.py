"""Measure V2 pipeline, ONNX lifecycle, memory and packaged artifact facts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scanner" / "src"))

from attendance_scanner.onnx_runtime import OnnxInferenceService, OnnxSessionConfig
from attendance_scanner.performance_benchmark import (
    load_pipeline_report,
    measure_onnx_performance,
    measure_pipeline_performance,
)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _image_paths(root: Path, limit: int) -> list[Path]:
    if limit < 1:
        raise ValueError("--limit must be at least 1")
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return paths[:limit]


def _parse_shape(value: str) -> tuple[int, ...]:
    shape = tuple(int(part.strip()) for part in value.split(","))
    if not shape or any(part <= 0 for part in shape):
        raise ValueError("--input-shape must contain positive comma-separated integers")
    return shape


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    parser.add_argument("--detector-mode", default="classic")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--input-shape", default="1,3,224,224")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--packaged-artifact", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        image_paths = _image_paths(args.input_root, args.limit)
        if not image_paths:
            raise ValueError(f"No supported images found under {args.input_root}")
        baseline = (
            load_pipeline_report(args.baseline) if args.baseline is not None else None
        )
        pipeline = measure_pipeline_performance(
            image_paths,
            detector_mode=args.detector_mode,
            baseline=baseline,
            model_path=args.model,
            packaged_artifact=args.packaged_artifact,
        )
        onnx = None
        if args.model is not None:
            if args.iterations < 1:
                raise ValueError("--iterations must be at least 1")
            service = OnnxInferenceService(
                args.model,
                config=OnnxSessionConfig(),
            )
            try:
                onnx = measure_onnx_performance(
                    service,
                    np.zeros(_parse_shape(args.input_shape), dtype=np.float32),
                    iterations=args.iterations,
                )
            finally:
                service.close()

        report = {
            "pipeline": pipeline.model_dump(by_alias=True),
            "onnx": onnx.model_dump(by_alias=True) if onnx is not None else None,
            "measurement": {
                "imagesAreStreamed": True,
                "maxInflightImages": pipeline.max_inflight_images,
                "cpuOnlyBaseline": True,
                "internetRequired": False,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI returns a stable failure code
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
