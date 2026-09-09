"""Measure ONNX Runtime session startup and reused inference latency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from attendance_scanner.onnx_runtime import (
    OnnxInferenceService,
    OnnxRuntimeError,
    OnnxSessionConfig,
    benchmark_service,
    runtime_diagnostics,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark ONNX Runtime session startup versus reused inference"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--input-shape",
        required=True,
        help="comma-separated tensor shape, for example 1,3,224,224",
    )
    parser.add_argument("--input-name", default=None)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--intra-op-threads", type=int, default=1)
    parser.add_argument("--inter-op-threads", type=int, default=1)
    args = parser.parse_args(argv)

    try:
        shape = tuple(int(value.strip()) for value in args.input_shape.split(","))
        if not shape or any(value <= 0 for value in shape):
            raise ValueError(
                "--input-shape must contain positive comma-separated integers"
            )
        if args.iterations < 1:
            raise ValueError("--iterations must be at least 1")
        service = OnnxInferenceService(
            args.model,
            config=OnnxSessionConfig(
                expected_input_name=args.input_name,
                intra_op_num_threads=args.intra_op_threads,
                inter_op_num_threads=args.inter_op_threads,
            ),
        )
        benchmark = benchmark_service(
            service,
            np.zeros(shape, dtype=np.float32),
            iterations=args.iterations,
        )
        print(
            json.dumps(
                {
                    "model": service.model_info.model_dump(mode="json")
                    if service.model_info is not None
                    else None,
                    "benchmark": benchmark.to_dict(),
                    "runtime": runtime_diagnostics(),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except OnnxRuntimeError as exc:
        print(json.dumps({"valid": False, "error": exc.to_dict()}, ensure_ascii=False))
        return 1
    except (TypeError, ValueError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
