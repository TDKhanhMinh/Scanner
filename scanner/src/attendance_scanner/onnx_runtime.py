"""Model-independent ONNX Runtime lifecycle and inference service."""

from __future__ import annotations

import hashlib
import os
import platform
import re
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class OnnxRuntimeErrorCode(str, Enum):
    """Stable infrastructure failure categories for user and developer surfaces."""

    MODEL_MISSING = "model_missing"
    MODEL_INVALID = "model_invalid"
    MODEL_UNSUPPORTED = "model_unsupported"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    INPUT_INVALID = "input_invalid"
    INFERENCE_FAILED = "inference_failed"
    RUNTIME_UNAVAILABLE = "runtime_unavailable"


class OnnxRuntimeError(Exception):
    """Structured ONNX failure with a safe user message and developer diagnostic."""

    def __init__(
        self,
        code: OnnxRuntimeErrorCode,
        user_message: str,
        diagnostic: str,
    ) -> None:
        super().__init__(diagnostic)
        self.code = code
        self.user_message = user_message
        self.diagnostic = diagnostic

    def to_dict(self) -> Dict[str, str]:
        """Return a JSON-safe error payload for non-JSONL diagnostic callers."""
        return {
            "code": self.code.value,
            "userMessage": self.user_message,
            "developerDiagnostic": self.diagnostic,
        }


class OnnxSessionConfig(BaseModel):
    """Deterministic ONNX session and optional model I/O expectations."""

    model_config = ConfigDict(extra="forbid")

    providers: List[str] = Field(default_factory=lambda: ["CPUExecutionProvider"], min_length=1)
    intra_op_num_threads: int = Field(default=1, ge=1, le=64)
    inter_op_num_threads: int = Field(default=1, ge=1, le=64)
    execution_mode: str = "sequential"
    graph_optimization_level: str = "all"
    expected_input_name: Optional[str] = None
    expected_input_shape: Optional[Tuple[Optional[int], ...]] = None
    expected_output_names: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_session_settings(self) -> "OnnxSessionConfig":
        if "CPUExecutionProvider" not in self.providers:
            raise ValueError("CPUExecutionProvider is mandatory for the baseline service")
        if len(set(self.providers)) != len(self.providers):
            raise ValueError("providers must not contain duplicates")
        if self.execution_mode not in {"sequential", "parallel"}:
            raise ValueError("execution_mode must be sequential or parallel")
        if self.graph_optimization_level not in {"disable", "basic", "extended", "all"}:
            raise ValueError("graph_optimization_level must be disable, basic, extended, or all")
        if self.expected_input_shape is not None and any(
            dimension is not None and dimension <= 0 for dimension in self.expected_input_shape
        ):
            raise ValueError("expected_input_shape dimensions must be positive or null")
        if len(set(self.expected_output_names)) != len(self.expected_output_names):
            raise ValueError("expected_output_names must not contain duplicates")
        return self


class OnnxModelInfo(BaseModel):
    """Validated model/session metadata safe to include in reports."""

    model_config = ConfigDict(extra="forbid")

    model_path: str
    model_sha256: str
    input_name: str
    input_shape: List[Any]
    input_type: str
    output_names: List[str]
    output_shapes: Dict[str, List[Any]]
    output_types: Dict[str, str]
    providers: List[str]
    runtime_version: str


@dataclass(frozen=True)
class OnnxInferenceResult:
    """Outputs and timing for one inference call."""

    outputs: Sequence[Any]
    input_name: str
    output_names: Sequence[str]
    duration_ms: float


@dataclass(frozen=True)
class OnnxMicroBenchmark:
    """Startup and steady-state measurements for one reused service."""

    session_startup_ms: float
    first_inference_ms: float
    repeated_inference_ms: Dict[str, float]
    iterations: int
    session_create_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sessionStartupMs": self.session_startup_ms,
            "firstInferenceMs": self.first_inference_ms,
            "repeatedInferenceMs": self.repeated_inference_ms,
            "iterations": self.iterations,
            "sessionCreateCount": self.session_create_count,
        }


def resolve_model_path(
    model_path: Union[str, Path],
    *,
    search_roots: Optional[Sequence[Union[str, Path]]] = None,
) -> Path:
    """Resolve a model in dev, PyInstaller extraction, or beside the executable."""
    requested = Path(model_path)
    if requested.is_absolute():
        candidates = [requested]
    else:
        roots = [Path(root) for root in search_roots or ()]
        package_root = Path(__file__).resolve().parents[2]
        candidates = [
            *(root / requested for root in roots),
            Path.cwd() / requested,
            package_root / requested,
        ]
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass) / requested)
        if getattr(sys, "executable", None):
            candidates.append(Path(sys.executable).resolve().parent / requested)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise OnnxRuntimeError(
        OnnxRuntimeErrorCode.MODEL_MISSING,
        "Không tìm thấy model AI được cấu hình.",
        f"ONNX model '{model_path}' was not found. Searched: {searched}",
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise OnnxRuntimeError(
            OnnxRuntimeErrorCode.MODEL_INVALID,
            "Không thể đọc file model AI.",
            f"Unable to hash ONNX model '{path}': {exc}",
        ) from exc
    return digest.hexdigest()


def _shape_matches(actual: Sequence[Any], expected: Sequence[Optional[int]]) -> bool:
    if len(actual) != len(expected):
        return False
    return all(
        expected_dimension is None
        or (isinstance(actual_dimension, int) and actual_dimension == expected_dimension)
        for actual_dimension, expected_dimension in zip(actual, expected, strict=True)
    )


def _shape_list(value: Sequence[Any]) -> List[Any]:
    return list(value)


def _expected_numpy_dtype(input_type: str) -> Optional[np.dtype[Any]]:
    match = re.fullmatch(r"tensor\(([^)]+)\)", input_type)
    if match is None:
        return None
    type_map = {
        "bool": np.dtype("bool"),
        "float": np.dtype("float32"),
        "float16": np.dtype("float16"),
        "double": np.dtype("float64"),
        "int8": np.dtype("int8"),
        "int16": np.dtype("int16"),
        "int32": np.dtype("int32"),
        "int64": np.dtype("int64"),
        "uint8": np.dtype("uint8"),
        "uint16": np.dtype("uint16"),
        "uint32": np.dtype("uint32"),
        "uint64": np.dtype("uint64"),
    }
    return type_map.get(match.group(1))


def _import_onnxruntime() -> Any:
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]
    except ImportError as exc:
        raise OnnxRuntimeError(
            OnnxRuntimeErrorCode.RUNTIME_UNAVAILABLE,
            "ONNX Runtime chưa được cài đặt trong sidecar.",
            f"Unable to import onnxruntime: {exc}",
        ) from exc
    return ort


class OnnxInferenceService:
    """Lazy, reusable, thread-safe ONNX Runtime session for one model path."""

    def __init__(
        self,
        model_path: Union[str, Path],
        *,
        config: Optional[OnnxSessionConfig] = None,
        search_roots: Optional[Sequence[Union[str, Path]]] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.config = config or OnnxSessionConfig()
        self.search_roots = tuple(Path(root) for root in search_roots or ())
        self._session: Any = None
        self._model_info: Optional[OnnxModelInfo] = None
        self._session_create_count = 0
        self._lock = threading.RLock()

    @property
    def session_create_count(self) -> int:
        """Number of successful session constructions in this lifecycle."""
        return self._session_create_count

    @property
    def model_info(self) -> Optional[OnnxModelInfo]:
        """Validated model metadata, or None before lazy load."""
        return self._model_info

    def load(self) -> OnnxModelInfo:
        """Create and validate the session once, then reuse it."""
        with self._lock:
            if self._session is not None and self._model_info is not None:
                return self._model_info

            path = resolve_model_path(self.model_path, search_roots=self.search_roots)
            model_hash = _hash_file(path)
            ort = _import_onnxruntime()
            available = set(ort.get_available_providers())
            unavailable = [
                provider for provider in self.config.providers if provider not in available
            ]
            if unavailable:
                raise OnnxRuntimeError(
                    OnnxRuntimeErrorCode.PROVIDER_UNAVAILABLE,
                    "Execution provider của model không khả dụng trên máy này.",
                    f"Requested ONNX providers {unavailable} are unavailable; installed: "
                    f"{sorted(available)}",
                )

            options = ort.SessionOptions()
            options.intra_op_num_threads = self.config.intra_op_num_threads
            options.inter_op_num_threads = self.config.inter_op_num_threads
            options.execution_mode = (
                ort.ExecutionMode.ORT_SEQUENTIAL
                if self.config.execution_mode == "sequential"
                else ort.ExecutionMode.ORT_PARALLEL
            )
            optimization_levels = {
                "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
                "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            }
            options.graph_optimization_level = optimization_levels[
                self.config.graph_optimization_level
            ]
            try:
                session = ort.InferenceSession(
                    str(path),
                    sess_options=options,
                    providers=list(self.config.providers),
                )
                info = self._validate_session(session, path, model_hash, ort.__version__)
            except OnnxRuntimeError:
                raise
            except Exception as exc:
                raise OnnxRuntimeError(
                    OnnxRuntimeErrorCode.MODEL_INVALID,
                    "Model AI không hợp lệ hoặc không tương thích với ONNX Runtime.",
                    f"Unable to create/inspect ONNX session for '{path}': "
                    f"{type(exc).__name__}: {exc}",
                ) from exc
            self._session = session
            self._model_info = info
            self._session_create_count += 1
            return info

    def _validate_session(
        self,
        session: Any,
        path: Path,
        model_hash: str,
        runtime_version: str,
    ) -> OnnxModelInfo:
        inputs = list(session.get_inputs())
        outputs = list(session.get_outputs())
        if not inputs:
            raise OnnxRuntimeError(
                OnnxRuntimeErrorCode.MODEL_UNSUPPORTED,
                "Model AI không có input hợp lệ.",
                f"ONNX model '{path}' exposes no inputs",
            )
        if not outputs:
            raise OnnxRuntimeError(
                OnnxRuntimeErrorCode.MODEL_UNSUPPORTED,
                "Model AI không có output hợp lệ.",
                f"ONNX model '{path}' exposes no outputs",
            )
        input_by_name = {item.name: item for item in inputs}
        input_name = self.config.expected_input_name or inputs[0].name
        if input_name not in input_by_name:
            raise OnnxRuntimeError(
                OnnxRuntimeErrorCode.MODEL_UNSUPPORTED,
                "Tên input của model AI không tương thích.",
                f"Expected input '{input_name}' not found; available: {sorted(input_by_name)}",
            )
        input_meta = input_by_name[input_name]
        if self.config.expected_input_shape is not None and not _shape_matches(
            input_meta.shape, self.config.expected_input_shape
        ):
            raise OnnxRuntimeError(
                OnnxRuntimeErrorCode.MODEL_UNSUPPORTED,
                "Shape input của model AI không tương thích.",
                f"Input '{input_name}' shape {input_meta.shape} does not match "
                f"{self.config.expected_input_shape}",
            )
        output_names = [item.name for item in outputs]
        missing_outputs = [
            name for name in self.config.expected_output_names if name not in output_names
        ]
        if missing_outputs:
            raise OnnxRuntimeError(
                OnnxRuntimeErrorCode.MODEL_UNSUPPORTED,
                "Output của model AI không tương thích.",
                f"Expected outputs {missing_outputs} not found; available: {output_names}",
            )
        return OnnxModelInfo(
            model_path=str(path),
            model_sha256=model_hash,
            input_name=input_name,
            input_shape=_shape_list(input_meta.shape),
            input_type=str(input_meta.type),
            output_names=output_names,
            output_shapes={item.name: _shape_list(item.shape) for item in outputs},
            output_types={item.name: str(item.type) for item in outputs},
            providers=list(session.get_providers()),
            runtime_version=runtime_version,
        )

    def infer(
        self,
        input_data: Union[np.ndarray, Mapping[str, np.ndarray]],
    ) -> OnnxInferenceResult:
        """Run one inference against the reused session with structured failures."""
        with self._lock:
            info = self.load()
            if isinstance(input_data, Mapping):
                feed = dict(input_data)
                if info.input_name not in feed:
                    raise OnnxRuntimeError(
                        OnnxRuntimeErrorCode.INPUT_INVALID,
                        "Dữ liệu input không chứa đúng tên input của model.",
                        f"Missing input '{info.input_name}'; provided: {sorted(feed)}",
                    )
            elif isinstance(input_data, np.ndarray):
                feed = {info.input_name: input_data}
            else:
                raise OnnxRuntimeError(
                    OnnxRuntimeErrorCode.INPUT_INVALID,
                    "Dữ liệu input phải là NumPy array hoặc mapping theo tên input.",
                    f"Unsupported input type: {type(input_data).__name__}",
                )

            normalized_feed: Dict[str, np.ndarray] = {}
            for name, value in feed.items():
                if not isinstance(value, np.ndarray):
                    raise OnnxRuntimeError(
                        OnnxRuntimeErrorCode.INPUT_INVALID,
                        "Dữ liệu input phải là NumPy array.",
                        f"Input '{name}' has type {type(value).__name__}",
                    )
                if name == info.input_name:
                    expected_dtype = _expected_numpy_dtype(info.input_type)
                    if expected_dtype is None or value.dtype != expected_dtype:
                        expected_label = (
                            str(expected_dtype) if expected_dtype is not None else info.input_type
                        )
                        raise OnnxRuntimeError(
                            OnnxRuntimeErrorCode.INPUT_INVALID,
                            "Kiểu dữ liệu input không khớp với model AI.",
                            f"Input '{name}' dtype {value.dtype} does not match {expected_label}",
                        )
                if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                    raise OnnxRuntimeError(
                        OnnxRuntimeErrorCode.INPUT_INVALID,
                        "Dữ liệu input chứa giá trị không hợp lệ.",
                        f"Input '{name}' contains NaN or infinity",
                    )
                if name == info.input_name and not _shape_matches(
                    value.shape,
                    [
                        dimension if isinstance(dimension, int) else None
                        for dimension in info.input_shape
                    ],
                ):
                    raise OnnxRuntimeError(
                        OnnxRuntimeErrorCode.INPUT_INVALID,
                        "Shape input không khớp với model AI.",
                        f"Input '{name}' shape {value.shape} does not match {info.input_shape}",
                    )
                normalized_feed[name] = np.ascontiguousarray(value)

            started_at = time.perf_counter()
            try:
                outputs = self._session.run(info.output_names, normalized_feed)
            except Exception as exc:
                raise OnnxRuntimeError(
                    OnnxRuntimeErrorCode.INFERENCE_FAILED,
                    "Model AI không thể xử lý ảnh này.",
                    f"ONNX inference failed for '{info.model_path}': {type(exc).__name__}: {exc}",
                ) from exc
            return OnnxInferenceResult(
                outputs=outputs,
                input_name=info.input_name,
                output_names=info.output_names,
                duration_ms=(time.perf_counter() - started_at) * 1000.0,
            )

    def close(self) -> None:
        """Release the session at an explicit batch lifecycle boundary."""
        with self._lock:
            self._session = None
            self._model_info = None


def benchmark_service(
    service: OnnxInferenceService,
    input_data: Union[np.ndarray, Mapping[str, np.ndarray]],
    *,
    iterations: int = 10,
) -> OnnxMicroBenchmark:
    """Measure one session startup, first inference, and reused steady state."""
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    started_at = time.perf_counter()
    service.load()
    startup_ms = (time.perf_counter() - started_at) * 1000.0
    first = service.infer(input_data)
    durations = [service.infer(input_data).duration_ms for _ in range(iterations)]
    values = np.asarray(durations, dtype=np.float64)
    return OnnxMicroBenchmark(
        session_startup_ms=startup_ms,
        first_inference_ms=first.duration_ms,
        repeated_inference_ms={
            "meanMs": float(np.mean(values)),
            "p50Ms": float(np.percentile(values, 50)),
            "p95Ms": float(np.percentile(values, 95)),
            "p99Ms": float(np.percentile(values, 99)),
        },
        iterations=iterations,
        session_create_count=service.session_create_count,
    )


def runtime_diagnostics() -> Dict[str, Any]:
    """Return environment facts for the AS-35/AS-58 benchmark report."""
    ort = _import_onnxruntime()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpuCount": os.cpu_count(),
        "onnxruntime": ort.__version__,
        "availableProviders": sorted(ort.get_available_providers()),
    }


__all__ = [
    "OnnxInferenceService",
    "OnnxInferenceResult",
    "OnnxMicroBenchmark",
    "OnnxModelInfo",
    "OnnxRuntimeError",
    "OnnxRuntimeErrorCode",
    "OnnxSessionConfig",
    "benchmark_service",
    "resolve_model_path",
    "runtime_diagnostics",
]
