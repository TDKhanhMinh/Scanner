"""AS-37 ONNX Runtime lifecycle and error-contract tests."""

from pathlib import Path
from shutil import copyfile

import numpy as np
import onnxruntime as ort
import pytest

from attendance_scanner.onnx_runtime import (
    OnnxInferenceService,
    OnnxRuntimeError,
    OnnxRuntimeErrorCode,
    OnnxSessionConfig,
    benchmark_service,
    resolve_model_path,
)


def _bundled_model() -> Path:
    return Path(ort.__file__).resolve().parent / "datasets" / "mul_1.onnx"


def _copy_test_model(tmp_path: Path) -> Path:
    model_path = tmp_path / "models" / "mul_1.onnx"
    model_path.parent.mkdir(parents=True)
    copyfile(_bundled_model(), model_path)
    return model_path


def test_load_is_lazy_validates_io_and_reuses_one_session(tmp_path: Path):
    model_path = _copy_test_model(tmp_path)
    service = OnnxInferenceService(
        model_path,
        config=OnnxSessionConfig(
            expected_input_name="X",
            expected_input_shape=(3, 2),
            expected_output_names=["Y"],
        ),
    )

    assert service.session_create_count == 0
    first_info = service.load()
    second_info = service.load()
    result = service.infer(np.ones((3, 2), dtype=np.float32))

    assert first_info == second_info
    assert first_info.input_name == "X"
    assert first_info.input_shape == [3, 2]
    assert first_info.output_names == ["Y"]
    assert result.outputs[0].shape == (3, 2)
    assert service.session_create_count == 1


def test_microbenchmark_reports_startup_and_reused_inference(tmp_path: Path):
    service = OnnxInferenceService(_copy_test_model(tmp_path))

    benchmark = benchmark_service(
        service,
        np.ones((3, 2), dtype=np.float32),
        iterations=3,
    )

    assert benchmark.session_startup_ms >= 0.0
    assert benchmark.first_inference_ms >= 0.0
    assert benchmark.repeated_inference_ms["meanMs"] >= 0.0
    assert benchmark.session_create_count == 1
    assert benchmark.iterations == 3


def test_model_path_resolution_supports_relative_search_root(tmp_path: Path):
    model_path = _copy_test_model(tmp_path)

    resolved = resolve_model_path("models/mul_1.onnx", search_roots=[tmp_path])

    assert resolved == model_path.resolve()


def test_missing_and_corrupt_models_return_structured_safe_errors(tmp_path: Path):
    with pytest.raises(OnnxRuntimeError) as missing_error:
        OnnxInferenceService("missing/model.onnx", search_roots=[tmp_path]).load()
    assert missing_error.value.code == OnnxRuntimeErrorCode.MODEL_MISSING
    assert missing_error.value.user_message
    assert missing_error.value.diagnostic

    corrupt = tmp_path / "corrupt.onnx"
    corrupt.write_bytes(b"not-an-onnx-model")
    with pytest.raises(OnnxRuntimeError) as corrupt_error:
        OnnxInferenceService(corrupt).load()
    assert corrupt_error.value.code == OnnxRuntimeErrorCode.MODEL_INVALID
    assert corrupt_error.value.to_dict()["code"] == "model_invalid"


def test_model_io_and_provider_mismatch_are_rejected(tmp_path: Path):
    model_path = _copy_test_model(tmp_path)
    wrong_shape = OnnxInferenceService(
        model_path,
        config=OnnxSessionConfig(expected_input_shape=(1, 2)),
    )
    with pytest.raises(OnnxRuntimeError) as shape_error:
        wrong_shape.load()
    assert shape_error.value.code == OnnxRuntimeErrorCode.MODEL_UNSUPPORTED

    with pytest.raises(ValueError, match="CPUExecutionProvider is mandatory"):
        OnnxSessionConfig(providers=["CUDAExecutionProvider"])


def test_invalid_inference_input_is_structured_and_close_allows_new_lifecycle(tmp_path: Path):
    service = OnnxInferenceService(_copy_test_model(tmp_path))
    with pytest.raises(OnnxRuntimeError) as shape_error:
        service.infer(np.ones((2, 2), dtype=np.float32))
    assert shape_error.value.code == OnnxRuntimeErrorCode.INPUT_INVALID

    with pytest.raises(OnnxRuntimeError) as dtype_error:
        service.infer(np.ones((3, 2), dtype=np.int32))
    assert dtype_error.value.code == OnnxRuntimeErrorCode.INPUT_INVALID

    with pytest.raises(OnnxRuntimeError) as nan_error:
        service.infer(np.full((3, 2), np.nan, dtype=np.float32))
    assert nan_error.value.code == OnnxRuntimeErrorCode.INPUT_INVALID

    service.close()
    service.infer(np.ones((3, 2), dtype=np.float32))
    assert service.session_create_count == 2
