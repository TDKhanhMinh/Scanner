"""AS-38 generic ONNX document-segmentation adapter tests."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from PIL import Image

from attendance_scanner.detector import DocumentDetectionResult
from attendance_scanner.onnx_runtime import (
    OnnxInferenceService,
    OnnxRuntimeError,
    OnnxRuntimeErrorCode,
    OnnxSessionConfig,
)
from attendance_scanner.pipeline.load import load_image
from attendance_scanner.segmentation import (
    OnnxSegmentationAdapter,
    SegmentationAdapterError,
    SegmentationConfig,
    SegmentationErrorCode,
    SegmentationModelCard,
    decode_segmentation_output,
    preprocess_segmentation_input,
)


def _write_image(path: Path, color: tuple[int, int, int] = (255, 0, 0)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (10, 8), color=color).save(path)
    return path


def _model_card() -> SegmentationModelCard:
    return SegmentationModelCard(
        version="test-segmentation-1",
        license="Test-only model license",
        attribution="Attendance Scanner tests",
    )


def test_preprocess_converts_bgr_to_rgb_and_keeps_transform_metadata(tmp_path: Path):
    image = load_image(_write_image(tmp_path / "red.png"))
    config = SegmentationConfig(
        input_width=10,
        input_height=8,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )

    tensor, transform = preprocess_segmentation_input(image, config)

    assert tensor.shape == (1, 3, 8, 10)
    assert tensor.dtype == np.float32
    assert np.allclose(tensor[0, 0], 1.0)
    assert np.allclose(tensor[0, 1:], 0.0)
    assert (transform.source_width, transform.source_height) == (10, 8)
    assert (transform.input_width, transform.input_height) == (10, 8)


def test_decode_supports_logits_and_probability_outputs():
    logits = np.array(
        [[[[0.0, 0.0], [0.0, 0.0]], [[4.0, -4.0], [4.0, -4.0]]]],
        dtype=np.float32,
    )
    decoded_logits = decode_segmentation_output(
        logits,
        SegmentationConfig(
            output_layout="nchw",
            activation="logits",
            document_class_index=1,
        ),
    )
    probabilities = np.array(
        [[[0.2, 0.8], [0.7, 0.3]], [[0.8, 0.2], [0.3, 0.7]]],
        dtype=np.float32,
    )
    decoded_probability = decode_segmentation_output(
        probabilities,
        SegmentationConfig(
            output_layout="chw",
            activation="probability",
            document_class_index=1,
        ),
    )

    assert decoded_logits.shape == (2, 2)
    assert decoded_logits.dtype == np.float32
    assert float(decoded_logits[0, 0]) > 0.9
    assert float(decoded_logits[0, 1]) < 0.1
    assert np.array_equal(decoded_probability, probabilities[1])


def test_decode_rejects_ambiguous_layout_and_bad_probability_range():
    with pytest.raises(SegmentationAdapterError) as layout_error:
        decode_segmentation_output(
            np.zeros((1, 3, 4, 5), dtype=np.float32),
            SegmentationConfig(),
        )
    assert layout_error.value.code == SegmentationErrorCode.OUTPUT_INVALID

    with pytest.raises(SegmentationAdapterError) as probability_error:
        decode_segmentation_output(
            np.array([[[[2.0]], [[-1.0]]]], dtype=np.float32),
            SegmentationConfig(
                output_layout="nchw",
                activation="probability",
            ),
        )
    assert probability_error.value.code == SegmentationErrorCode.OUTPUT_INVALID

    with pytest.raises(SegmentationAdapterError) as one_channel_error:
        decode_segmentation_output(
            np.array([[[[2.0, -1.0]]]], dtype=np.float32),
            SegmentationConfig(
                output_layout="nchw",
                activation="probability",
                document_class_index=0,
            ),
        )
    assert one_channel_error.value.code == SegmentationErrorCode.OUTPUT_INVALID


def test_sigmoid_onnx_adapter_returns_original_size_internal_masks_and_summary(tmp_path: Path):
    model_path = Path(ort.__file__).resolve().parent / "datasets" / "sigmoid.onnx"
    service = OnnxInferenceService(
        model_path,
        config=OnnxSessionConfig(
            expected_input_name="x",
            expected_input_shape=(3, 4, 5),
            expected_output_names=["y"],
        ),
    )
    debug_dir = tmp_path / "debug-masks"
    adapter = OnnxSegmentationAdapter(
        service,
        model_card=_model_card(),
        config=SegmentationConfig(
            input_width=5,
            input_height=4,
            add_batch_dimension=False,
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            output_layout="chw",
            activation="sigmoid",
            debug_artifact_dir=debug_dir,
        ),
    )
    image = load_image(_write_image(tmp_path / "input.png"))

    output = adapter.segment(image)

    assert output.probability_mask.shape == (8, 10)
    assert output.probability_mask.dtype == np.float32
    assert float(output.probability_mask.min()) >= 0.0
    assert float(output.probability_mask.max()) <= 1.0
    assert output.binary_mask.shape == (8, 10)
    assert output.binary_mask.dtype == np.bool_
    assert output.transform.scale_x == 0.5
    assert output.transform.scale_y == 0.5
    assert output.detection.detected is False
    assert output.detection.failure_code == "mask_only"
    assert output.detection.evidence.component_count is not None
    assert output.detection.model_version == "test-segmentation-1"
    assert len(output.debug_artifacts) == 3
    assert all(Path(path).is_file() for path in output.debug_artifacts)
    assert service.session_create_count == 1
    assert "probabilityMask" not in output.detection.diagnostics_extension()["documentDetection"]

    first_duplicate = load_image(_write_image(tmp_path / "sheet.jpg"))
    second_duplicate = load_image(_write_image(tmp_path / "sheet.png"))
    first_debug = adapter.segment(first_duplicate).debug_artifacts
    second_debug = adapter.segment(second_duplicate).debug_artifacts
    assert set(first_debug).isdisjoint(second_debug)


def test_segmentation_failure_maps_to_document_detector_contract(tmp_path: Path):
    class BrokenService:
        model_info = None

        def infer(self, _input: np.ndarray):  # type: ignore[no-untyped-def]
            raise OnnxRuntimeError(
                OnnxRuntimeErrorCode.MODEL_MISSING,
                "safe",
                "developer diagnostic",
            )

    adapter = OnnxSegmentationAdapter(BrokenService(), model_card=_model_card())  # type: ignore[arg-type]
    result = adapter.detect(load_image(_write_image(tmp_path / "input.png")))

    assert isinstance(result, DocumentDetectionResult)
    assert result.detected is False
    assert result.failure_code == "inference_failed"
    assert result.metadata["adapterErrorCode"] == "inference_failed"
