"""Generic ONNX semantic document-segmentation adapter.

This module binds model I/O and mask decoding only. It deliberately does not fit
corners; that responsibility belongs to the following postprocessing task.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from attendance_scanner.detector import (
    DetectionTiming,
    DetectorEvidence,
    DocumentDetectionResult,
)
from attendance_scanner.mask_postprocess import (
    MaskPostprocessConfig,
    MaskPostprocessResult,
    postprocess_document_mask,
)
from attendance_scanner.onnx_runtime import (
    OnnxInferenceService,
    OnnxRuntimeError,
)
from attendance_scanner.pipeline.load import LoadedImage


class SegmentationErrorCode(str, Enum):
    """Stable adapter-level failures."""

    PREPROCESSING_FAILED = "preprocessing_failed"
    OUTPUT_INVALID = "output_invalid"
    INFERENCE_FAILED = "inference_failed"


class SegmentationAdapterError(Exception):
    """Structured mask adapter error with user-safe and developer messages."""

    def __init__(self, code: SegmentationErrorCode, user_message: str, diagnostic: str) -> None:
        super().__init__(diagnostic)
        self.code = code
        self.user_message = user_message
        self.diagnostic = diagnostic

    def to_dict(self) -> Dict[str, str]:
        return {
            "code": self.code.value,
            "userMessage": self.user_message,
            "developerDiagnostic": self.diagnostic,
        }


class SegmentationModelCard(BaseModel):
    """Required provenance metadata for a segmentation artifact."""

    model_config = ConfigDict(extra="forbid")

    version: str = Field(min_length=1)
    license: str = Field(min_length=1)
    attribution: str = Field(min_length=1)
    url: Optional[str] = None


class SegmentationConfig(BaseModel):
    """Model-independent preprocessing and output decoding settings."""

    model_config = ConfigDict(extra="forbid")

    input_width: int = Field(default=224, gt=0, le=4096)
    input_height: int = Field(default=224, gt=0, le=4096)
    resize_mode: Literal["stretch", "letterbox"] = "stretch"
    letterbox_fill: Tuple[int, int, int] = (0, 0, 0)
    input_layout: Literal["nchw", "nhwc"] = "nchw"
    add_batch_dimension: bool = True
    channel_order: Literal["rgb"] = "rgb"
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    output_index: int = Field(default=0, ge=0)
    output_layout: Literal["auto", "nchw", "nhwc", "chw", "hwc", "hw"] = "auto"
    activation: Literal["auto", "logits", "probability", "sigmoid", "softmax"] = "auto"
    document_class_index: int = Field(default=1, ge=0)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    postprocess: MaskPostprocessConfig = Field(default_factory=MaskPostprocessConfig)
    debug_artifact_dir: Optional[Path] = None

    @model_validator(mode="after")
    def validate_normalization(self) -> "SegmentationConfig":
        if any(not math.isfinite(value) for value in (*self.mean, *self.std)):
            raise ValueError("mean/std values must be finite")
        if any(abs(value) < 1e-9 for value in self.std):
            raise ValueError("std values must be non-zero")
        if any(value < 0 or value > 255 for value in self.letterbox_fill):
            raise ValueError("letterbox_fill values must be in [0,255]")
        return self


@dataclass(frozen=True)
class SegmentationTransform:
    """Input/output geometry metadata for later coordinate-aware postprocessing."""

    source_width: int
    source_height: int
    input_width: int
    input_height: int
    scale_x: float
    scale_y: float
    resize_mode: Literal["stretch", "letterbox"] = "stretch"
    pad_x: int = 0
    pad_y: int = 0
    resized_width: int = 0
    resized_height: int = 0

    def map_points_to_source(
        self, points: Sequence[Tuple[float, float]]
    ) -> Tuple[Tuple[float, float], ...]:
        """Map model-input pixel points back to EXIF-normalized source pixels."""
        return tuple(
            ((x - self.pad_x) / self.scale_x, (y - self.pad_y) / self.scale_y) for x, y in points
        )

    def mask_to_source(self, probability_mask: np.ndarray) -> np.ndarray:
        """Remove letterbox padding and map a model mask to source dimensions."""
        if self.resize_mode == "stretch":
            crop = probability_mask
        else:
            mask_height, mask_width = probability_mask.shape[:2]
            x0 = round(self.pad_x / self.input_width * mask_width)
            y0 = round(self.pad_y / self.input_height * mask_height)
            x1 = round((self.pad_x + self.resized_width) / self.input_width * mask_width)
            y1 = round((self.pad_y + self.resized_height) / self.input_height * mask_height)
            x0 = max(0, min(x0, mask_width - 1))
            y0 = max(0, min(y0, mask_height - 1))
            x1 = max(x0 + 1, min(x1, mask_width))
            y1 = max(y0 + 1, min(y1, mask_height))
            crop = probability_mask[y0:y1, x0:x1]
        return cv2.resize(
            crop,
            (self.source_width, self.source_height),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32, copy=False)


@dataclass(frozen=True)
class SegmentationOutput:
    """Internal probability/binary masks plus the generic detector summary."""

    probability_mask: np.ndarray
    binary_mask: np.ndarray
    transform: SegmentationTransform
    postprocess: MaskPostprocessResult
    detection: DocumentDetectionResult
    debug_artifacts: Tuple[str, ...] = ()


def preprocess_segmentation_input(
    image: LoadedImage,
    config: SegmentationConfig,
) -> Tuple[np.ndarray, SegmentationTransform]:
    """Convert normalized BGR pixels to the configured model tensor."""
    try:
        rgb = cv2.cvtColor(image.image, cv2.COLOR_BGR2RGB)
        if config.resize_mode == "stretch":
            resized = cv2.resize(
                rgb,
                (config.input_width, config.input_height),
                interpolation=cv2.INTER_AREA,
            )
            pad_x = pad_y = 0
            resized_width = config.input_width
            resized_height = config.input_height
        else:
            scale = min(config.input_width / image.width, config.input_height / image.height)
            resized_width = max(1, round(image.width * scale))
            resized_height = max(1, round(image.height * scale))
            resized_content = cv2.resize(
                rgb,
                (resized_width, resized_height),
                interpolation=cv2.INTER_AREA,
            )
            resized = np.full(
                (config.input_height, config.input_width, 3),
                config.letterbox_fill,
                dtype=np.uint8,
            )
            pad_x = (config.input_width - resized_width) // 2
            pad_y = (config.input_height - resized_height) // 2
            resized[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized_content
        tensor: np.ndarray = resized.astype(np.float32) / 255.0
        tensor = (tensor - np.asarray(config.mean, dtype=np.float32)) / np.asarray(
            config.std, dtype=np.float32
        )
        if config.input_layout == "nchw":
            tensor = np.transpose(tensor, (2, 0, 1))
        if config.add_batch_dimension:
            tensor = np.expand_dims(tensor, axis=0)
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)
    except (cv2.error, TypeError, ValueError) as exc:
        raise SegmentationAdapterError(
            SegmentationErrorCode.PREPROCESSING_FAILED,
            "Không thể chuẩn bị ảnh cho model segmentation.",
            f"Segmentation preprocessing failed: {type(exc).__name__}: {exc}",
        ) from exc
    return tensor, SegmentationTransform(
        source_width=image.width,
        source_height=image.height,
        input_width=config.input_width,
        input_height=config.input_height,
        scale_x=(resized_width / image.width),
        scale_y=(resized_height / image.height),
        resize_mode=config.resize_mode,
        pad_x=pad_x,
        pad_y=pad_y,
        resized_width=resized_width,
        resized_height=resized_height,
    )


def _spatial_channels(
    output: Any,
    config: SegmentationConfig,
) -> np.ndarray:
    array = np.asarray(output)
    if array.size == 0 or array.ndim not in {2, 3, 4}:
        raise SegmentationAdapterError(
            SegmentationErrorCode.OUTPUT_INVALID,
            "Output model segmentation không đúng shape.",
            f"Expected 2D/3D/4D output, received shape {array.shape}",
        )
    if not np.issubdtype(array.dtype, np.number):
        raise SegmentationAdapterError(
            SegmentationErrorCode.OUTPUT_INVALID,
            "Output model segmentation không phải tensor số.",
            f"Expected numeric output, received dtype {array.dtype}",
        )
    if not np.isfinite(array).all():
        raise SegmentationAdapterError(
            SegmentationErrorCode.OUTPUT_INVALID,
            "Output model segmentation chứa giá trị không hợp lệ.",
            "Segmentation output contains NaN or infinity",
        )

    layout = config.output_layout
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise SegmentationAdapterError(
                SegmentationErrorCode.OUTPUT_INVALID,
                "Output model segmentation phải có batch size bằng 1.",
                f"Expected batch dimension 1, received {array.shape}",
            )
        body = array[0]
        if layout == "auto":
            channels_first = body.shape[0] <= 8 and body.shape[-1] > 8
            channels_last = body.shape[-1] <= 8 and body.shape[0] > 8
            if channels_first:
                layout = "nchw"
            elif channels_last:
                layout = "nhwc"
            else:
                raise SegmentationAdapterError(
                    SegmentationErrorCode.OUTPUT_INVALID,
                    "Không xác định được layout output segmentation.",
                    "Ambiguous auto layout for output shape "
                    f"{array.shape}; configure output_layout",
                )
        if layout == "nchw":
            return body
        if layout == "nhwc":
            return np.transpose(body, (2, 0, 1))
        raise SegmentationAdapterError(
            SegmentationErrorCode.OUTPUT_INVALID,
            "Layout output segmentation không khớp tensor 4D.",
            f"Output layout {layout!r} is invalid for shape {array.shape}",
        )

    if array.ndim == 3:
        if layout == "auto":
            channels_first = array.shape[0] <= 8 and array.shape[-1] > 8
            channels_last = array.shape[-1] <= 8 and array.shape[0] > 8
            if channels_first:
                layout = "chw"
            elif channels_last:
                layout = "hwc"
            else:
                raise SegmentationAdapterError(
                    SegmentationErrorCode.OUTPUT_INVALID,
                    "Không xác định được layout output segmentation.",
                    "Ambiguous auto layout for output shape "
                    f"{array.shape}; configure output_layout",
                )
        if layout in {"nchw", "chw"}:
            return array
        if layout in {"nhwc", "hwc"}:
            return np.transpose(array, (2, 0, 1))
        raise SegmentationAdapterError(
            SegmentationErrorCode.OUTPUT_INVALID,
            "Layout output segmentation không khớp tensor 3D.",
            f"Output layout {layout!r} is invalid for shape {array.shape}",
        )

    if layout not in {"auto", "hw"}:
        raise SegmentationAdapterError(
            SegmentationErrorCode.OUTPUT_INVALID,
            "Layout output segmentation không khớp tensor 2D.",
            f"Output layout {layout!r} is invalid for shape {array.shape}",
        )
    return np.expand_dims(array, axis=0)


def decode_segmentation_output(
    output: Any,
    config: SegmentationConfig,
) -> np.ndarray:
    """Decode logits/probabilities into a float32 document probability mask."""
    channels = _spatial_channels(output, config).astype(np.float32, copy=False)
    if channels.ndim != 3 or channels.shape[0] < 1:
        raise SegmentationAdapterError(
            SegmentationErrorCode.OUTPUT_INVALID,
            "Output model segmentation không có channel hợp lệ.",
            f"Decoded channel shape {channels.shape}",
        )
    if config.document_class_index >= channels.shape[0]:
        raise SegmentationAdapterError(
            SegmentationErrorCode.OUTPUT_INVALID,
            "Document class index không tồn tại trong output model.",
            f"Class index {config.document_class_index} for {channels.shape[0]} channels",
        )

    probability_channels: np.ndarray
    if channels.shape[0] == 1:
        values = channels[0]
        if config.activation == "probability":
            if float(values.min()) < 0.0 or float(values.max()) > 1.0:
                raise SegmentationAdapterError(
                    SegmentationErrorCode.OUTPUT_INVALID,
                    "Output probability segmentation phải nằm trong [0,1].",
                    "Configured one-channel probability output contains values outside [0,1]",
                )
            probability = values
        elif (
            config.activation == "auto" and 0.0 <= float(values.min()) <= float(values.max()) <= 1.0
        ):
            probability = values
        else:
            probability = 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))
    else:
        if config.activation == "probability":
            if float(channels.min()) < 0.0 or float(channels.max()) > 1.0:
                raise SegmentationAdapterError(
                    SegmentationErrorCode.OUTPUT_INVALID,
                    "Output probability segmentation phải nằm trong [0,1].",
                    "Configured probability output contains values outside [0,1]",
                )
            probability_channels = channels
        elif config.activation == "auto":
            channel_sums = np.sum(channels, axis=0)
            is_probability = (
                float(channels.min()) >= 0.0
                and float(channels.max()) <= 1.0
                and bool(np.allclose(channel_sums, 1.0, atol=1e-3))
            )
            if is_probability:
                probability_channels = channels
            else:
                shifted = channels - np.max(channels, axis=0, keepdims=True)
                exponent = np.exp(np.clip(shifted, -60.0, 60.0))
                probability_channels = exponent / np.sum(exponent, axis=0, keepdims=True)
        elif config.activation in {"softmax", "logits"}:
            shifted = channels - np.max(channels, axis=0, keepdims=True)
            exponent = np.exp(np.clip(shifted, -60.0, 60.0))
            probability_channels = exponent / np.sum(exponent, axis=0, keepdims=True)
        elif config.activation == "sigmoid":
            probability_channels = 1.0 / (1.0 + np.exp(-np.clip(channels, -60.0, 60.0)))
        else:
            raise SegmentationAdapterError(
                SegmentationErrorCode.OUTPUT_INVALID,
                "Activation output segmentation không được hỗ trợ.",
                f"Unsupported activation {config.activation!r}",
            )
        probability = probability_channels[config.document_class_index]
    return np.clip(probability, 0.0, 1.0).astype(np.float32, copy=False)


def _write_debug_artifacts(
    image: LoadedImage,
    probability_mask: np.ndarray,
    binary_mask: np.ndarray,
    directory: Optional[Path],
) -> Tuple[str, ...]:
    if directory is None:
        return ()
    directory.mkdir(parents=True, exist_ok=True)
    stem = image.path.stem if image.path.stem not in {"", "<memory>"} else "image"
    identity = str(image.path.resolve() if image.path.exists() else image.path).encode("utf-8")
    suffix = hashlib.sha256(identity).hexdigest()[:10]
    probability_path = directory / f"{stem}_{suffix}.probability.npy"
    normalized_path = directory / f"{stem}_{suffix}.probability.png"
    binary_path = directory / f"{stem}_{suffix}.binary.png"
    np.save(probability_path, probability_mask)
    if not cv2.imwrite(str(normalized_path), np.round(probability_mask * 255.0).astype(np.uint8)):
        raise SegmentationAdapterError(
            SegmentationErrorCode.PREPROCESSING_FAILED,
            "Không thể ghi debug mask segmentation.",
            f"Unable to write debug probability mask: {normalized_path}",
        )
    if not cv2.imwrite(str(binary_path), binary_mask.astype(np.uint8) * 255):
        raise SegmentationAdapterError(
            SegmentationErrorCode.PREPROCESSING_FAILED,
            "Không thể ghi debug mask segmentation.",
            f"Unable to write debug binary mask: {binary_path}",
        )
    return tuple(str(path) for path in (probability_path, normalized_path, binary_path))


class OnnxSegmentationAdapter:
    """Generic segmentation provider backed by the reusable ONNX service."""

    detector_version = "onnx_segmentation"
    pipeline_version = "segmentation-v1"

    def __init__(
        self,
        service: OnnxInferenceService,
        *,
        model_card: SegmentationModelCard,
        config: Optional[SegmentationConfig] = None,
    ) -> None:
        self.service = service
        self.model_card = model_card
        self.config = config or SegmentationConfig()

    def segment(self, image: LoadedImage) -> SegmentationOutput:
        """Return internal masks and a mask-only generic detector summary."""
        total_started_at = time.perf_counter()
        preprocess_started_at = time.perf_counter()
        tensor, transform = preprocess_segmentation_input(image, self.config)
        preprocess_ms = (time.perf_counter() - preprocess_started_at) * 1000.0
        try:
            inference = self.service.infer(tensor)
        except OnnxRuntimeError as exc:
            raise SegmentationAdapterError(
                SegmentationErrorCode.INFERENCE_FAILED,
                "Model segmentation không thể xử lý ảnh này.",
                exc.diagnostic,
            ) from exc
        postprocess_started_at = time.perf_counter()
        if self.config.output_index >= len(inference.outputs):
            raise SegmentationAdapterError(
                SegmentationErrorCode.OUTPUT_INVALID,
                "Output index segmentation không tồn tại.",
                f"Output index {self.config.output_index} for {len(inference.outputs)} outputs",
            )
        probability_small = decode_segmentation_output(
            inference.outputs[self.config.output_index], self.config
        )
        probability_mask = transform.mask_to_source(probability_small)
        postprocess = postprocess_document_mask(
            probability_mask,
            self.config.postprocess.model_copy(update={"threshold": self.config.threshold}),
        )
        binary_mask = postprocess.selected_mask
        component_count = postprocess.component_count
        mask_area_ratio = float(binary_mask.mean())
        mask_confidence = float(probability_mask[binary_mask].mean()) if binary_mask.any() else 0.0
        mask_postprocess_ms = (time.perf_counter() - postprocess_started_at) * 1000.0
        detection = DocumentDetectionResult(
            detected=False,
            detector_version=self.detector_version,
            model_version=self.model_card.version,
            pipeline_version=self.pipeline_version,
            candidate_corners=[],
            evidence=DetectorEvidence(
                mask_confidence=mask_confidence,
                mask_area_ratio=mask_area_ratio,
                component_count=component_count,
                mask_to_quad_iou=None,
                model_specific={
                    "outputIndex": self.config.output_index,
                    "outputShape": str(probability_small.shape),
                    "threshold": self.config.threshold,
                    "componentAmbiguous": postprocess.ambiguous,
                    "selectedComponentLabel": postprocess.selected_label or 0,
                    "borderContact": ",".join(postprocess.border_contact),
                    "resizeMode": self.config.resize_mode,
                    "padding": f"{transform.pad_x},{transform.pad_y}",
                    "modelSha256": self.service.model_info.model_sha256
                    if self.service.model_info is not None
                    else "unknown",
                },
            ),
            timing=DetectionTiming(
                segmentation_inference_ms=inference.duration_ms,
                mask_postprocess_ms=mask_postprocess_ms,
                total_detection_ms=(time.perf_counter() - total_started_at) * 1000.0,
                additional_ms={"preprocess_ms": preprocess_ms},
            ),
            metadata={
                "modelLicense": self.model_card.license,
                "modelAttribution": self.model_card.attribution,
                "inputShape": str(tensor.shape),
                "channelOrder": self.config.channel_order,
                "resizeMode": self.config.resize_mode,
            },
            failure_code="mask_only",
        )
        debug_artifacts = _write_debug_artifacts(
            image,
            probability_mask,
            binary_mask,
            self.config.debug_artifact_dir,
        )
        return SegmentationOutput(
            probability_mask=probability_mask,
            binary_mask=binary_mask,
            transform=transform,
            postprocess=postprocess,
            detection=detection,
            debug_artifacts=debug_artifacts,
        )

    def detect(self, image: LoadedImage) -> DocumentDetectionResult:
        """Satisfy DocumentDetector while exposing only mask evidence downstream."""
        try:
            return self.segment(image).detection
        except SegmentationAdapterError as exc:
            return DocumentDetectionResult(
                detected=False,
                detector_version=self.detector_version,
                model_version=self.model_card.version,
                pipeline_version=self.pipeline_version,
                failure_code="inference_failed",
                warnings=[f"SEGMENTATION_{exc.code.value.upper()}"],
                metadata={"adapterErrorCode": exc.code.value},
            )


__all__ = [
    "OnnxSegmentationAdapter",
    "SegmentationAdapterError",
    "SegmentationConfig",
    "SegmentationErrorCode",
    "SegmentationModelCard",
    "SegmentationOutput",
    "SegmentationTransform",
    "decode_segmentation_output",
    "preprocess_segmentation_input",
]
