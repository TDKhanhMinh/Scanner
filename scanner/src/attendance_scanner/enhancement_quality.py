"""Readability metrics and visual comparison artifacts for enhancement tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import cv2
import numpy as np
from pydantic import Field

from attendance_scanner.contracts import BaseContract, ScanMode
from attendance_scanner.pipeline.enhance import EnhancementConfig, enhance_image
from attendance_scanner.pipeline.load import LoadedImage
from attendance_scanner.pipeline.perspective import WarpedDocument


class EnhancementModeMetrics(BaseContract):
    """Interpretable, image-size-independent proxies for timesheet readability."""

    mode: ScanMode
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    channels: int = Field(ge=1, le=3)
    luminance_contrast: float = Field(ge=0.0)
    edge_density: float = Field(ge=0.0, le=1.0)
    source_edge_recall: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    fine_stroke_recall: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    color_ink_retention: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    readability_score: float = Field(ge=0.0, le=1.0)


class EnhancementComparison(BaseContract):
    """Comparison report for all supported enhancement modes on one warped page."""

    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    modes: List[EnhancementModeMetrics] = Field(min_length=1)
    recommended_mode: ScanMode
    recommendation_reason: str = Field(min_length=1)
    artifacts: Dict[str, str] = Field(default_factory=dict)


def _as_bgr(image: Union[np.ndarray, WarpedDocument, LoadedImage]) -> np.ndarray:
    if isinstance(image, WarpedDocument):
        value = image.image
    elif isinstance(image, LoadedImage):
        value = image.image
    elif isinstance(image, np.ndarray):
        value = image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    if value.dtype != np.uint8 or value.ndim not in {2, 3}:
        raise ValueError("comparison image must be a uint8 2D/3D array")
    if value.ndim == 2:
        return cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)
    if value.shape[2] != 3:
        raise ValueError("comparison image must have three color channels")
    return value


def _luminance(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def _edge_recall(source_gray: np.ndarray, output_gray: np.ndarray) -> float:
    source_edges = cv2.Canny(source_gray, 40, 120) > 0
    output_edges = cv2.Canny(output_gray, 40, 120) > 0
    source_count = int(source_edges.sum())
    if source_count == 0:
        return 1.0
    return float(np.logical_and(source_edges, output_edges).sum() / source_count)


def _color_ink_retention(source: np.ndarray, output: np.ndarray) -> Optional[float]:
    if output.ndim != 3:
        return None
    source_chroma = source.max(axis=2).astype(np.int16) - source.min(axis=2).astype(np.int16)
    output_chroma = output.max(axis=2).astype(np.int16) - output.min(axis=2).astype(np.int16)
    source_ink = (source_chroma >= 24) & (_luminance(source) < 235)
    source_count = int(source_ink.sum())
    if source_count == 0:
        return 1.0
    return float(np.logical_and(source_ink, output_chroma >= 16).sum() / source_count)


def _metrics(
    mode: ScanMode,
    source: np.ndarray,
    output: np.ndarray,
    source_gray: np.ndarray,
) -> EnhancementModeMetrics:
    output_gray = _luminance(output)
    low, high = np.percentile(output_gray, (5.0, 95.0))
    contrast = float(max(0.0, high - low))
    edges = cv2.Canny(output_gray, 40, 120) > 0
    edge_density = float(edges.mean())
    edge_recall = _edge_recall(source_gray, output_gray)
    color_retention = _color_ink_retention(source, output)
    # These weights favor visible strokes and stable contrast. Color retention is
    # included only for color-capable outputs; no file-size or pixel-count term is used.
    contrast_score = min(1.0, contrast / 128.0)
    density_score = min(1.0, edge_density / 0.20)
    color_score = color_retention if color_retention is not None else 1.0
    readability = min(
        1.0,
        0.50 * edge_recall + 0.25 * contrast_score + 0.10 * density_score + 0.15 * color_score,
    )
    return EnhancementModeMetrics(
        mode=mode,
        width=output.shape[1],
        height=output.shape[0],
        channels=1 if output.ndim == 2 else output.shape[2],
        luminance_contrast=contrast,
        edge_density=edge_density,
        source_edge_recall=edge_recall,
        fine_stroke_recall=edge_recall,
        color_ink_retention=color_retention,
        readability_score=readability,
    )


def compare_enhancement_modes(
    image: Union[np.ndarray, WarpedDocument, LoadedImage],
    *,
    config: Optional[EnhancementConfig] = None,
    modes: Optional[Sequence[ScanMode]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> EnhancementComparison:
    """Compare modes on one page and optionally write per-mode visual artifacts."""
    source = _as_bgr(image)
    source_gray = _luminance(source)
    selected_modes = list(
        modes or (ScanMode.GRAY, ScanMode.BW, ScanMode.COLOR, ScanMode.SMART_DOCUMENT)
    )
    if not selected_modes:
        raise ValueError("At least one enhancement mode is required")

    metrics: List[EnhancementModeMetrics] = []
    artifacts: Dict[str, str] = {}
    artifact_root = Path(output_dir) if output_dir is not None else None
    if artifact_root is not None:
        artifact_root.mkdir(parents=True, exist_ok=True)
    for mode in selected_modes:
        output = enhance_image(source, mode=mode, config=config)
        metrics.append(_metrics(mode, source, output, source_gray))
        if artifact_root is not None:
            artifact_path = artifact_root / f"{mode.value}.png"
            if not cv2.imwrite(str(artifact_path), output):
                raise OSError(f"Unable to write enhancement artifact: {artifact_path}")
            artifacts[mode.value] = str(artifact_path)

    # Stable tie-break keeps Gray as the conservative default when readability
    # scores are effectively equal; the decision never depends on file size.
    priority = {mode: index for index, mode in enumerate(selected_modes)}
    recommended = max(metrics, key=lambda item: (item.readability_score, -priority[item.mode]))
    reason = (
        f"{recommended.mode.value} đạt điểm readability {recommended.readability_score:.3f}; "
        "điểm ưu tiên edge/stroke và contrast, không dùng file size."
    )
    comparison = EnhancementComparison(
        source_width=source.shape[1],
        source_height=source.shape[0],
        modes=metrics,
        recommended_mode=recommended.mode,
        recommendation_reason=reason,
        artifacts=artifacts,
    )
    if artifact_root is not None:
        comparison_path = artifact_root / "comparison.json"
        comparison_path.write_text(
            comparison.model_dump_json(by_alias=True, indent=2),
            encoding="utf-8",
        )
        artifacts["comparison"] = str(comparison_path)
        comparison = comparison.model_copy(update={"artifacts": artifacts})
        comparison_path.write_text(
            comparison.model_dump_json(by_alias=True, indent=2),
            encoding="utf-8",
        )
    return comparison


def load_comparison_report(path: Union[str, Path]) -> EnhancementComparison:
    """Load a previously generated comparison report."""
    return EnhancementComparison.model_validate_json(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "EnhancementComparison",
    "EnhancementModeMetrics",
    "compare_enhancement_modes",
    "load_comparison_report",
]
