"""Lightweight document-mask cleanup and connected-component selection."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class MaskPostprocessConfig(BaseModel):
    """Bounded cleanup and component-ranking policy for a probability mask."""

    model_config = ConfigDict(extra="forbid")

    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    open_kernel_size: int = Field(default=0, ge=0, le=15)
    close_kernel_size: int = Field(default=3, ge=0, le=15)
    morphology_iterations: int = Field(default=1, ge=0, le=2)
    min_component_area_ratio: float = Field(default=0.0005, ge=0.0, le=0.25)
    area_weight: float = Field(default=0.45, ge=0.0, le=1.0)
    centrality_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    confidence_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    border_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    ambiguity_margin: float = Field(default=0.08, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_kernel_settings(self) -> "MaskPostprocessConfig":
        for name, size in (
            ("open_kernel_size", self.open_kernel_size),
            ("close_kernel_size", self.close_kernel_size),
        ):
            if size and size % 2 == 0:
                raise ValueError(f"{name} must be odd or zero")
        if self.morphology_iterations and not (self.open_kernel_size or self.close_kernel_size):
            raise ValueError("morphology_iterations requires an open or close kernel")
        return self


class MaskComponent(BaseModel):
    """Reviewable connected-component evidence."""

    model_config = ConfigDict(extra="forbid")

    label: int = Field(gt=0)
    area_px: int = Field(gt=0)
    area_ratio: float = Field(gt=0.0, le=1.0)
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    mean_confidence: float = Field(ge=0.0, le=1.0)
    centrality: float = Field(ge=0.0, le=1.0)
    border_contact: List[str] = Field(default_factory=list)
    score: float = Field(ge=0.0, le=1.0)


class MaskPostprocessResult:
    """Internal cleaned/selected masks and component diagnostics."""

    def __init__(
        self,
        *,
        probability_mask: np.ndarray,
        cleaned_mask: np.ndarray,
        selected_mask: np.ndarray,
        components: List[MaskComponent],
        selected_label: Optional[int],
        ambiguous: bool,
        border_contact: List[str],
    ) -> None:
        self.probability_mask = probability_mask
        self.cleaned_mask = cleaned_mask
        self.selected_mask = selected_mask
        self.components = components
        self.selected_label = selected_label
        self.ambiguous = ambiguous
        self.border_contact = border_contact

    @property
    def component_count(self) -> int:
        return len(self.components)


def _kernel(size: int) -> Optional[np.ndarray]:
    if size == 0:
        return None
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def postprocess_document_mask(
    probability_mask: np.ndarray,
    config: Optional[MaskPostprocessConfig] = None,
) -> MaskPostprocessResult:
    """Threshold, lightly clean, and rank mask components deterministically."""
    policy = config or MaskPostprocessConfig()
    probability = np.asarray(probability_mask, dtype=np.float32)
    if probability.ndim != 2 or probability.size == 0:
        raise ValueError(f"probability_mask must be a non-empty 2D array, got {probability.shape}")
    if (
        not np.isfinite(probability).all()
        or float(probability.min()) < 0.0
        or float(probability.max()) > 1.0
    ):
        raise ValueError("probability_mask must contain finite values in [0,1]")

    binary = (probability >= policy.threshold).astype(np.uint8)
    open_kernel = _kernel(policy.open_kernel_size)
    close_kernel = _kernel(policy.close_kernel_size)
    cleaned: np.ndarray = binary
    if policy.morphology_iterations:
        if open_kernel is not None:
            cleaned = cv2.morphologyEx(
                cleaned,
                cv2.MORPH_OPEN,
                open_kernel,
                iterations=policy.morphology_iterations,
            )
        if close_kernel is not None:
            cleaned = cv2.morphologyEx(
                cleaned,
                cv2.MORPH_CLOSE,
                close_kernel,
                iterations=policy.morphology_iterations,
            )

    height, width = cleaned.shape
    total_area = float(height * width)
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )
    components: List[MaskComponent] = []
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    max_distance = max(math.hypot(center_x, center_y), 1.0)
    min_area = total_area * policy.min_component_area_ratio
    weight_total = (
        policy.area_weight
        + policy.centrality_weight
        + policy.confidence_weight
        + policy.border_weight
    ) or 1.0
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        area_ratio = area / total_area
        if area <= 0 or area < min_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        centroid = (float(centroids[label][0]), float(centroids[label][1]))
        centrality = max(
            0.0,
            1.0 - math.hypot(centroid[0] - center_x, centroid[1] - center_y) / max_distance,
        )
        contacts: List[str] = []
        if x == 0:
            contacts.append("left")
        if y == 0:
            contacts.append("top")
        if x + component_width == width:
            contacts.append("right")
        if y + component_height == height:
            contacts.append("bottom")
        confidence = float(probability[labels == label].mean())
        border_score = 0.0 if contacts else 1.0
        area_score = min(1.0, area_ratio / 0.5)
        score = (
            policy.area_weight * area_score
            + policy.centrality_weight * centrality
            + policy.confidence_weight * confidence
            + policy.border_weight * border_score
        ) / weight_total
        components.append(
            MaskComponent(
                label=label,
                area_px=area,
                area_ratio=area_ratio,
                bbox=(x, y, component_width, component_height),
                centroid=centroid,
                mean_confidence=confidence,
                centrality=centrality,
                border_contact=contacts,
                score=max(0.0, min(1.0, score)),
            )
        )
    components.sort(key=lambda component: (-component.score, -component.area_px, component.label))
    selected = components[0] if components else None
    ambiguous = bool(
        selected is not None
        and len(components) > 1
        and selected.score - components[1].score <= policy.ambiguity_margin
    )
    selected_mask = np.zeros_like(cleaned, dtype=bool)
    if selected is not None:
        selected_mask = labels == selected.label
    return MaskPostprocessResult(
        probability_mask=probability,
        cleaned_mask=cleaned.astype(bool),
        selected_mask=selected_mask,
        components=components,
        selected_label=selected.label if selected is not None else None,
        ambiguous=ambiguous,
        border_contact=list(selected.border_contact) if selected is not None else [],
    )


__all__ = [
    "MaskComponent",
    "MaskPostprocessConfig",
    "MaskPostprocessResult",
    "postprocess_document_mask",
]
