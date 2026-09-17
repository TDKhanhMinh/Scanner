"""Grid-guided, content-preserving dewarping for photographed forms.

The detector produces a planar draft first. This module then estimates the
curved row and column lines that remain in that draft and builds one inverse
pixel map back to the source image. It is deliberately conservative: a map is
rejected when line evidence is weak, curves cross, or the remap is not
monotone. No text or table content is synthesized.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from pydantic import Field, model_validator

from attendance_scanner.contracts import BaseContract

GRID_DEWARP_VERSION = "1.0"
GridDiagnosticValue = Union[str, int, float, bool, None]


class GridDewarpConfig(BaseContract):
    """Bounded configuration for automatic row/column grid dewarping."""

    # Keep experimental auto-dewarp opt-in until a representative corpus has
    # passed the visual preservation gate. The pipeline still records the
    # reason it was skipped when this remains disabled.
    enabled: bool = False
    analysis_max_dimension: int = Field(default=1400, ge=400, le=4000)
    roi_top_ratio: float = Field(default=0.14, ge=0.0, lt=0.5)
    roi_bottom_ratio: float = Field(default=0.96, gt=0.5, le=1.0)
    horizontal_kernel_ratio: float = Field(default=0.10, gt=0.01, le=0.5)
    vertical_kernel_ratio: float = Field(default=0.05, gt=0.01, le=0.5)
    minimum_horizontal_lines: int = Field(default=6, ge=3, le=64)
    minimum_vertical_lines: int = Field(default=4, ge=2, le=64)
    minimum_response: float = Field(default=8.0, ge=1.0, le=255.0)
    minimum_support_ratio: float = Field(default=0.35, ge=0.0, le=1.0)
    peak_min_distance_ratio: float = Field(default=0.022, gt=0.001, le=0.2)
    track_radius_ratio: float = Field(default=0.035, gt=0.005, le=0.2)
    track_smoothness: float = Field(default=0.08, ge=0.0, le=10.0)
    track_anchor_penalty: float = Field(default=0.12, ge=0.0, le=10.0)
    smoothing_sigma_ratio: float = Field(default=0.006, gt=0.0, le=0.1)
    max_displacement_ratio: float = Field(default=0.18, gt=0.0, le=0.5)
    minimum_valid_map_ratio: float = Field(default=0.985, ge=0.8, le=1.0)
    max_inverse_iterations: int = Field(default=1, ge=1, le=8)
    interpolation: int = cv2.INTER_CUBIC
    border_mode: int = cv2.BORDER_CONSTANT
    border_value: Tuple[int, int, int] = (255, 255, 255)

    @model_validator(mode="after")
    def validate_region(self) -> "GridDewarpConfig":
        if self.roi_top_ratio >= self.roi_bottom_ratio:
            raise ValueError("roi_top_ratio must be less than roi_bottom_ratio")
        return self


@dataclass(frozen=True)
class GridDewarpResult:
    """Image plus bounded evidence from one dewarp attempt."""

    image: np.ndarray
    applied: bool
    diagnostics: Dict[str, GridDiagnosticValue]


def _gray(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim not in (2, 3):
        raise ValueError("grid dewarp expects an uint8 grayscale or BGR image")
    if array.ndim == 2:
        return array
    if array.shape[2] != 3:
        raise ValueError("grid dewarp expects a three-channel BGR image")
    return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)


def _analysis_image(image: np.ndarray, maximum: int) -> Tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= maximum:
        return image.copy(), 1.0
    scale = maximum / float(longest)
    resized = cv2.resize(
        image,
        (max(2, round(width * scale)), max(2, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _line_response(gray: np.ndarray, *, horizontal: bool, kernel_ratio: float) -> np.ndarray:
    height, width = gray.shape[:2]
    if horizontal:
        kernel_size = max(9, round(width * kernel_ratio))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
    else:
        kernel_size = max(9, round(height * kernel_ratio))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size))
    response = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    return cv2.GaussianBlur(response, (0, 0), sigmaX=1.2)


def _profile_peaks(
    response: np.ndarray,
    *,
    axis_start: int,
    axis_end: int,
    travel_start: int,
    travel_end: int,
    minimum_distance: int,
    minimum_response: float,
    maximum_lines: int,
) -> List[int]:
    profile = response[axis_start:axis_end, travel_start:travel_end].mean(axis=1)
    if profile.size < 3 or float(profile.max()) < minimum_response:
        return []
    profile = cv2.GaussianBlur(profile.astype(np.float32).reshape(-1, 1), (0, 0), 2.0).ravel()
    threshold = max(minimum_response, float(np.percentile(profile, 75.0)))
    candidates = [
        index
        for index in range(1, len(profile) - 1)
        if profile[index] >= threshold
        and profile[index] >= profile[index - 1]
        and profile[index] >= profile[index + 1]
    ]
    candidates.sort(key=lambda index: (-float(profile[index]), index))
    selected: List[int] = []
    for index in candidates:
        if all(abs(index - previous) >= minimum_distance for previous in selected):
            selected.append(index)
            if len(selected) >= maximum_lines:
                break
    return sorted(axis_start + index for index in selected)


def _hough_seeds(
    gray: np.ndarray,
    *,
    horizontal: bool,
    axis_start: int,
    axis_end: int,
    travel_start: int,
    travel_end: int,
    minimum_distance: int,
    maximum_lines: int,
) -> List[int]:
    """Find long line-support seeds before curve tracking.

    Long segments are preferred over a projection profile because text strokes
    and signatures create many short peaks. The returned coordinate is only a
    seed; the continuous tracker still determines the actual curved path.
    """
    observations = [
        (axis_mid, length)
        for axis_mid, length, _, _, _, _ in _hough_observations(
            gray,
            horizontal=horizontal,
            axis_start=axis_start,
            axis_end=axis_end,
            travel_start=travel_start,
            travel_end=travel_end,
        )
    ]
    observations.sort(key=lambda item: item[0])
    seeds: List[int] = []
    weights: List[float] = []
    for coordinate, length in observations:
        if not seeds or coordinate - seeds[-1] >= minimum_distance:
            seeds.append(round(coordinate))
            weights.append(length)
            continue
        total = weights[-1] + length
        seeds[-1] = round((seeds[-1] * weights[-1] + coordinate * length) / total)
        weights[-1] = total
    return seeds[:maximum_lines]


def _hough_observations(
    gray: np.ndarray,
    *,
    horizontal: bool,
    axis_start: int,
    axis_end: int,
    travel_start: int,
    travel_end: int,
) -> List[Tuple[float, float, float, float, float, float]]:
    """Return long Hough segments as (axis_mid, length, x1, y1, x2, y2)."""
    edges = cv2.Canny(gray, 50, 150)
    travel_length = max(1, travel_end - travel_start)
    lines = cv2.HoughLinesP(
        edges,
        1.0,
        np.pi / 1800.0,
        threshold=max(25, round(travel_length * 0.04)),
        minLineLength=max(30, round(travel_length * 0.16)),
        maxLineGap=max(8, round(travel_length * 0.02)),
    )
    if lines is None:
        return []
    observations: List[Tuple[float, float, float, float, float, float]] = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = float(np.hypot(dx, dy))
        angle = abs(math.degrees(math.atan2(dy, dx)))
        angle = min(angle, 180.0 - angle)
        if horizontal:
            if angle > 16.0:
                continue
            axis_mid = (float(y1) + float(y2)) / 2.0
        else:
            if abs(angle - 90.0) > 16.0:
                continue
            axis_mid = (float(x1) + float(x2)) / 2.0
        travel_min = min(float(x1), float(x2)) if horizontal else min(float(y1), float(y2))
        travel_max = max(float(x1), float(x2)) if horizontal else max(float(y1), float(y2))
        if (
            length >= max(30.0, travel_length * 0.16)
            and axis_start <= axis_mid < axis_end
            and travel_max >= travel_start
            and travel_min <= travel_end
        ):
            observations.append((axis_mid, length, float(x1), float(y1), float(x2), float(y2)))
    return observations


def _fit_hough_curves(
    observations: Sequence[Tuple[float, float, float, float, float, float]],
    seeds: Sequence[int],
    *,
    horizontal: bool,
    travel_length: int,
    radius: int,
) -> Tuple[List[np.ndarray], List[float]]:
    """Fit a bounded quadratic through Hough endpoints near each seed."""
    curves: List[np.ndarray] = []
    supports: List[float] = []
    travel = np.arange(travel_length, dtype=np.float32)
    normalized_travel = (travel / max(travel_length - 1, 1)) * 2.0 - 1.0
    for seed in seeds:
        association_radius = max(6.0, radius * 0.55)
        nearby = [
            observation
            for observation in observations
            if abs(observation[0] - seed) <= association_radius
        ]
        if len(nearby) < 2:
            continue
        points_travel: List[float] = []
        points_axis: List[float] = []
        total_length = 0.0
        for _, length, x1, y1, x2, y2 in nearby:
            total_length += length
            if horizontal:
                points_travel.extend((x1, x2))
                points_axis.extend((y1, y2))
            else:
                points_travel.extend((y1, y2))
                points_axis.extend((x1, x2))
        if len(points_travel) < 4:
            continue
        points_t = (
            np.asarray(points_travel, dtype=np.float64) / max(travel_length - 1, 1)
        ) * 2.0 - 1.0
        points_a = np.asarray(points_axis, dtype=np.float64)
        try:
            coefficients = np.polyfit(points_t, points_a, deg=2)
        except (TypeError, ValueError, np.linalg.LinAlgError):
            continue
        fitted = np.polyval(coefficients, points_t)
        residuals = np.abs(fitted - points_a)
        residual_limit = max(5.0, radius * 0.45)
        inlier_mask = residuals <= residual_limit
        if int(inlier_mask.sum()) < 4:
            continue
        refit = np.polyfit(points_t[inlier_mask], points_a[inlier_mask], deg=2)
        curve = np.asarray(np.polyval(refit, normalized_travel), dtype=np.float32)
        curve = np.clip(curve, seed - radius, seed + radius)
        inlier_length = 0.0
        for _, length, x1, y1, x2, y2 in nearby:
            segment_travel = np.asarray(
                (x1, x2) if horizontal else (y1, y2),
                dtype=np.float64,
            )
            segment_axis = np.asarray(
                (y1, y2) if horizontal else (x1, x2),
                dtype=np.float64,
            )
            segment_t = (segment_travel / max(travel_length - 1, 1)) * 2.0 - 1.0
            if float(np.max(np.abs(np.polyval(refit, segment_t) - segment_axis))) <= residual_limit:
                inlier_length += length
        support = min(1.0, inlier_length / max(total_length, 1e-9))
        if support < 0.25:
            continue
        curves.append(_smooth_curve(curve, max(1.0, radius * 0.2)))
        supports.append(float(support))
    return curves, supports


def _trim_leading_line_outliers(seeds: List[int], minimum_distance: int) -> List[int]:
    """Drop an isolated preamble before the regular table-row run."""
    if len(seeds) < 3:
        return seeds
    for index in range(len(seeds) - 1):
        if seeds[index + 1] - seeds[index] >= round(minimum_distance * 2.2):
            return seeds[index + 1 :]
    return seeds


def _track_curve(
    response: np.ndarray,
    seed: int,
    *,
    travel_start: int,
    travel_end: int,
    radius: int,
    smoothness: float,
    anchor_penalty: float,
    minimum_response: float,
) -> Tuple[np.ndarray, float]:
    """Track one continuous response ridge in both directions from a seed."""
    axis_size, travel_size = response.shape
    curve = np.full(travel_size, float(seed), dtype=np.float32)
    center = min(max((travel_start + travel_end) // 2, 0), travel_size - 1)
    curve[center] = float(np.clip(seed, 0, axis_size - 1))
    supported = 0
    measured = 0

    for direction in (-1, 1):
        previous = int(round(curve[center]))
        positions = range(center + direction, travel_start - 1, direction)
        if direction > 0:
            positions = range(center + 1, travel_end)
        for travel in positions:
            low = max(0, previous - radius)
            high = min(axis_size, previous + radius + 1)
            values = response[low:high, travel].astype(np.float32)
            offsets = np.arange(low, high, dtype=np.float32) - float(previous)
            if not len(values) or float(values.max()) < minimum_response:
                best = previous
            else:
                anchor_offsets = np.arange(low, high, dtype=np.float32) - float(seed)
                scores = (
                    values
                    - smoothness * offsets * offsets
                    - anchor_penalty * anchor_offsets * anchor_offsets
                )
                best = int(low + int(np.argmax(scores)))
            best = int(np.clip(best, seed - radius, seed + radius))
            if float(response[best, travel]) >= minimum_response:
                supported += 1
            measured += 1
            curve[travel] = float(best)
            previous = best

    if measured == 0:
        return curve, 0.0
    return curve, supported / float(measured)


def _smooth_curve(curve: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return curve
    smoothed = cv2.GaussianBlur(curve.reshape(1, -1), (0, 0), sigmaX=sigma).ravel()
    return np.asarray(smoothed, dtype=np.float32)


def _monotone_curves(curves: Sequence[np.ndarray], minimum_gap: float) -> bool:
    if len(curves) < 2:
        return True
    stacked = np.asarray(curves, dtype=np.float32)
    return bool(np.all(np.diff(stacked, axis=0) >= minimum_gap))


def _deduplicate_tracked_curves(
    curves: List[np.ndarray],
    supports: List[float],
    *,
    minimum_gap: int,
) -> Tuple[List[np.ndarray], List[float]]:
    """Remove two trackers that converged to the same physical stroke."""
    if not curves:
        return curves, supports
    order = sorted(range(len(curves)), key=lambda index: float(np.median(curves[index])))
    kept: List[int] = []
    for index in order:
        if not kept:
            kept.append(index)
            continue
        previous = kept[-1]
        gap = float(np.median(curves[index])) - float(np.median(curves[previous]))
        if gap < minimum_gap:
            if supports[index] > supports[previous]:
                kept[-1] = index
        else:
            kept.append(index)
    kept.sort(key=lambda index: float(np.median(curves[index])))
    return [curves[index] for index in kept], [supports[index] for index in kept]


def _project_ordered_curves(
    curves: List[np.ndarray],
    *,
    minimum_gap: float,
) -> Tuple[List[np.ndarray], float]:
    """Project small local crossings back to the original curve order."""
    if len(curves) < 2:
        return curves, 0.0
    projected = np.asarray(curves, dtype=np.float32).copy()
    original = projected.copy()
    for travel in range(projected.shape[1]):
        for index in range(1, projected.shape[0]):
            projected[index, travel] = max(
                projected[index, travel],
                projected[index - 1, travel] + minimum_gap,
            )
    displacement = float(np.max(np.abs(projected - original)))
    return [projected[index] for index in range(projected.shape[0])], displacement


def _inverse_control_map(
    source_curves: Sequence[np.ndarray],
    target_positions: Sequence[float],
    *,
    axis_length: int,
    travel_length: int,
    horizontal: bool,
) -> np.ndarray:
    """Build a target-axis -> source-axis map for every source travel position."""
    curves = np.asarray(source_curves, dtype=np.float32)
    targets = np.asarray(target_positions, dtype=np.float32)
    inverse = np.empty((axis_length, travel_length), dtype=np.float32)
    if horizontal:
        for travel in range(travel_length):
            inverse[:, travel] = np.interp(
                np.arange(axis_length, dtype=np.float32),
                np.r_[0.0, targets, float(axis_length - 1)],
                np.r_[0.0, curves[:, travel], float(axis_length - 1)],
            )
    else:
        for travel in range(travel_length):
            inverse[:, travel] = np.interp(
                np.arange(axis_length, dtype=np.float32),
                np.r_[0.0, targets, float(axis_length - 1)],
                np.r_[0.0, curves[:, travel], float(axis_length - 1)],
            )
    return inverse


def _build_inverse_maps(
    image_shape: Tuple[int, int],
    rows: Sequence[np.ndarray],
    row_targets: Sequence[float],
    columns: Sequence[np.ndarray],
    column_targets: Sequence[float],
    *,
    iterations: int,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = image_shape
    row_inverse = _inverse_control_map(
        rows,
        row_targets,
        axis_length=height,
        travel_length=width,
        horizontal=True,
    )
    column_inverse = _inverse_control_map(
        columns,
        column_targets,
        axis_length=width,
        travel_length=height,
        horizontal=False,
    )
    destination_x, destination_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    source_x = destination_x.copy()
    source_y = destination_y.copy()
    for _ in range(iterations):
        source_x = np.asarray(
            cv2.remap(
                column_inverse.T,
                destination_x,
                source_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            ),
            dtype=np.float32,
        )
        source_y = np.asarray(
            cv2.remap(
                row_inverse,
                source_x,
                destination_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            ),
            dtype=np.float32,
        )
    return source_x, source_y


def _safe_map(
    source_x: np.ndarray,
    source_y: np.ndarray,
    *,
    source_shape: Tuple[int, int],
    minimum_valid_ratio: float,
) -> Tuple[bool, float, float, float, float]:
    """Return the legacy compact safety result for a candidate map.

    ``_evaluate_map`` contains the complete quality metrics.  Keeping this
    small wrapper preserves the focused contract used by unit tests and by
    callers that only need the original five values.
    """
    safe, valid_ratio, displacement, minimum_step, maximum_step, _, _, _ = _evaluate_map(
        source_x,
        source_y,
        source_shape=source_shape,
        minimum_valid_ratio=minimum_valid_ratio,
    )
    return safe, valid_ratio, displacement, minimum_step, maximum_step


def _evaluate_map(
    source_x: np.ndarray,
    source_y: np.ndarray,
    *,
    source_shape: Tuple[int, int],
    minimum_valid_ratio: float,
) -> Tuple[bool, float, float, float, float, float, float, float]:
    """Evaluate monotonicity and local Jacobian quality of an inverse map.

    Checking only ``x`` steps across columns and ``y`` steps across rows is
    insufficient: a map can pass those checks while its cross-axis component
    folds or makes a long jump.  The Jacobian and cross-step bounds reject
    that class of remap before any pixels are sampled.
    """
    height, width = source_shape
    finite = np.isfinite(source_x) & np.isfinite(source_y)
    in_bounds = (
        (source_x >= 0.0)
        & (source_x <= width - 1.0)
        & (source_y >= 0.0)
        & (source_y <= height - 1.0)
    )
    valid_ratio = float(np.mean(finite & in_bounds))
    dx = np.diff(source_x, axis=1)
    dy = np.diff(source_y, axis=0)
    minimum_step = min(float(dx.min()), float(dy.min()))
    maximum_step = max(float(np.abs(dx).max()), float(np.abs(dy).max()))
    cross_x = np.diff(source_x, axis=0)
    cross_y = np.diff(source_y, axis=1)
    maximum_cross_step = max(
        float(np.abs(cross_x).max()),
        float(np.abs(cross_y).max()),
    )
    if height > 1 and width > 1:
        jacobian = dx[:-1, :] * dy[:, :-1] - cross_x[:, :-1] * cross_y[:-1, :]
        minimum_jacobian = float(jacobian.min())
        maximum_jacobian = float(jacobian.max())
    else:
        minimum_jacobian = 1.0
        maximum_jacobian = 1.0
    # A sub-pixel negative step or a large local jump means the tracked grid
    # is folding or switching strokes. Such a map can create black seams and
    # must never be accepted merely because it is finite and in bounds. The
    # Jacobian bounds also reject cross-axis tearing that the axis-only checks
    # cannot see.
    monotone = bool(
        minimum_step >= 0.0
        and maximum_step <= 2.0
        and maximum_cross_step <= 2.0
        and minimum_jacobian >= 0.15
        and maximum_jacobian <= 4.0
        and np.all(dx >= 0.0)
        and np.all(dy >= 0.0)
    )
    displacement = float(
        max(
            np.max(np.abs(source_x - np.arange(width, dtype=np.float32)[None, :])),
            np.max(np.abs(source_y - np.arange(height, dtype=np.float32)[:, None])),
        )
    )
    return (
        valid_ratio >= minimum_valid_ratio and monotone,
        valid_ratio,
        displacement,
        minimum_step,
        maximum_step,
        maximum_cross_step,
        minimum_jacobian,
        maximum_jacobian,
    )


def _not_applied(
    image: np.ndarray,
    reason: str,
    **diagnostics: GridDiagnosticValue,
) -> GridDewarpResult:
    return GridDewarpResult(
        image=image.copy(),
        applied=False,
        diagnostics={
            "version": GRID_DEWARP_VERSION,
            "applied": False,
            "reason": reason,
            **diagnostics,
        },
    )


def dewarp_document_grid(
    image: np.ndarray,
    *,
    config: Optional[GridDewarpConfig] = None,
) -> GridDewarpResult:
    """Estimate a safe row/column mesh and dewarp one document image."""
    policy = config or GridDewarpConfig()
    if not policy.enabled:
        return _not_applied(image, "disabled")
    gray = _gray(image)
    analysis, scale = _analysis_image(image, policy.analysis_max_dimension)
    analysis_gray = _gray(analysis)
    height, width = analysis_gray.shape[:2]
    top = round(height * policy.roi_top_ratio)
    bottom = round(height * policy.roi_bottom_ratio)
    if bottom - top < 32:
        return _not_applied(image, "roi_too_small")

    horizontal_response = _line_response(
        analysis_gray,
        horizontal=True,
        kernel_ratio=policy.horizontal_kernel_ratio,
    ).astype(np.float32)
    vertical_response = _line_response(
        analysis_gray,
        horizontal=False,
        kernel_ratio=policy.vertical_kernel_ratio,
    ).astype(np.float32)
    minimum_row_distance = max(5, round(height * policy.peak_min_distance_ratio))
    minimum_column_distance = max(5, round(width * policy.peak_min_distance_ratio))
    row_seeds = _hough_seeds(
        analysis_gray,
        horizontal=True,
        axis_start=top,
        axis_end=bottom,
        travel_start=round(width * 0.04),
        travel_end=round(width * 0.96),
        minimum_distance=minimum_row_distance,
        maximum_lines=64,
    )
    row_seeds = _trim_leading_line_outliers(row_seeds, minimum_row_distance)
    if len(row_seeds) < policy.minimum_horizontal_lines:
        row_seeds = _profile_peaks(
            horizontal_response,
            axis_start=top,
            axis_end=bottom,
            travel_start=round(width * 0.04),
            travel_end=round(width * 0.96),
            minimum_distance=minimum_row_distance,
            minimum_response=policy.minimum_response,
            maximum_lines=64,
        )
        row_seeds = _trim_leading_line_outliers(row_seeds, minimum_row_distance)
    column_seeds = _hough_seeds(
        analysis_gray,
        horizontal=False,
        axis_start=round(width * 0.04),
        axis_end=round(width * 0.96),
        travel_start=top,
        travel_end=bottom,
        minimum_distance=minimum_column_distance,
        maximum_lines=64,
    )
    if len(column_seeds) < policy.minimum_vertical_lines:
        column_seeds = _profile_peaks(
            vertical_response.T,
            axis_start=round(width * 0.04),
            axis_end=round(width * 0.96),
            travel_start=top,
            travel_end=bottom,
            minimum_distance=minimum_column_distance,
            minimum_response=policy.minimum_response,
            maximum_lines=64,
        )
    if len(row_seeds) < policy.minimum_horizontal_lines:
        return _not_applied(
            image,
            "insufficient_horizontal_lines",
            horizontalLines=len(row_seeds),
            verticalLines=len(column_seeds),
        )
    if len(column_seeds) < policy.minimum_vertical_lines:
        return _not_applied(
            image,
            "insufficient_vertical_lines",
            horizontalLines=len(row_seeds),
            verticalLines=len(column_seeds),
        )

    row_radius = max(4, round(height * policy.track_radius_ratio))
    column_radius = max(4, round(width * policy.track_radius_ratio))
    row_observations = _hough_observations(
        analysis_gray,
        horizontal=True,
        axis_start=top,
        axis_end=bottom,
        travel_start=round(width * 0.03),
        travel_end=round(width * 0.97),
    )
    rows, row_support = _fit_hough_curves(
        row_observations,
        row_seeds,
        horizontal=True,
        travel_length=width,
        radius=row_radius,
    )
    if len(rows) < policy.minimum_horizontal_lines:
        rows = []
        row_support = []
        for seed in row_seeds:
            curve, support = _track_curve(
                horizontal_response,
                seed,
                travel_start=round(width * 0.03),
                travel_end=round(width * 0.97),
                radius=row_radius,
                smoothness=policy.track_smoothness,
                anchor_penalty=policy.track_anchor_penalty,
                minimum_response=policy.minimum_response,
            )
            rows.append(_smooth_curve(curve, max(1.0, height * policy.smoothing_sigma_ratio)))
            row_support.append(support)
    rows, row_support = _deduplicate_tracked_curves(
        rows,
        row_support,
        minimum_gap=max(4, round(min(height, width) * 0.006)),
    )
    column_observations = _hough_observations(
        analysis_gray,
        horizontal=False,
        axis_start=round(width * 0.03),
        axis_end=round(width * 0.97),
        travel_start=top,
        travel_end=bottom,
    )
    columns, column_support = _fit_hough_curves(
        column_observations,
        column_seeds,
        horizontal=False,
        travel_length=height,
        radius=column_radius,
    )
    if len(columns) < policy.minimum_vertical_lines:
        columns = []
        column_support = []
        for seed in column_seeds:
            curve, support = _track_curve(
                vertical_response.T,
                seed,
                travel_start=top,
                travel_end=bottom,
                radius=column_radius,
                smoothness=policy.track_smoothness,
                anchor_penalty=policy.track_anchor_penalty,
                minimum_response=policy.minimum_response,
            )
            columns.append(_smooth_curve(curve, max(1.0, width * policy.smoothing_sigma_ratio)))
            column_support.append(support)
    columns, column_support = _deduplicate_tracked_curves(
        columns,
        column_support,
        minimum_gap=max(4, round(min(height, width) * 0.006)),
    )

    row_support_mean = float(np.mean(row_support)) if row_support else 0.0
    column_support_mean = float(np.mean(column_support)) if column_support else 0.0
    if min(row_support_mean, column_support_mean) < policy.minimum_support_ratio:
        return _not_applied(
            image,
            "weak_line_support",
            horizontalLines=len(rows),
            verticalLines=len(columns),
            horizontalSupport=round(row_support_mean, 4),
            verticalSupport=round(column_support_mean, 4),
        )
    # Adjacent strokes can share a pixel after analysis downscaling. Equality
    # is safe here; only an actual crossing (negative gap) is rejected.
    rows, row_order_projection = _project_ordered_curves(rows, minimum_gap=0.25)
    columns, column_order_projection = _project_ordered_curves(columns, minimum_gap=0.25)
    max_projection = max(row_order_projection, column_order_projection)
    if max_projection > max(height, width) * 0.04:
        return _not_applied(
            image,
            "crossing_grid_curves",
            horizontalLines=len(rows),
            verticalLines=len(columns),
            orderProjectionPx=round(max_projection / max(scale, 1e-9), 3),
        )

    row_targets = [float(np.median(curve)) for curve in rows]
    column_targets = [float(np.median(curve)) for curve in columns]
    source_x, source_y = _build_inverse_maps(
        (height, width),
        rows,
        row_targets,
        columns,
        column_targets,
        iterations=policy.max_inverse_iterations,
    )
    (
        safe,
        valid_ratio,
        displacement,
        minimum_step,
        maximum_step,
        maximum_cross_step,
        minimum_jacobian,
        maximum_jacobian,
    ) = _evaluate_map(
        source_x,
        source_y,
        source_shape=(height, width),
        minimum_valid_ratio=policy.minimum_valid_map_ratio,
    )
    max_displacement = policy.max_displacement_ratio * min(height, width)
    if not safe:
        return _not_applied(
            image,
            "unsafe_inverse_map",
            horizontalLines=len(rows),
            verticalLines=len(columns),
            validMapRatio=round(valid_ratio, 4),
            maxDisplacementPx=round(displacement / max(scale, 1e-9), 3),
            minimumMapStep=round(minimum_step, 4),
            maximumMapStep=round(maximum_step, 4),
            maximumCrossStep=round(maximum_cross_step, 4),
            minimumJacobian=round(minimum_jacobian, 4),
            maximumJacobian=round(maximum_jacobian, 4),
        )
    if displacement > max_displacement:
        return _not_applied(
            image,
            "displacement_limit",
            horizontalLines=len(rows),
            verticalLines=len(columns),
            validMapRatio=round(valid_ratio, 4),
            maxDisplacementPx=round(displacement / max(scale, 1e-9), 3),
        )

    source_height, source_width = gray.shape[:2]
    full_x = np.asarray(
        cv2.resize(source_x, (source_width, source_height), interpolation=cv2.INTER_LINEAR),
        dtype=np.float32,
    )
    full_y = np.asarray(
        cv2.resize(source_y, (source_width, source_height), interpolation=cv2.INTER_LINEAR),
        dtype=np.float32,
    )
    full_x /= max(scale, 1e-9)
    full_y /= max(scale, 1e-9)
    (
        safe_full,
        valid_full,
        full_displacement,
        full_min_step,
        full_max_step,
        full_cross_step,
        full_min_jacobian,
        full_max_jacobian,
    ) = _evaluate_map(
        full_x,
        full_y,
        source_shape=(source_height, source_width),
        minimum_valid_ratio=policy.minimum_valid_map_ratio,
    )
    if not safe_full:
        return _not_applied(
            image,
            "unsafe_full_resolution_map",
            horizontalLines=len(rows),
            verticalLines=len(columns),
            validMapRatio=round(valid_full, 4),
            maxDisplacementPx=round(full_displacement, 3),
            minimumMapStep=round(full_min_step, 4),
            maximumMapStep=round(full_max_step, 4),
            maximumCrossStep=round(full_cross_step, 4),
            minimumJacobian=round(full_min_jacobian, 4),
            maximumJacobian=round(full_max_jacobian, 4),
        )
    remapped = cv2.remap(
        image,
        full_x.astype(np.float32),
        full_y.astype(np.float32),
        policy.interpolation,
        borderMode=policy.border_mode,
        borderValue=policy.border_value,
    )
    diagnostics: Dict[str, GridDiagnosticValue] = {
        "version": GRID_DEWARP_VERSION,
        "applied": True,
        "reason": "accepted",
        "horizontalLines": len(rows),
        "verticalLines": len(columns),
        "horizontalSupport": round(row_support_mean, 4),
        "verticalSupport": round(column_support_mean, 4),
        "orderProjectionPx": round(max_projection / max(scale, 1e-9), 3),
        "validMapRatio": round(valid_full, 4),
        "maxDisplacementPx": round(full_displacement, 3),
        "minimumMapStep": round(full_min_step, 4),
        "maximumMapStep": round(full_max_step, 4),
        "maximumCrossStep": round(full_cross_step, 4),
        "minimumJacobian": round(full_min_jacobian, 4),
        "maximumJacobian": round(full_max_jacobian, 4),
        "analysisScale": round(scale, 6),
        "outputWidth": source_width,
        "outputHeight": source_height,
    }
    return GridDewarpResult(image=remapped, applied=True, diagnostics=diagnostics)


__all__ = [
    "GRID_DEWARP_VERSION",
    "GridDewarpConfig",
    "GridDewarpResult",
    "dewarp_document_grid",
]
