"""Grid-guided, content-preserving dewarping for photographed forms.

The detector produces a planar draft first. This module then estimates the
curved row and column lines that remain in that draft and builds one inverse
pixel map back to the source image. It is deliberately conservative: a map is
rejected when line evidence is weak, curves cross, or the remap is not
monotone. No text or table content is synthesized.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from pydantic import Field, model_validator

from attendance_scanner.contracts import BaseContract

GRID_DEWARP_VERSION = "1.1"
GridDiagnosticValue = Union[str, int, float, bool, None]


class GridDewarpConfig(BaseContract):
    """Bounded configuration for automatic row/column grid dewarping."""

    # The product pipeline decides whether this guarded stage is enabled. The
    # function itself remains opt-in for direct library callers.
    enabled: bool = False
    analysis_max_dimension: int = Field(default=1400, ge=400, le=4000)
    roi_top_ratio: float = Field(default=0.14, ge=0.0, lt=0.5)
    roi_bottom_ratio: float = Field(default=0.96, gt=0.5, le=1.0)
    horizontal_kernel_ratio: float = Field(default=0.10, gt=0.01, le=0.5)
    vertical_kernel_ratio: float = Field(default=0.05, gt=0.01, le=0.5)
    minimum_horizontal_lines: int = Field(default=6, ge=3, le=64)
    minimum_vertical_lines: int = Field(default=4, ge=2, le=64)
    minimum_response: float = Field(default=8.0, ge=1.0, le=255.0)
    minimum_support_ratio: float = Field(default=0.45, ge=0.0, le=1.0)
    peak_min_distance_ratio: float = Field(default=0.022, gt=0.001, le=0.2)
    track_radius_ratio: float = Field(default=0.020, gt=0.005, le=0.2)
    track_smoothness: float = Field(default=0.80, ge=0.0, le=10.0)
    track_anchor_penalty: float = Field(default=0.08, ge=0.0, le=10.0)
    smoothing_sigma_ratio: float = Field(default=0.015, gt=0.0, le=0.1)
    max_displacement_ratio: float = Field(default=0.06, gt=0.0, le=0.5)
    minimum_valid_map_ratio: float = Field(default=0.999, ge=0.8, le=1.0)
    maximum_order_projection_ratio: float = Field(default=0.003, ge=0.0, le=0.05)
    maximum_axis_step: float = Field(default=1.8, gt=1.0, le=8.0)
    maximum_cross_step: float = Field(default=1.5, gt=0.0, le=8.0)
    minimum_jacobian: float = Field(default=0.15, gt=0.0, le=1.0)
    maximum_jacobian: float = Field(default=2.5, ge=1.0, le=8.0)
    minimum_straightness_gain: float = Field(default=0.004, ge=0.0, le=0.5)
    maximum_straightness_regression: float = Field(default=0.02, ge=0.0, le=0.5)
    minimum_hough_fast_path_score: float = Field(default=0.30, ge=0.0, le=1.0)
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


def _line_response_decoupled(
    gray: np.ndarray,
    *,
    horizontal_kernel_ratio: float,
    vertical_kernel_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute decoupled horizontal and vertical line responses with cross-subtraction.

    For dark lines on a lighter document:
    - Horizontal lines are thin vertically, so a vertical structuring element (1, kv)
      closes across them. Blackhat (closing - src) yields strong positive response.
    - Vertical lines are thin horizontally, so a horizontal structuring element (kh, 1)
      closes across them.
    Cross-subtracting the perpendicular response suppresses high-response blobs at
    row-column crossings, preventing trackers from getting pulled off-axis.
    """
    height, width = gray.shape[:2]
    kh = max(9, round(width * horizontal_kernel_ratio))
    kv = max(9, round(height * vertical_kernel_ratio))
    k_vert_for_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kv))
    k_horiz_for_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (kh, 1))
    h_raw = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_vert_for_horiz).astype(np.float32)
    v_raw = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_horiz_for_vert).astype(np.float32)
    h_clean = np.maximum(0.0, h_raw - 0.4 * v_raw)
    v_clean = np.maximum(0.0, v_raw - 0.4 * h_raw)
    h_resp = cv2.GaussianBlur(h_clean, (0, 0), sigmaX=1.2)
    v_resp = cv2.GaussianBlur(v_clean, (0, 0), sigmaX=1.2)
    return h_resp, v_resp


def _line_straightness_score(
    response: np.ndarray,
    *,
    horizontal: bool,
    top: int,
    bottom: int,
) -> float:
    """Measure how strongly line evidence concentrates on axis-aligned peaks."""
    height, width = response.shape[:2]
    left = round(width * 0.04)
    right = round(width * 0.96)
    top = max(0, min(top, height - 1))
    bottom = max(top + 1, min(bottom, height))
    if horizontal:
        profile = response[top:bottom, left:right].mean(axis=1)
    else:
        profile = response[top:bottom, left:right].mean(axis=0)
    if profile.size < 3:
        return 0.0
    baseline = float(np.percentile(profile, 25.0))
    signal = np.maximum(profile.astype(np.float32) - baseline, 0.0)
    total = float(signal.sum())
    if total <= 1e-9:
        return 0.0
    peak_count = max(1, round(signal.size * 0.08))
    return float(np.sort(signal)[-peak_count:].sum() / total)


def _line_response(gray: np.ndarray, *, horizontal: bool, kernel_ratio: float) -> np.ndarray:
    height, width = gray.shape[:2]
    if horizontal:
        kernel_size = max(9, round(height * kernel_ratio))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size))
    else:
        kernel_size = max(9, round(width * kernel_ratio))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
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


def _multi_slice_peaks(
    response: np.ndarray,
    *,
    axis: int,
    axis_start: int,
    axis_end: int,
    travel_start: int,
    travel_end: int,
    minimum_distance: int,
    minimum_response: float,
    maximum_lines: int,
    num_slices: int = 7,
) -> List[int]:
    """Find robust line seeds using median projection across multiple spatial slices."""
    h, w = response.shape
    t_span = max(1, travel_end - travel_start)
    step = max(1, t_span // (num_slices + 2))
    slice_profs: List[np.ndarray] = []
    if axis == 0:
        for s in range(1, num_slices + 1):
            c_s = travel_start + s * step
            c_e = min(travel_end, c_s + step)
            if c_e > c_s:
                slice_profs.append(np.mean(response[axis_start:axis_end, c_s:c_e], axis=1))
    else:
        for s in range(1, num_slices + 1):
            r_s = travel_start + s * step
            r_e = min(travel_end, r_s + step)
            if r_e > r_s:
                slice_profs.append(np.mean(response[r_s:r_e, axis_start:axis_end], axis=0))

    if not slice_profs:
        return []
    profile = np.median(slice_profs, axis=0)
    if profile.size < 3 or float(np.max(profile)) < minimum_response:
        return []

    profile_smooth = cv2.GaussianBlur(
        profile.astype(np.float32).reshape(-1, 1), (0, 0), sigmaX=2.0
    ).ravel()
    threshold = max(minimum_response, float(np.percentile(profile_smooth, 60.0)))
    candidates = [
        (axis_start + i, float(profile_smooth[i]))
        for i in range(1, len(profile_smooth) - 1)
        if profile_smooth[i] >= threshold
        and profile_smooth[i] >= profile_smooth[i - 1]
        and profile_smooth[i] >= profile_smooth[i + 1]
    ]
    candidates.sort(key=lambda item: item[1], reverse=True)
    selected: List[int] = []
    for pos, _ in candidates:
        if all(abs(pos - prev) >= minimum_distance for prev in selected):
            selected.append(pos)
            if len(selected) >= maximum_lines:
                break
    return sorted(selected)


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
            with warnings.catch_warnings():
                warnings.simplefilter("error", np.exceptions.RankWarning)
                coefficients = np.polyfit(points_t, points_a, deg=2)
        except (TypeError, ValueError, np.linalg.LinAlgError, np.exceptions.RankWarning):
            continue
        fitted = np.polyval(coefficients, points_t)
        residuals = np.abs(fitted - points_a)
        residual_limit = max(5.0, radius * 0.45)
        inlier_mask = residuals <= residual_limit
        if int(inlier_mask.sum()) < 4:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", np.exceptions.RankWarning)
                refit = np.polyfit(points_t[inlier_mask], points_a[inlier_mask], deg=2)
        except (TypeError, ValueError, np.linalg.LinAlgError, np.exceptions.RankWarning):
            continue
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


def _fill_seed_gaps(
    primary_seeds: Sequence[int],
    fallback_seeds: Sequence[int],
    minimum_distance: int,
) -> List[int]:
    """Fill gaps in Hough line seeds using profile peak candidates."""
    if not primary_seeds:
        return list(fallback_seeds)
    if len(primary_seeds) < 2:
        combined = sorted(set(list(primary_seeds) + list(fallback_seeds)))
        filtered: List[int] = []
        for s in combined:
            if not filtered or s - filtered[-1] >= minimum_distance:
                filtered.append(s)
        return filtered
    gaps = np.diff(primary_seeds)
    median_gap = float(np.median(gaps))
    filled = list(primary_seeds)
    for i in range(len(primary_seeds) - 1):
        gap = primary_seeds[i + 1] - primary_seeds[i]
        if gap >= 1.6 * median_gap:
            candidates = [
                s
                for s in fallback_seeds
                if primary_seeds[i] + minimum_distance
                <= s
                <= primary_seeds[i + 1] - minimum_distance
            ]
            filled.extend(candidates)
    filled.sort()
    result: List[int] = []
    for s in filled:
        if not result or s - result[-1] >= minimum_distance:
            result.append(s)
    return result


def _trim_leading_line_outliers(seeds: List[int], minimum_distance: int) -> List[int]:
    """Drop an isolated preamble before the regular table-row run."""
    if len(seeds) < 3:
        return seeds
    if seeds[1] - seeds[0] >= round(minimum_distance * 2.2):
        return seeds[1:]
    return seeds


def _fit_slope_trend(
    observations: Sequence[Tuple[float, float, float, float, float, float]],
    *,
    horizontal: bool,
    min_length: float,
) -> Optional[np.ndarray]:
    """Fit a 1D linear trend of line segment slope as a function of perpendicular coordinate."""
    slopes: List[Tuple[float, float, float]] = []
    for mid, length, x1, y1, x2, y2 in observations:
        if horizontal:
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) >= min_length:
                slopes.append((mid, dy / dx, length))
        else:
            dx = x2 - x1
            dy = y2 - y1
            if abs(dy) >= min_length:
                slopes.append((mid, dx / dy, length))
    if len(slopes) < 3:
        return None
    try:
        mids = np.array([s[0] for s in slopes], dtype=np.float64)
        sl_vals = np.array([s[1] for s in slopes], dtype=np.float64)
        weights = np.array([s[2] for s in slopes], dtype=np.float64)
        return np.polyfit(mids, sl_vals, 1, w=weights)
    except (ValueError, TypeError, np.linalg.LinAlgError):
        return None


def _track_curve(
    response: np.ndarray,
    seed: int,
    *,
    expected: Optional[np.ndarray] = None,
    travel_start: int,
    travel_end: int,
    radius: int,
    smoothness: float,
    anchor_penalty: float,
    minimum_response: float,
) -> Tuple[np.ndarray, float]:
    """Track one continuous response ridge using dynamic programming.

    When ``expected`` is provided, the DP band is dynamically centered along that
    guiding path (following document slant and curl), preventing tracker wander
    into adjacent rows or handwriting.
    """
    axis_size, travel_size = response.shape
    if expected is None:
        expected_arr = np.full(travel_size, float(seed), dtype=np.float32)
    else:
        expected_arr = np.asarray(expected, dtype=np.float32)

    t_start = max(0, min(travel_start, travel_size - 1))
    t_end = max(t_start + 1, min(travel_end, travel_size))
    t_len = t_end - t_start
    if t_len <= 0:
        return expected_arr.copy(), 0.0

    r = max(2, radius)
    deltas = np.arange(-r, r + 1, dtype=np.int32)
    k_states = len(deltas)

    xs = np.arange(t_start, t_end)
    ys = np.clip(
        np.rint(expected_arr[xs]).astype(int)[:, None] + deltas[None, :],
        0,
        axis_size - 1,
    )

    band = response[ys, xs[:, None]].astype(np.float32)
    anchor = anchor_penalty * (deltas**2)
    unary = -np.minimum(band, 85.0) / 25.0 + anchor[None, :]

    backpointer = np.zeros((t_len, k_states), dtype=np.int16)
    previous_cost = unary[0, :].copy()
    state_indices = np.arange(k_states, dtype=np.int32)
    max_state_transition = max(2, min(6, round(radius * 0.15)))
    transition_offsets = np.arange(
        -max_state_transition,
        max_state_transition + 1,
        dtype=np.int32,
    )[:, None]
    previous_index_matrix = state_indices[None, :] + transition_offsets
    valid_transitions = (previous_index_matrix >= 0) & (previous_index_matrix < k_states)
    clipped_previous_indices = np.clip(previous_index_matrix, 0, k_states - 1)
    for t in range(1, t_len):
        delta_y = ys[t][None, :] - ys[t - 1, clipped_previous_indices]
        transition = previous_cost[clipped_previous_indices] + smoothness * (delta_y**2)
        transition = np.where(valid_transitions, transition, np.inf)
        best_offset = np.argmin(transition, axis=0)
        best_previous = clipped_previous_indices[best_offset, state_indices]
        previous_cost = unary[t, :] + transition[best_offset, state_indices]
        backpointer[t, :] = best_previous.astype(np.int16)

    path = np.empty(t_len, dtype=np.int32)
    path[-1] = int(np.argmin(previous_cost))
    for t in range(t_len - 1, 0, -1):
        path[t - 1] = backpointer[t, path[t]]

    seg = ys[np.arange(t_len), path].astype(np.float32)
    curve = expected_arr.copy()
    curve[t_start:t_end] = seg
    if t_start > 0:
        curve[:t_start] = seg[0]
    if t_end < travel_size:
        curve[t_end:] = seg[-1]

    sampled = response[
        np.clip(np.round(seg).astype(int), 0, axis_size - 1), np.arange(t_start, t_end)
    ]
    support = float(np.mean(sampled >= minimum_response)) if t_len > 0 else 0.0
    return curve, support


def _smooth_curve(curve: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return curve
    smoothed = cv2.GaussianBlur(curve.reshape(1, -1), (0, 0), sigmaX=sigma).ravel()
    return np.asarray(smoothed, dtype=np.float32)


def _regularize_curve_family(
    curves: Sequence[np.ndarray],
    *,
    travel_sigma: float,
    family_sigma: float = 1.0,
) -> List[np.ndarray]:
    """Suppress independent tracker wobble while retaining smooth page curl.

    A physical sheet produces a coherent displacement field. Treating every
    row or column as an unrelated curve lets handwriting and intersections
    introduce local kinks that later shear content. We therefore smooth the
    displacement field both along each curve and across neighboring curves.
    """
    if not curves:
        return []
    stacked = np.asarray(curves, dtype=np.float32)
    targets = np.median(stacked, axis=1, keepdims=True)
    displacement = stacked - targets
    regularized = cv2.GaussianBlur(
        displacement,
        (0, 0),
        sigmaX=max(1.0, travel_sigma),
        sigmaY=max(0.5, family_sigma),
        borderType=cv2.BORDER_REPLICATE,
    )
    return [
        np.asarray(targets[index, 0] + regularized[index], dtype=np.float32)
        for index in range(stacked.shape[0])
    ]


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


def _inverse_map_1d(
    curves: Sequence[np.ndarray],
    targets: Sequence[float],
    axis_len: int,
    travel_len: int,
) -> np.ndarray:
    """Build an in-bounds inverse map with identity anchors at both borders."""
    inv = np.empty((axis_len, travel_len), dtype=np.float32)
    t_arr = np.asarray(targets, dtype=np.float32)
    if len(t_arr) == 0:
        return np.broadcast_to(
            np.arange(axis_len, dtype=np.float32)[:, None],
            (axis_len, travel_len),
        ).copy()

    axis_coords = np.arange(axis_len, dtype=np.float32)
    target_knots = np.r_[0.0, t_arr, float(axis_len - 1)]

    for t in range(travel_len):
        c_pts = np.asarray([c[t] for c in curves], dtype=np.float32)
        source_knots = np.r_[0.0, c_pts, float(axis_len - 1)]
        inv[:, t] = np.interp(axis_coords, target_knots, source_knots)

    return inv


def _build_inverse_maps(
    image_shape: Tuple[int, int],
    rows: Sequence[np.ndarray],
    row_targets: Sequence[float],
    columns: Sequence[np.ndarray],
    column_targets: Sequence[float],
    *,
    iterations: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build decoupled 2-stage inverse maps composed analytically into a single pass."""
    height, width = image_shape
    map1_y = _inverse_map_1d(rows, row_targets, height, width)
    col_inv = _inverse_map_1d(columns, column_targets, width, height)
    map2_x = col_inv.T
    dest_y = np.broadcast_to(np.arange(height, dtype=np.float32)[:, None], (height, width))
    composed_x = map2_x.astype(np.float32)
    composed_y = cv2.remap(
        map1_y,
        map2_x,
        dest_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).astype(np.float32)
    return composed_x, composed_y


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
    maximum_axis_step: float = 2.0,
    maximum_cross_step_limit: float = 2.0,
    minimum_jacobian_limit: float = 0.15,
    maximum_jacobian_limit: float = 4.0,
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
        and maximum_step <= maximum_axis_step
        and maximum_cross_step <= maximum_cross_step_limit
        and minimum_jacobian >= minimum_jacobian_limit
        and maximum_jacobian <= maximum_jacobian_limit
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


def _render_safe_result(
    image: np.ndarray,
    source_x: np.ndarray,
    source_y: np.ndarray,
    *,
    scale: float,
    policy: GridDewarpConfig,
    diagnostics: Dict[str, GridDiagnosticValue],
) -> GridDewarpResult:
    """Upscale, revalidate, and render one accepted analysis-resolution map."""
    source_height, source_width = image.shape[:2]
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
        maximum_axis_step=policy.maximum_axis_step,
        maximum_cross_step_limit=policy.maximum_cross_step,
        minimum_jacobian_limit=policy.minimum_jacobian,
        maximum_jacobian_limit=policy.maximum_jacobian,
    )
    if not safe_full:
        rejection_diagnostics = {
            key: value for key, value in diagnostics.items() if key != "reason"
        }
        return _not_applied(
            image,
            "unsafe_full_resolution_map",
            validMapRatio=round(valid_full, 4),
            maxDisplacementPx=round(full_displacement, 3),
            minimumMapStep=round(full_min_step, 4),
            maximumMapStep=round(full_max_step, 4),
            maximumCrossStep=round(full_cross_step, 4),
            minimumJacobian=round(full_min_jacobian, 4),
            maximumJacobian=round(full_max_jacobian, 4),
            **rejection_diagnostics,
        )
    remapped = cv2.remap(
        image,
        full_x,
        full_y,
        policy.interpolation,
        borderMode=policy.border_mode,
        borderValue=policy.border_value,
    )
    return GridDewarpResult(
        image=remapped,
        applied=True,
        diagnostics={
            "version": GRID_DEWARP_VERSION,
            "applied": True,
            "validMapRatio": round(valid_full, 4),
            "maxDisplacementPx": round(full_displacement, 3),
            "minimumMapStep": round(full_min_step, 4),
            "maximumMapStep": round(full_max_step, 4),
            "maximumCrossStep": round(full_cross_step, 4),
            "minimumJacobian": round(full_min_jacobian, 4),
            "maximumJacobian": round(full_max_jacobian, 4),
            "outputWidth": source_width,
            "outputHeight": source_height,
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
    _gray(image)
    analysis, scale = _analysis_image(image, policy.analysis_max_dimension)
    analysis_gray = _gray(analysis)
    height, width = analysis_gray.shape[:2]
    top = round(height * policy.roi_top_ratio)
    bottom = round(height * policy.roi_bottom_ratio)
    if bottom - top < 32:
        return _not_applied(image, "roi_too_small")

    horizontal_response, vertical_response = _line_response_decoupled(
        analysis_gray,
        horizontal_kernel_ratio=policy.horizontal_kernel_ratio,
        vertical_kernel_ratio=policy.vertical_kernel_ratio,
    )
    baseline_horizontal_score = _line_straightness_score(
        horizontal_response,
        horizontal=True,
        top=top,
        bottom=bottom,
    )
    baseline_vertical_score = _line_straightness_score(
        vertical_response,
        horizontal=False,
        top=top,
        bottom=bottom,
    )
    minimum_row_distance = max(5, round(height * policy.peak_min_distance_ratio))
    minimum_column_distance = max(5, round(width * policy.peak_min_distance_ratio))

    hough_row_seeds = _hough_seeds(
        analysis_gray,
        horizontal=True,
        axis_start=top,
        axis_end=bottom,
        travel_start=round(width * 0.04),
        travel_end=round(width * 0.96),
        minimum_distance=minimum_row_distance,
        maximum_lines=64,
    )
    profile_row_seeds = _profile_peaks(
        horizontal_response,
        axis_start=top,
        axis_end=bottom,
        travel_start=round(width * 0.04),
        travel_end=round(width * 0.96),
        minimum_distance=minimum_row_distance,
        minimum_response=policy.minimum_response,
        maximum_lines=64,
    )
    row_seeds = _fill_seed_gaps(hough_row_seeds, profile_row_seeds, minimum_row_distance)
    row_seeds = _trim_leading_line_outliers(row_seeds, minimum_row_distance)

    if len(row_seeds) < policy.minimum_horizontal_lines:
        return _not_applied(
            image,
            "insufficient_horizontal_lines",
            horizontalLines=len(row_seeds),
            verticalLines=0,
        )

    hough_row_obs = _hough_observations(
        analysis_gray,
        horizontal=True,
        axis_start=top,
        axis_end=bottom,
        travel_start=round(width * 0.04),
        travel_end=round(width * 0.96),
    )
    poly_row_slope = _fit_slope_trend(
        hough_row_obs, horizontal=True, min_length=max(30.0, width * 0.04)
    )

    x_coords = np.arange(width, dtype=np.float32)
    row_radius = max(4, round(height * policy.track_radius_ratio))
    if baseline_horizontal_score >= policy.minimum_hough_fast_path_score:
        rows, row_support = _fit_hough_curves(
            hough_row_obs,
            row_seeds,
            horizontal=True,
            travel_length=width,
            radius=row_radius,
        )
    else:
        rows, row_support = [], []
    if len(rows) < policy.minimum_horizontal_lines:
        rows = []
        row_support = []
        for seed in row_seeds:
            expected_row = (
                seed + (x_coords - (width / 2.0)) * float(np.polyval(poly_row_slope, seed))
                if poly_row_slope is not None
                else None
            )
            curve, support = _track_curve(
                horizontal_response,
                seed,
                expected=expected_row,
                travel_start=round(width * 0.03),
                travel_end=round(width * 0.97),
                radius=row_radius,
                smoothness=policy.track_smoothness,
                anchor_penalty=policy.track_anchor_penalty,
                minimum_response=policy.minimum_response,
            )
            rows.append(_smooth_curve(curve, max(2.0, height * policy.smoothing_sigma_ratio)))
            row_support.append(support)
    rows, row_support = _deduplicate_tracked_curves(
        rows,
        row_support,
        minimum_gap=max(8, round(min(height, width) * 0.012)),
    )
    if len(rows) < policy.minimum_horizontal_lines:
        return _not_applied(
            image,
            "insufficient_horizontal_lines",
            horizontalLines=len(rows),
            verticalLines=0,
        )

    rows = _regularize_curve_family(
        rows,
        travel_sigma=max(2.0, width * policy.smoothing_sigma_ratio),
    )
    rows, row_order_projection = _project_ordered_curves(rows, minimum_gap=0.25)
    maximum_projection = policy.maximum_order_projection_ratio * min(height, width)
    if row_order_projection > maximum_projection:
        return _not_applied(
            image,
            "crossing_grid_curves",
            horizontalLines=len(rows),
            verticalLines=0,
            orderProjectionPx=round(row_order_projection / max(scale, 1e-9), 3),
        )

    # Stage 1: Flatten rows to obtain intermediate step1 image
    row_targets = [float(np.median(curve)) for curve in rows]
    map1_y = _inverse_map_1d(rows, row_targets, height, width)
    dest_x = np.broadcast_to(np.arange(width, dtype=np.float32)[None, :], (height, width))
    (
        row_map_safe,
        row_valid_ratio,
        row_displacement,
        row_minimum_step,
        row_maximum_step,
        row_cross_step,
        row_minimum_jacobian,
        row_maximum_jacobian,
    ) = _evaluate_map(
        dest_x,
        map1_y,
        source_shape=(height, width),
        minimum_valid_ratio=policy.minimum_valid_map_ratio,
        maximum_axis_step=policy.maximum_axis_step,
        maximum_cross_step_limit=policy.maximum_cross_step,
        minimum_jacobian_limit=policy.minimum_jacobian,
        maximum_jacobian_limit=policy.maximum_jacobian,
    )
    if not row_map_safe:
        return _not_applied(
            image,
            "unsafe_row_map",
            horizontalLines=len(rows),
            verticalLines=0,
            validMapRatio=round(row_valid_ratio, 4),
            maxDisplacementPx=round(row_displacement / max(scale, 1e-9), 3),
            minimumMapStep=round(row_minimum_step, 4),
            maximumMapStep=round(row_maximum_step, 4),
            maximumCrossStep=round(row_cross_step, 4),
            minimumJacobian=round(row_minimum_jacobian, 4),
            maximumJacobian=round(row_maximum_jacobian, 4),
        )
    max_displacement = policy.max_displacement_ratio * min(height, width)
    if row_displacement > max_displacement:
        return _not_applied(
            image,
            "row_displacement_limit",
            horizontalLines=len(rows),
            verticalLines=0,
            validMapRatio=round(row_valid_ratio, 4),
            maxDisplacementPx=round(row_displacement / max(scale, 1e-9), 3),
        )
    step1_gray = cv2.remap(
        analysis_gray,
        dest_x,
        map1_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    step1_horizontal_response, step1_vertical_response = _line_response_decoupled(
        step1_gray,
        horizontal_kernel_ratio=policy.horizontal_kernel_ratio,
        vertical_kernel_ratio=policy.vertical_kernel_ratio,
    )
    row_straightness_score = _line_straightness_score(
        step1_horizontal_response,
        horizontal=True,
        top=top,
        bottom=bottom,
    )
    row_vertical_score = _line_straightness_score(
        step1_vertical_response,
        horizontal=False,
        top=top,
        bottom=bottom,
    )
    if (
        row_straightness_score < baseline_horizontal_score + policy.minimum_straightness_gain
        or row_vertical_score < baseline_vertical_score - policy.maximum_straightness_regression
    ):
        return _not_applied(
            image,
            "horizontal_straightness_not_improved",
            horizontalLines=len(rows),
            verticalLines=0,
            horizontalScoreBefore=round(baseline_horizontal_score, 4),
            horizontalScoreAfter=round(row_straightness_score, 4),
            verticalScoreBefore=round(baseline_vertical_score, 4),
            verticalScoreAfter=round(row_vertical_score, 4),
        )

    row_support_mean = float(np.mean(row_support)) if row_support else 0.0
    if row_support_mean < policy.minimum_support_ratio:
        return _not_applied(
            image,
            "weak_line_support",
            horizontalLines=len(rows),
            verticalLines=0,
            horizontalSupport=round(row_support_mean, 4),
            verticalSupport=None,
        )

    def render_rows_only(fallback_reason: str, vertical_lines: int = 0) -> GridDewarpResult:
        return _render_safe_result(
            image,
            dest_x,
            map1_y,
            scale=scale,
            policy=policy,
            diagnostics={
                "reason": "accepted_rows_only",
                "fallbackReason": fallback_reason,
                "appliedAxes": "rows",
                "horizontalLines": len(rows),
                "verticalLines": vertical_lines,
                "horizontalSupport": round(row_support_mean, 4),
                "verticalSupport": None,
                "orderProjectionPx": round(row_order_projection / max(scale, 1e-9), 3),
                "horizontalScoreBefore": round(baseline_horizontal_score, 4),
                "horizontalScoreAfter": round(row_straightness_score, 4),
                "verticalScoreBefore": round(baseline_vertical_score, 4),
                "verticalScoreAfter": round(row_vertical_score, 4),
                "analysisScale": round(scale, 6),
            },
        )

    # Stage 2: Track vertical columns on step1 where rows are horizontal
    _, v_resp_step1 = _line_response_decoupled(
        step1_gray,
        horizontal_kernel_ratio=policy.horizontal_kernel_ratio,
        vertical_kernel_ratio=policy.vertical_kernel_ratio,
    )
    hough_column_seeds = _hough_seeds(
        step1_gray,
        horizontal=False,
        axis_start=round(width * 0.04),
        axis_end=round(width * 0.96),
        travel_start=top,
        travel_end=bottom,
        minimum_distance=minimum_column_distance,
        maximum_lines=64,
    )
    profile_column_seeds = _profile_peaks(
        v_resp_step1.T,
        axis_start=round(width * 0.04),
        axis_end=round(width * 0.96),
        travel_start=top,
        travel_end=bottom,
        minimum_distance=minimum_column_distance,
        minimum_response=policy.minimum_response,
        maximum_lines=64,
    )
    column_seeds = _fill_seed_gaps(
        hough_column_seeds, profile_column_seeds, minimum_column_distance
    )

    if len(column_seeds) < policy.minimum_vertical_lines:
        return render_rows_only("insufficient_vertical_lines", len(column_seeds))

    hough_col_obs = _hough_observations(
        step1_gray,
        horizontal=False,
        axis_start=round(width * 0.04),
        axis_end=round(width * 0.96),
        travel_start=top,
        travel_end=bottom,
    )
    poly_col_slope = _fit_slope_trend(
        hough_col_obs, horizontal=False, min_length=max(30.0, height * 0.04)
    )

    y_coords = np.arange(height, dtype=np.float32)
    column_radius = max(4, round(width * policy.track_radius_ratio))
    columns, column_support = _fit_hough_curves(
        hough_col_obs,
        column_seeds,
        horizontal=False,
        travel_length=height,
        radius=column_radius,
    )
    if len(columns) < policy.minimum_vertical_lines:
        columns = []
        column_support = []
        for seed in column_seeds:
            expected_col = (
                seed + (y_coords - (height / 2.0)) * float(np.polyval(poly_col_slope, seed))
                if poly_col_slope is not None
                else None
            )
            curve, support = _track_curve(
                v_resp_step1.T,
                seed,
                expected=expected_col,
                travel_start=top,
                travel_end=bottom,
                radius=column_radius,
                smoothness=policy.track_smoothness,
                anchor_penalty=policy.track_anchor_penalty,
                minimum_response=policy.minimum_response,
            )
            columns.append(_smooth_curve(curve, max(2.0, width * policy.smoothing_sigma_ratio)))
            column_support.append(support)
    columns, column_support = _deduplicate_tracked_curves(
        columns,
        column_support,
        minimum_gap=max(8, round(min(height, width) * 0.012)),
    )
    if len(columns) < policy.minimum_vertical_lines:
        return render_rows_only("insufficient_vertical_lines", len(columns))

    columns = _regularize_curve_family(
        columns,
        travel_sigma=max(2.0, height * policy.smoothing_sigma_ratio),
    )
    columns, column_order_projection = _project_ordered_curves(columns, minimum_gap=0.25)
    if column_order_projection > maximum_projection:
        return render_rows_only("crossing_grid_curves", len(columns))

    column_support_mean = float(np.mean(column_support)) if column_support else 0.0
    if column_support_mean < policy.minimum_support_ratio:
        return render_rows_only("weak_vertical_line_support", len(columns))
    max_projection = max(row_order_projection, column_order_projection)

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
    combined_before = (baseline_horizontal_score + baseline_vertical_score) / 2.0
    max_displacement = policy.max_displacement_ratio * min(height, width)
    for column_blend in (1.0, 0.75, 0.5, 0.25):
        candidate_x = dest_x + column_blend * (source_x - dest_x)
        candidate_y = map1_y + column_blend * (source_y - map1_y)
        safe, _, displacement, _, _, _, _, _ = _evaluate_map(
            candidate_x,
            candidate_y,
            source_shape=(height, width),
            minimum_valid_ratio=policy.minimum_valid_map_ratio,
            maximum_axis_step=policy.maximum_axis_step,
            maximum_cross_step_limit=policy.maximum_cross_step,
            minimum_jacobian_limit=policy.minimum_jacobian,
            maximum_jacobian_limit=policy.maximum_jacobian,
        )
        if not safe or displacement > max_displacement:
            continue
        analysis_remapped = cv2.remap(
            analysis_gray,
            candidate_x,
            candidate_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        final_horizontal_response, final_vertical_response = _line_response_decoupled(
            analysis_remapped,
            horizontal_kernel_ratio=policy.horizontal_kernel_ratio,
            vertical_kernel_ratio=policy.vertical_kernel_ratio,
        )
        candidate_horizontal_score = _line_straightness_score(
            final_horizontal_response,
            horizontal=True,
            top=top,
            bottom=bottom,
        )
        candidate_vertical_score = _line_straightness_score(
            final_vertical_response,
            horizontal=False,
            top=top,
            bottom=bottom,
        )
        combined_after = (candidate_horizontal_score + candidate_vertical_score) / 2.0
        if (
            candidate_horizontal_score
            < row_straightness_score - policy.maximum_straightness_regression
            or candidate_vertical_score
            < baseline_vertical_score - policy.maximum_straightness_regression
            or combined_after < combined_before + policy.minimum_straightness_gain
        ):
            continue
        rendered = _render_safe_result(
            image,
            np.asarray(candidate_x, dtype=np.float32),
            np.asarray(candidate_y, dtype=np.float32),
            scale=scale,
            policy=policy,
            diagnostics={
                "reason": "accepted",
                "fallbackReason": None,
                "appliedAxes": ("rows_columns" if column_blend == 1.0 else "rows_columns_blended"),
                "columnBlend": round(column_blend, 2),
                "horizontalLines": len(rows),
                "verticalLines": len(columns),
                "horizontalSupport": round(row_support_mean, 4),
                "verticalSupport": round(column_support_mean, 4),
                "orderProjectionPx": round(max_projection / max(scale, 1e-9), 3),
                "horizontalScoreBefore": round(baseline_horizontal_score, 4),
                "horizontalScoreAfter": round(candidate_horizontal_score, 4),
                "verticalScoreBefore": round(baseline_vertical_score, 4),
                "verticalScoreAfter": round(candidate_vertical_score, 4),
                "analysisScale": round(scale, 6),
            },
        )
        if rendered.applied:
            return rendered

    return render_rows_only("column_quality_gate", len(columns))


__all__ = [
    "GRID_DEWARP_VERSION",
    "GridDewarpConfig",
    "GridDewarpResult",
    "dewarp_document_grid",
]
