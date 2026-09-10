"""Product-safe detector mode names shared by CLI and runtime selection."""

from typing import Final, Literal, Optional

DetectorMode = Literal[
    "ai_enhanced",
    "classic",
    "v1_cv",
    "segmentation_only",
    "cv_v2",
    "hybrid",
    "docaligner_reference",
]
RuntimeDetectorMode = Literal[
    "v1_cv",
    "segmentation_only",
    "cv_v2",
    "hybrid",
    "docaligner_reference",
]

PRODUCT_DETECTOR_MODES: Final[tuple[str, ...]] = ("ai_enhanced", "classic")
DEVELOPMENT_DETECTOR_MODES: Final[tuple[str, ...]] = (
    "v1_cv",
    "segmentation_only",
    "cv_v2",
    "hybrid",
    "docaligner_reference",
)
ALL_DETECTOR_MODES: Final[tuple[str, ...]] = PRODUCT_DETECTOR_MODES + DEVELOPMENT_DETECTOR_MODES
DEFAULT_DETECTOR_MODE: Final[str] = "ai_enhanced"


def normalize_detector_mode(
    value: Optional[str],
    *,
    allow_development: bool = True,
) -> DetectorMode:
    """Normalize a detector mode and reject unknown values before processing starts."""
    normalized = (value or DEFAULT_DETECTOR_MODE).strip().lower().replace("-", "_")
    valid_modes = ALL_DETECTOR_MODES if allow_development else PRODUCT_DETECTOR_MODES
    if normalized not in valid_modes:
        choices = ", ".join(valid_modes)
        raise ValueError(f"Unsupported detector mode: {value!r}. Expected one of: {choices}")
    return normalized  # type: ignore[return-value]


def internal_detector_mode(mode: DetectorMode) -> RuntimeDetectorMode:
    """Map product labels to the provider-neutral runtime factory names."""
    if mode == "ai_enhanced":
        return "hybrid"
    if mode == "classic":
        return "v1_cv"
    return mode


def detector_name_for_mode(mode: DetectorMode) -> str:
    """Return the stable detector family name persisted in manifest metadata."""
    return "hybrid" if mode == "ai_enhanced" else "v1_cv" if mode == "classic" else mode


__all__ = [
    "ALL_DETECTOR_MODES",
    "DEFAULT_DETECTOR_MODE",
    "DEVELOPMENT_DETECTOR_MODES",
    "DetectorMode",
    "PRODUCT_DETECTOR_MODES",
    "RuntimeDetectorMode",
    "detector_name_for_mode",
    "internal_detector_mode",
    "normalize_detector_mode",
]
