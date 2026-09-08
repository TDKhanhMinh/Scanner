"""Deterministic, synthetic image corpus used by the scanner regression suite."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class FixtureCase:
    """One privacy-safe fixture and its expected pipeline outcome."""

    key: str
    relative_path: str
    expected_document_detected: bool
    expected_error_code: str | None = None
    builder: Callable[[Path], None] | None = None


def _save_bgr(path: Path, image: np.ndarray) -> None:
    """Save a BGR fixture through Pillow so Unicode paths work on Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(path)


def _document(
    path: Path,
    *,
    width: int = 800,
    height: int = 600,
    points: np.ndarray | None = None,
    background: int = 32,
    paper: int = 242,
    border: int = 20,
    border_width: int = 4,
) -> None:
    canvas = np.full((height, width, 3), background, dtype=np.uint8)
    corners = points if points is not None else np.array(
        [[100, 80], [700, 80], [700, 520], [100, 520]],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(canvas, corners, (paper, paper, paper))
    cv2.polylines(canvas, [corners], True, (border, border, border), border_width)
    cv2.line(canvas, tuple(corners[0]), tuple(corners[2]), (paper - 25,) * 3, 2)
    cv2.line(canvas, tuple(corners[1]), tuple(corners[3]), (paper - 25,) * 3, 2)
    _save_bgr(path, canvas)


def _shadow_document(path: Path) -> None:
    image = np.full((600, 800, 3), 42, dtype=np.uint8)
    shadow = np.array([[115, 95], [715, 95], [715, 535], [115, 535]], dtype=np.int32)
    cv2.fillConvexPoly(image, shadow, (12, 12, 12))
    corners = np.array([[100, 80], [700, 80], [700, 520], [100, 520]], dtype=np.int32)
    cv2.fillConvexPoly(image, corners, (235, 235, 235))
    cv2.polylines(image, [corners], True, (22, 22, 22), 4)
    _save_bgr(path, image)


def _low_contrast_document(path: Path) -> None:
    _document(path, background=105, paper=165, border=92, border_width=3)


def _weak_edges_document(path: Path) -> None:
    _document(path, background=118, paper=145, border=112, border_width=2)


def _portrait_document(path: Path) -> None:
    points = np.array([[90, 70], [510, 70], [510, 730], [90, 730]], dtype=np.int32)
    _document(path, width=600, height=800, points=points)


def _exif_rotated_document(path: Path) -> None:
    image = np.full((600, 800, 3), 35, dtype=np.uint8)
    corners = np.array([[100, 80], [700, 80], [700, 520], [100, 520]], dtype=np.int32)
    cv2.fillConvexPoly(image, corners, (242, 242, 242))
    cv2.polylines(image, [corners], True, (20, 20, 20), 4)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    exif = pil_image.getexif()
    exif[0x0112] = 6
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(path, "JPEG", quality=95, exif=exif)


def _corrupt(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not-a-valid-image-payload")


def fixture_cases() -> List[FixtureCase]:
    """Return the complete synthetic corpus definition in stable order."""
    return [
        FixtureCase("straight", "straight/straight.png", True, builder=_document),
        FixtureCase(
            "perspective_left",
            "perspective/perspective-left.png",
            True,
            builder=lambda path: _document(
                path,
                points=np.array([[135, 80], [700, 115], [680, 520], [95, 485]], dtype=np.int32),
            ),
        ),
        FixtureCase(
            "perspective_right",
            "perspective/perspective-right.png",
            True,
            builder=lambda path: _document(
                path,
                points=np.array([[100, 115], [665, 80], [705, 485], [120, 520]], dtype=np.int32),
            ),
        ),
        FixtureCase(
            "dark_background",
            "background/dark-document.png",
            True,
            builder=lambda path: _document(path, background=12),
        ),
        FixtureCase("mild_shadow", "lighting/mild-shadow.png", True, builder=_shadow_document),
        FixtureCase(
            "low_contrast",
            "lighting/low-contrast.png",
            True,
            builder=_low_contrast_document,
        ),
        FixtureCase(
            "weak_edges",
            "lighting/weak-edges.png",
            False,
            builder=_weak_edges_document,
        ),
        FixtureCase(
            "no_border",
            "fallback/no-detectable-border.png",
            False,
            builder=lambda path: _save_bgr(
                path, np.full((400, 600, 3), 128, dtype=np.uint8)
            ),
        ),
        FixtureCase("portrait", "orientation/portrait.png", True, builder=_portrait_document),
        FixtureCase("landscape", "orientation/landscape.png", True, builder=_document),
        FixtureCase(
            "exif_rotated",
            "orientation/exif-rotated.jpg",
            True,
            builder=_exif_rotated_document,
        ),
        FixtureCase(
            "unicode_filename",
            "unicode/Nguyễn Văn A/ảnh chấm công.png",
            True,
            builder=_document,
        ),
        FixtureCase(
            "corrupt_image",
            "errors/corrupt-image.jpg",
            False,
            expected_error_code="IMAGE_DECODE_FAILED",
            builder=_corrupt,
        ),
    ]


def create_fixture_corpus(root: Path) -> List[tuple[FixtureCase, Path]]:
    """Materialize the synthetic corpus below ``root`` and return its paths."""
    materialized: List[tuple[FixtureCase, Path]] = []
    for case in fixture_cases():
        if case.builder is None:
            raise AssertionError(f"Fixture {case.key} has no builder")
        path = root / case.relative_path
        case.builder(path)
        materialized.append((case, path))
    return materialized
