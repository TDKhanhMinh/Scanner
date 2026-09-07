"""Unit tests for the scanner pipeline image loader, EXIF orientation, and safe decoding."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from attendance_scanner.contracts import ImageDecodeError, ScannerErrorCode
from attendance_scanner.pipeline.load import (
    LoadedImage,
    LoadedImageMetadata,
    load_image,
)


def create_test_image_with_exif(
    path: Path,
    size: tuple[int, int],
    color: tuple[int, int, int],
    orientation: int,
) -> None:
    """Helper to create a JPEG image with a specific EXIF orientation tag and distinct landmark."""
    img = Image.new("RGB", size, color=color)
    # Mark top-left 3x3 block with a distinct green landmark (R=0, G=255, B=0)
    for y in range(3):
        for x in range(3):
            img.putpixel((x, y), (0, 255, 0))

    exif = img.getexif()
    exif[0x0112] = orientation
    img.save(path, "JPEG", quality=100, exif=exif)


def test_load_exif_orientations(tmp_path: Path):
    """Verify EXIF orientations 1, 3, 6, and 8 are transposed upright with pixel landmarks."""
    # Test case: original image is 100 wide x 60 high with dark background
    orig_w, orig_h = 100, 60
    base_color = (10, 10, 10)

    # Orientation 1: Normal (0 deg rotation)
    # Landmark at original top-left (x=0..2, y=0..2) remains at top-left (y=1, x=1)
    p1 = tmp_path / "img_exif_1.jpg"
    create_test_image_with_exif(p1, (orig_w, orig_h), base_color, orientation=1)
    res1 = load_image(p1)
    assert isinstance(res1, LoadedImage)
    assert res1.width == orig_w
    assert res1.height == orig_h
    assert res1.metadata.original_width == orig_w
    assert res1.metadata.original_height == orig_h
    assert res1.metadata.exif_orientation == 1
    assert res1.shape == (orig_h, orig_w, 3)
    assert res1.channels == 3
    # Pixel landmark check: top-left is green (high G, low B and R)
    pix1_landmark = res1.image[1, 1]
    assert pix1_landmark[1] > 180 and pix1_landmark[0] < 80 and pix1_landmark[2] < 80
    # Opposite corner (bottom-right) should be base dark color
    pix1_opposite = res1.image[orig_h - 2, orig_w - 2]
    assert pix1_opposite[1] < 50

    # Orientation 3: 180 degree rotation
    # Landmark at original top-left rotates 180 deg to bottom-right (y=orig_h - 2, x=orig_w - 2)
    p3 = tmp_path / "img_exif_3.jpg"
    create_test_image_with_exif(p3, (orig_w, orig_h), base_color, orientation=3)
    res3 = load_image(p3)
    assert res3.width == orig_w
    assert res3.height == orig_h
    assert res3.metadata.exif_orientation == 3
    assert res3.shape == (orig_h, orig_w, 3)
    pix3_landmark = res3.image[orig_h - 2, orig_w - 2]
    assert pix3_landmark[1] > 180 and pix3_landmark[0] < 80 and pix3_landmark[2] < 80
    pix3_opposite = res3.image[1, 1]
    assert pix3_opposite[1] < 50

    # Orientation 6: Rotated 90 CW in camera -> upright transpose rotates 90 CCW
    # Dimensions swap: 100x60 becomes 60x100
    # Original top-left landmark moves to top-right: y=1, x=norm_w - 2 (58)
    p6 = tmp_path / "img_exif_6.jpg"
    create_test_image_with_exif(p6, (orig_w, orig_h), base_color, orientation=6)
    res6 = load_image(p6)
    assert res6.metadata.original_width == orig_w
    assert res6.metadata.original_height == orig_h
    assert res6.width == orig_h  # Dimensions swapped: 60
    assert res6.height == orig_w  # Dimensions swapped: 100
    assert res6.metadata.exif_orientation == 6
    assert res6.shape == (orig_w, orig_h, 3)
    pix6_landmark = res6.image[1, orig_h - 2]  # y=1, x=58
    assert pix6_landmark[1] > 180 and pix6_landmark[0] < 80 and pix6_landmark[2] < 80
    pix6_opposite = res6.image[orig_w - 2, 1]  # y=98, x=1
    assert pix6_opposite[1] < 50

    # Orientation 8: Rotated 270 CW in camera -> upright transpose rotates 90 CW
    # Dimensions swap: 100x60 becomes 60x100
    # Original top-left landmark moves to bottom-left: y=norm_h - 2 (98), x=1
    p8 = tmp_path / "img_exif_8.jpg"
    create_test_image_with_exif(p8, (orig_w, orig_h), base_color, orientation=8)
    res8 = load_image(p8)
    assert res8.metadata.original_width == orig_w
    assert res8.metadata.original_height == orig_h
    assert res8.width == orig_h  # Dimensions swapped: 60
    assert res8.height == orig_w  # Dimensions swapped: 100
    assert res8.metadata.exif_orientation == 8
    assert res8.shape == (orig_w, orig_h, 3)
    pix8_landmark = res8.image[orig_w - 2, 1]  # y=98, x=1
    assert pix8_landmark[1] > 180 and pix8_landmark[0] < 80 and pix8_landmark[2] < 80
    pix8_opposite = res8.image[1, orig_h - 2]  # y=1, x=58
    assert pix8_opposite[1] < 50


def test_alpha_transparency_composited_onto_white(tmp_path: Path):
    """Verify transparent and semi-transparent pixels are composited onto pure white."""
    p = tmp_path / "transparent_sample.png"

    # Create 40x40 RGBA image
    # Top half (y < 20): solid red (R=255, G=0, B=0, A=255)
    # Bottom half (y >= 20): fully transparent (R=0, G=0, B=255, A=0)
    rgba_img = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
    for y in range(40):
        for x in range(40):
            if y < 20:
                rgba_img.putpixel((x, y), (255, 0, 0, 255))
            else:
                rgba_img.putpixel((x, y), (0, 0, 255, 0))
    rgba_img.save(p)

    loaded = load_image(p)
    assert loaded.metadata.has_transparency is True
    assert loaded.metadata.original_mode == "RGBA"
    assert loaded.metadata.normalized_mode == "BGR"

    # Top half should be red in BGR: B=0, G=0, R=255
    top_pixel = loaded.image[5, 5]
    np.testing.assert_array_equal(top_pixel, [0, 0, 255])

    # Bottom half was transparent (A=0) -> must be composited onto pure white (255, 255, 255)
    bottom_pixel = loaded.image[25, 25]
    np.testing.assert_array_equal(bottom_pixel, [255, 255, 255])


def test_grayscale_and_palette_modes(tmp_path: Path):
    """Verify Grayscale ('L') and Palette ('P') images are normalized to 3-channel BGR."""
    # Grayscale image
    p_gray = tmp_path / "sample_gray.png"
    gray_img = Image.new("L", (30, 20), color=128)
    gray_img.save(p_gray)

    loaded_gray = load_image(p_gray)
    assert loaded_gray.channels == 3
    assert loaded_gray.shape == (20, 30, 3)
    assert loaded_gray.metadata.original_mode == "L"
    np.testing.assert_array_equal(loaded_gray.image[0, 0], [128, 128, 128])

    # Palette image with transparency
    p_pal = tmp_path / "sample_pal.png"
    pal_img = Image.new("P", (30, 20))
    pal_img.putpalette([255, 255, 255, 255, 0, 0, 0, 255, 0] + [0] * 759)
    pal_img.info["transparency"] = 0
    pal_img.save(p_pal)

    loaded_pal = load_image(p_pal)
    assert loaded_pal.channels == 3
    assert loaded_pal.shape == (20, 30, 3)


def test_supported_formats_and_case_insensitivity(tmp_path: Path):
    """Verify JPG, JPEG, PNG, WEBP load cleanly regardless of extension case."""
    formats = [
        ("test.jpg", "JPEG"),
        ("test.JPEG", "JPEG"),
        ("test.png", "PNG"),
        ("test.PNG", "PNG"),
        ("test.webp", "WEBP"),
        ("test.WebP", "WEBP"),
    ]

    for fname, fmt in formats:
        img_path = tmp_path / fname
        img = Image.new("RGB", (20, 15), color=(50, 100, 150))
        img.save(img_path, fmt)

        loaded = load_image(img_path)
        assert loaded.width == 20
        assert loaded.height == 15
        assert loaded.image.dtype == np.uint8


def test_source_file_is_not_mutated(tmp_path: Path):
    """Verify the source file on disk is strictly unmutated (hash and mtime unchanged)."""
    p = tmp_path / "source_check.png"
    img = Image.new("RGBA", (50, 50), color=(100, 150, 200, 128))
    img.save(p)

    initial_mtime = p.stat().st_mtime_ns
    initial_hash = hashlib.sha256(p.read_bytes()).hexdigest()

    loaded = load_image(p)
    assert loaded.width == 50

    after_mtime = p.stat().st_mtime_ns
    after_hash = hashlib.sha256(p.read_bytes()).hexdigest()

    assert initial_hash == after_hash
    assert initial_mtime == after_mtime


def test_error_non_existent_file(tmp_path: Path):
    """Verify ImageDecodeError when file does not exist."""
    missing = tmp_path / "does_not_exist.jpg"
    with pytest.raises(ImageDecodeError) as exc_info:
        load_image(missing)
    assert exc_info.value.code == ScannerErrorCode.IMAGE_DECODE_FAILED
    assert "does not exist" in exc_info.value.message.lower()


def test_error_directory_path(tmp_path: Path):
    """Verify ImageDecodeError when path is a directory."""
    dir_path = tmp_path / "folder.png"
    dir_path.mkdir()
    with pytest.raises(ImageDecodeError) as exc_info:
        load_image(dir_path)
    assert exc_info.value.code == ScannerErrorCode.IMAGE_DECODE_FAILED


def test_error_empty_file(tmp_path: Path):
    """Verify ImageDecodeError when file is 0 bytes."""
    empty_file = tmp_path / "empty.jpg"
    empty_file.touch()
    with pytest.raises(ImageDecodeError) as exc_info:
        load_image(empty_file)
    assert exc_info.value.code == ScannerErrorCode.IMAGE_DECODE_FAILED
    assert "empty" in exc_info.value.message.lower()


def test_error_corrupted_file(tmp_path: Path):
    """Verify ImageDecodeError on corrupted / garbage image file."""
    corrupt_file = tmp_path / "corrupt.jpg"
    corrupt_file.write_bytes(
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01NOTANIMAGEGARBAGE"
    )
    with pytest.raises(ImageDecodeError) as exc_info:
        load_image(corrupt_file)
    assert exc_info.value.code == ScannerErrorCode.IMAGE_DECODE_FAILED


def test_error_unsupported_extension(tmp_path: Path):
    """Verify ImageDecodeError for unsupported file types (e.g. .txt, .pdf, .gif)."""
    txt_file = tmp_path / "document.txt"
    txt_file.write_text("not an image", encoding="utf-8")
    with pytest.raises(ImageDecodeError) as exc_info:
        load_image(txt_file)
    assert exc_info.value.code == ScannerErrorCode.IMAGE_DECODE_FAILED
    assert "unsupported" in exc_info.value.message.lower()


def test_loaded_image_metadata_contract():
    """Verify LoadedImageMetadata Pydantic model serialization and fields."""
    meta = LoadedImageMetadata(
        original_width=100,
        original_height=200,
        width=200,
        height=100,
        original_mode="RGBA",
        normalized_mode="BGR",
        exif_orientation=6,
        has_transparency=True,
    )
    data = meta.model_dump(by_alias=True)
    assert data["originalWidth"] == 100
    assert data["originalHeight"] == 200
    assert data["width"] == 200
    assert data["height"] == 100
    assert data["exifOrientation"] == 6
    assert data["hasTransparency"] is True
