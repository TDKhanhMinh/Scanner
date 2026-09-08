"""Create deterministic, non-PII image fixtures for the packaged sidecar smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    employee_root = args.output / "NV01"
    employee_root.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (96, 72), (180, 180, 180))
    image.save(employee_root / "01.jpg")
    image.save(employee_root / "02.png")
    image.save(employee_root / "03.webp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
