"""Generate visual and metric comparison artifacts for all enhancement modes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scanner" / "src"))

from attendance_scanner.enhancement_quality import (
    compare_enhancement_modes,
)
from attendance_scanner.pipeline.load import load_image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Artifact output directory")
    args = parser.parse_args()
    report = compare_enhancement_modes(load_image(args.input), output_dir=args.output)
    print(json.dumps(report.model_dump(by_alias=True), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
