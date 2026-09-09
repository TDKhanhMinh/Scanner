"""Materialize a deterministic SmartDoc subset for the AS-32 manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from attendance_scanner.benchmark_import import (
    SmartDocImportConfig,
    import_smartdoc_subset,
    summarize_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import a reproducible SmartDoc benchmark subset without bundling the dataset"
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument(
        "--model-type",
        action="append",
        dest="model_types",
        default=None,
        help="Eligible SmartDoc model type; repeat for multiple types (default: datasheet,tax)",
    )
    parser.add_argument(
        "--background",
        action="append",
        dest="backgrounds",
        default=None,
        help="Eligible SmartDoc background; repeat for multiple backgrounds (default: all)",
    )
    args = parser.parse_args()
    if not 250 <= args.count <= 300:
        parser.error("--count must be between 250 and 300 for the AS-33 corpus")

    manifest = import_smartdoc_subset(
        args.source_root,
        args.output_root,
        metadata_path=args.metadata,
        config=SmartDocImportConfig(
            target_count=args.count,
            seed=args.seed,
            model_types=tuple(args.model_types or ("datasheet", "tax")),
            backgrounds=tuple(args.backgrounds) if args.backgrounds else None,
        ),
    )
    print(json.dumps(summarize_manifest(manifest), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
