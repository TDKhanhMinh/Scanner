# Version 2 Benchmark Manifest

AS-32 defines the canonical benchmark format used by the V2 detector and future
benchmark runner. The machine-readable schema is
`schemas/benchmark_manifest.v1.json`; the executable validator is
`attendance_scanner.benchmark.BenchmarkManifest`.

## Coordinate contract

- `coordinate_space` is always `exif_normalized_pixels`.
- `width` and `height` describe the decoded image after EXIF orientation is applied.
- `tl`, `tr`, `br`, `bl` are pixel coordinates in clockwise canonical order.
- Visible-document corners must be inside `[0, width) x [0, height)`.
- An out-of-frame or occluded sample must set `document_visible=false` and provide
  `visibility_reason` as `border_touching`, `out_of_frame`, or `occluded`.
- Ground truth is never inferred from a filename.

## Sample fields

| Field | Required | Meaning |
| --- | --- | --- |
| `sample_id` | yes | Stable unique sample identifier. |
| `image_path` | yes | Relative path below the dataset root. |
| `dataset` | yes | Dataset or corpus name. |
| `split` | yes | `train`, `tune`, or `acceptance`. |
| `width`, `height` | yes | EXIF-normalized image dimensions. |
| `tl`, `tr`, `br`, `bl` | yes | Canonical four-corner ground truth. |
| `document_polygon` | no | Optional polygon for segmentation/mask evaluation. |
| `mask_path` | no | Optional relative mask path. |
| `document_visible` | yes | Whether the full document is visible. |
| `visibility_reason` | conditional | Required when the document is not fully visible. |
| `scenario_tags` | yes | Conditions such as `shadow`, `overlap`, or `grid-heavy`. |
| `source` | yes | Dataset name, license, attribution, and optional URL. |
| `sequence_id` | no | Shared capture/video sequence used for split leakage checks. |

## Split policy

The manifest uses `sequence_group`: all samples with the same `sequence_id` must
belong to one split. This prevents frames or near-duplicates from leaking between
training/tuning data and the acceptance set. Samples without a sequence ID still
require unique `sample_id` and `image_path` values.

## Validation

Schema-only validation:

```powershell
python -m attendance_scanner.benchmark path\to\manifest.json
```

Schema plus image/mask references and EXIF-normalized dimensions:

```powershell
python -m attendance_scanner.benchmark path\to\manifest.json --image-root path\to\dataset
```

The validator rejects missing fields, duplicate IDs/paths, unsafe relative paths,
invalid corner order, non-finite points, self-intersecting polygons, incorrect
visible-document bounds, missing visibility policy, split leakage, missing files,
and dimension mismatches.

AS-32 includes five valid synthetic samples and five invalid payload cases in
`scanner/tests/test_benchmark.py`. The fixtures are generated in memory and do
not contain attendance PII.
