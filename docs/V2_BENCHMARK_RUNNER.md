# V2 Benchmark Metrics and Report Runner

AS-35 provides a benchmark runner independent of the desktop UI. It reads the
AS-32 manifest, evaluates one or more datasets, and writes deterministic
machine-readable and reviewer-friendly reports. It does not train a model or
silently invent a prediction when an adapter has no output.

## Detector adapters

The runner uses one prediction contract for all adapter names:

| Adapter | Input | Purpose |
| --- | --- | --- |
| `v1_cv` | built-in | Execute the current classical OpenCV detector. |
| `reference` | built-in | Oracle ground truth for metric and report sanity checks; not a latency baseline. |
| `segmentation_only` | prediction JSON | Score a segmentation model's exported predictions. |
| `cv_v2` | prediction JSON | Score the V2 CV/refinement adapter. |
| `hybrid` | prediction JSON | Score a segmentation + CV adapter. |

`segmentation_only`, `cv_v2`, and `hybrid` intentionally require a prediction
bundle because those model runtimes are delivered by later V2 tasks. This keeps
AS-35 useful now without claiming that an unimplemented detector has been
benchmarked.

## Prediction contract

The file can be a bare JSON array or a versioned object:

```json
{
  "schema_version": "1.0",
  "detector": "hybrid",
  "model_version": "hybrid-v2.0.0",
  "pipeline_version": "scanner-v2",
  "predictions": [
    {
      "sample_id": "timesheet-v1:sheet-001.jpg",
      "detected": true,
      "corners": [[10, 10], [1190, 12], [1188, 1680], [12, 1678]],
      "polygon": [[10, 10], [1190, 12], [1188, 1680], [12, 1678]],
      "mask_path": "masks/sheet-001.png",
      "timings_ms": {
        "segmentation_inference_ms": 12.4,
        "mask_postprocess_ms": 1.2,
        "quadrilateral_fitting_ms": 0.8,
        "cv_refinement_ms": 2.1,
        "total_detection_ms": 17.0
      },
      "memory_peak_bytes": 10485760
    }
  ]
}
```

Coordinates use the manifest's EXIF-normalized pixel space and canonical
`TL/TR/BR/BL` order. Missing sample predictions are reported as
`missing_prediction`; invalid or non-canonical quads are reported as
`invalid_geometry` instead of aborting the whole report. Prediction mask paths
must be relative to that sample's image root and cannot escape it.

## Metrics

The JSON report contains one per-sample record and aggregates:

- detection success/failure and failure taxonomy;
- convex polygon IoU;
- per-corner Euclidean error in pixels and normalized by image diagonal;
- mean, P50, P95, and P99 corner error;
- invalid-geometry/false-detection count and rate;
- optional mask IoU, predicted mask coverage, connected-component count, and
  component ambiguity rate when masks are present;
- observed segmentation, postprocess, quadrilateral fitting, CV refinement, and
  total detection timings, each with count/mean/P50/P95/P99;
- observed peak Python allocation memory when the adapter supplies it.

Timing and memory distributions exclude missing observations rather than
turning absent instrumentation into fake zero values. The `reference` adapter
is an accuracy sanity check and intentionally supplies no latency evidence.

Every report records detector/model/pipeline versions, Git `HEAD`, a SHA-256 of
the canonical combined manifest, and runtime facts (Python, platform, CPU
count, OpenCV/NumPy versions, and OpenCV thread count). No timestamp is added,
so identical inputs/configuration produce stable report metadata; measured
latency and memory are expected to vary within normal runtime tolerance.

## Run examples

Score the current V1 detector and write all report formats:

```powershell
python scripts/run-v2-benchmark.py `
  --manifest D:\datasets\timesheet-domain-v1\manifest.json `
  --image-root D:\datasets\timesheet-domain-v1 `
  --detector v1_cv `
  --json-output D:\reports\v1-cv.json `
  --markdown-output D:\reports\v1-cv.md `
  --csv-output D:\reports\v1-cv.csv
```

Run the oracle sanity check:

```powershell
python scripts/run-v2-benchmark.py `
  --manifest D:\datasets\timesheet-domain-v1\manifest.json `
  --image-root D:\datasets\timesheet-domain-v1 `
  --detector reference `
  --json-output D:\reports\reference.json
```

Score a future model adapter through the same schema:

```powershell
python scripts/run-v2-benchmark.py `
  --manifest D:\datasets\timesheet-domain-v1\manifest.json `
  --image-root D:\datasets\timesheet-domain-v1 `
  --detector hybrid `
  --predictions D:\runs\hybrid\predictions.json `
  --json-output D:\reports\hybrid.json
```

Combined datasets use repeated paired `--manifest` and `--image-root` options:

```powershell
python scripts/run-v2-benchmark.py `
  --manifest D:\datasets\smartdoc\manifest.json `
  --image-root D:\datasets\smartdoc `
  --manifest D:\datasets\timesheet-domain-v1\manifest.json `
  --image-root D:\datasets\timesheet-domain-v1 `
  --detector reference `
  --json-output D:\reports\combined-reference.json
```

Enable the tracked release gate with `--gate`. It checks detection success
`>=98%`, mean IoU `>=0.95`, P95 normalized corner error `<=1%` of the image
diagonal, and invalid-geometry false-detection rate `<=1%`. Custom gates can
be supplied with `--min-success-rate`, `--min-mean-iou`,
`--max-p95-normalized-corner-error`, and `--max-false-detection-rate`.
The process exits `0` when no gate is configured or all configured checks pass,
`2` when a gate fails, and `1` for invalid input/runner errors.

## Current boundary

AS-35 supplies the measurement/reporting contract and a working V1/reference
path. It does not close the later model-runtime, real 200–300-image corpus,
packaged desktop, or clean-environment release gates. Those require materialized
approved data and detector adapters from subsequent tasks.
