# SmartDoc V2 Benchmark Import

AS-33 provides a deterministic importer for the public SmartDoc 2015 Challenge 1
dataset. The repository does not bundle the source frames. Materialize the
licensed archive locally, then run the importer against its `metadata.csv` or
`metadata.csv.gz` file.

## Source and attribution

- Dataset page: https://sites.google.com/site/icdar15smartdoc/challenge-1/dataset
- Automation-friendly format: https://github.com/jchazalon/smartdoc15-ch1-dataset
- License: CC BY 4.0.
- Attribution: Burie et al., ICDAR2015 Competition on Smartphone Document Capture
  and OCR (SmartDoc).

The upstream dataset contains smartphone-captured document frames with perspective,
blur, illumination changes, and partial occlusion. Its metadata includes the image
path and four document corner coordinates. The importer keeps those coordinates as
ground truth and converts them to the AS-32 post-EXIF coordinate convention.

## Reproducible import

```powershell
scanner\.venv\Scripts\python.exe scripts/import-smartdoc-benchmark.py `
  --source-root D:\Datasets\smartdoc15\frames `
  --metadata D:\Datasets\smartdoc15\frames\metadata.csv.gz `
  --output-root D:\Datasets\attendance-scanner\smartdoc-v1 `
  --count 250 `
  --seed 20260909
```

The command selects 250–300 samples by a stable seed, round-robins across the
selected model-type/background buckets, assigns whole capture sequences to one
of `train`, `tune`, or `acceptance`, copies only the selected images, and writes
`manifest.json` in the output root. Running the same command again produces the
same sample IDs, splits, corners, and manifest content without duplicating IDs.

The default selection includes `datasheet` and `tax`, matching the V2 plan's
preference for forms/tables. Use repeated `--model-type` or `--background`
arguments to make the selection policy explicit.

## Gap and release boundary

The full SmartDoc archive is not committed to this repository because it is a
large external dataset and must retain its license/attribution context. AS-33 is
complete when the importer, schema integration, deterministic policy, and license
instructions are present. A materialized 250–300-sample corpus requires the
source archive to be provided or downloaded by the operator. The importer fails
closed if metadata or a referenced frame is missing; it never fabricates ground
truth or silently substitutes synthetic samples.
