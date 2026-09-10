# V2 Detector Decision Benchmark

AS-59 compares the required detector adapters through one reproducible decision
harness. It does not select a production model from a handful of demo images.

## Run

```powershell
.\scanner\.venv\Scripts\python scripts\run-v2-decision.py `
  --manifest D:\datasets\timesheet\manifest.json `
  --image-root D:\datasets\timesheet `
  --predictions segmentation_only=D:\runs\segmentation\predictions.json `
  --predictions cv_v2=D:\runs\cv-v2\predictions.json `
  --predictions hybrid=D:\runs\hybrid\predictions.json `
  --performance hybrid=D:\reports\hybrid-performance.json `
  --output D:\reports\v2-decision.json `
  --markdown-output D:\reports\v2-decision.md
```

`v1_cv` and `reference` are built in. Prediction bundles are required for
`segmentation_only`, `cv_v2` and `hybrid`; missing bundles produce a report with
exit code `2` and explicit remediation instead of a false decision.

## Report contents

The report records the canonical dataset manifest hash, Git commit, model
version/checksum and prediction-bundle hash, per-adapter accuracy metrics,
silent-wrong-crop count, failure taxonomy, scenario breakdown and optional
AS-58 performance facts. Scenario tags should cover grid-heavy, low-contrast,
paper overlap, border touching, shadow, strong perspective and blur.

Hybrid can be selected only when all required evidence is present, its release
quality gate passes, and it improves on the V1 baseline. Otherwise the decision
is `BLOCKED`; targets are never silently lowered. DocAligner/reference remains
an explicit comparator and is never promoted by the harness.
