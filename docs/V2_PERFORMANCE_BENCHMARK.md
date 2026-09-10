# V2 Performance, RAM and Footprint Benchmark

AS-58 adds a reproducible report for desktop measurements: pipeline latency,
stage timing, peak memory, ONNX session lifecycle, model size/hash and packaged
sidecar size/hash.

## Run

```powershell
.\scanner\.venv\Scripts\python scripts\benchmark-v2-performance.py `
  --input-root C:\path\to\representative-images `
  --output C:\path\to\reports\v2-performance.json `
  --detector-mode classic `
  --limit 200
```

For ONNX lifecycle measurements, pass an approved model:

```powershell
.\scanner\.venv\Scripts\python scripts\benchmark-v2-performance.py `
  --input-root C:\path\to\representative-images `
  --output C:\path\to\reports\v2-performance-ai.json `
  --detector-mode ai_enhanced `
  --model C:\path\to\document-segmentation.onnx `
  --input-shape 1,3,224,224 `
  --iterations 20 `
  --packaged-artifact apps\desktop\src-tauri\binaries\attendance-scanner-sidecar-x86_64-pc-windows-msvc.exe
```

Use `--baseline <report.json>` to add mean/P95 latency and memory deltas.
Missing RSS/model/artifact observations remain `null`; the report never treats
unavailable instrumentation as zero or as a release pass.

## Guarantees measured by the report

- Pipeline inputs are iterated one path at a time and the report records
  `maxInflightImages=1`.
- ONNX service load/inference reuses one session and reports cold start,
  first inference, steady-state distribution and session creation count.
- Default ONNX configuration includes CPU execution and does not download from
  the network. GPU providers are optional and cannot replace the CPU baseline.
- Model and packaged artifact measurements are content-addressed by SHA-256.
- Stage timing separates load, detection, warp, enhancement and resize for the
  current scanner pipeline; V2 providers expose their own detailed timing in
  their detector contracts.

## Interpretation and release gates

Performance numbers are environment-specific. Compare before/after reports on
the same machine, Python/runtime, image corpus and worker settings. Accuracy
must be checked with the AS-35 benchmark in the same change; a faster report
does not approve a model if accuracy regresses. FP16/INT8 is intentionally not
enabled by this task and must wait for a locked FP32 accuracy baseline.

The repository has no approved production segmentation model in source control,
so model-specific inference and packaged model-size claims remain later manual
gates until an approved artifact is supplied.
