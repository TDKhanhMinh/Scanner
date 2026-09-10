# V2 Detector Modes and Desktop Controls

AS-54 separates the operator-facing document detector choice from the existing
image enhancement filter. `--mode` continues to mean `gray`, `bw`, `color`, or
`smart_document`; `--detector-mode` controls how the four document corners are
found.

## Product modes

- `ai_enhanced` is the default and selects the provider-neutral hybrid path:
  segmentation evidence, quadrilateral fitting, OpenCV refinement, and safe
  fallback when evidence is unavailable.
- `classic` selects the existing OpenCV-only V1 path and is the explicit
  operator fallback.

The desktop UI shows only **AI Enhanced** and **Classic**. Provider, model,
variant, and version stay in diagnostics/manifest metadata and are not normal
operator choices.

The runtime accepts an injected configured provider through the Python library
`run_batch(..., detector=provider, detector_mode="ai_enhanced")`. If no
segmentation provider is configured, the hybrid policy records that condition
and uses its bounded CV rescue/fallback behavior; it never claims a model that
is not installed. Approved model artifacts remain a separate packaging gate.

## Development modes

The CLI additionally accepts `v1_cv`, `segmentation_only`, `cv_v2`, `hybrid`,
and `docaligner_reference` for benchmark and integration work. These values are
not displayed in the normal desktop selector.

Examples:

```text
attendance-scanner-sidecar plan --input <folder>
attendance-scanner-sidecar plan --input <folder> --detector-mode classic
attendance-scanner-sidecar scan-batch --input <folder> --detector-mode ai_enhanced --debug-diagnostics
attendance-scanner-sidecar scan-batch --input <folder> --reprocess
```

`--debug-diagnostics` adds compact, scalar decision facts to the in-memory
result metadata. Raw masks and model tensors remain outside the JSONL and
manifest contracts. `--reprocess` is explicit and one-shot: it is not persisted
across application restarts.

## Desktop persistence

The selected product detector and diagnostics toggle are stored in the
versioned local preference key `attendance-scanner.detector-settings.v1`.
Reprocess is always reset to off when the app starts, so an old output cannot be
rebuilt accidentally after reopening the app. The scan plan displays the
`needs_reprocess` count before the operator opts in.
