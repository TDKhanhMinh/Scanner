# V2 Detection Preview

AS-55 adds a bounded diagnostic preview for files that emit a scan warning or
when diagnostics are enabled. The preview is evidence for support/debugging;
it is not a manual corner editor and does not change the batch result.

## Contract

`DetectionPreview` uses the normalized source-image coordinate space:

- `sourceWidth` / `sourceHeight` identify the coordinate canvas;
- `finalCorners` is the selected crop quad, when one exists;
- `candidateCorners` contains available provider candidates;
- `refinedCorners` is optional and is shown only when a provider supplies it;
- `maskAvailable` and `maskOverlayUrl` are optional evidence fields;
- `confidence` is presented as a quality score, never as a calibrated probability;
- `fallbackUsed`, reason codes, warning codes, detector and model metadata remain
  explicit and separate from user-facing explanations.

Raw masks, tensors, traces and filesystem paths are not serialized into the
preview contract. A future mask overlay must be a materialized `data:image/*`
payload.

## Bounded rendering

The sidecar encodes a JPEG preview with a maximum dimension of 1280 pixels only
for warning/diagnostic results. The desktop panel consumes that bounded data URL
and does not fall back to loading the original source image. If the preview
artifact is missing, the panel explains that state without blocking the batch.

The SVG overlay uses a normalized `0..100` viewBox and clamps source points to
the preview bounds. This keeps portrait and landscape images aligned without
depending on CSS pixel dimensions.

## Operator behavior

The result list opens the in-app preview only when bounded diagnostic data is
available. A result without preview data keeps the existing external source-file
preview action. Full-image fallback is labeled clearly as “không auto-crop”.
Manual drag/edit of corners remains outside V2.0 scope.
