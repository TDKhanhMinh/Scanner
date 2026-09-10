# V2 Perspective Transform and Crop Safety

AS-50 adds `warp_perspective_v2()` as an additive, validated path over the
existing Perspective V1 API. It accepts canonical AS-36/AS-49 corners or
compatible point inputs, infers output dimensions from opposite edge lengths,
and never forces an A4 aspect ratio before warp.

## Safety policy

- source bounds are checked with configurable tolerance;
- non-finite, degenerate, too-small, too-large, and max-pixel outputs are
  rejected;
- optional destination safety margin is bounded and does not upscale the source
  to simulate DPI;
- `natural`, `landscape`, and `portrait` orientation policies are explicit;
- invalid transforms return a copied full-resolution source image with
  `fallback_used=true`, `PERSPECTIVE_V2_FALLBACK_FULL_IMAGE`, identity matrix,
  and a developer diagnostic instead of crashing the batch.

`WarpedDocument.diagnostics` records source/final dimensions, inferred-aspect
flag, safety margin, orientation rotation, and transform matrix. The source
image and existing V1 `warp_perspective()` behavior remain unchanged.

## Validation

Tests cover non-A4 trapezoid geometry, portrait/landscape orientation, near
border quads, safety margin/max-size guards, degenerate fallback, content
preservation, canonical inputs, and no-upscale behavior. Packaged and clean
Windows end-to-end evidence remains a later release gate.
