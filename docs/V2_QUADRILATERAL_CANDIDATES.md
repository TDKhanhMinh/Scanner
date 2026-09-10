# V2 Multi-Source Quadrilateral Candidates

AS-44 builds a bounded quadrilateral pool before final fusion/refinement. The
primary path is selected document mask to quad fitting; CV contour and Hough
line evidence remain complementary sources. The builder does not choose a
final result and does not crop or warp any image.

## Candidate sources

`build_quadrilateral_candidates()` accepts any combination of:

- `mask_fit`: contour approximation and `minAreaRect` from the selected mask;
- `contour`: canonical candidates from AS-41;
- `hough`: bounded horizontal/vertical line combinations from AS-43;
- `mixed`: blended mask/contour geometry when both sources are available.

Every candidate carries canonical corners, source, bounded score, geometry
quality, optional `mask_quad_iou`, optional edge support and source evidence.
Mask fitting computes coverage IoU by rasterizing only the candidate polygon for
scoring; no source image is modified.

## Guardrails

Candidates pass the AS-40 geometry gate before entering the pool. Duplicate
quads are merged when corner distance or polygon IoU crosses configured limits;
the pool is then capped at `max_candidates`. Candidate IDs are reindexed after
deduplication in deterministic order. Tiny internal rectangles can remain as
`contour` evidence but retain area/geometry/mask-fit evidence for downstream
scoring to reject.

Candidates that fail the cheap AS-40 gate are not silently discarded: a bounded
`rejected_candidates` evidence pool records source, canonical corners when
available, reason codes, geometry quality and mask IoU. This lets AS-45 explain
why a tiny/internal distractor was rejected without allowing it into the valid
pool. When a mask is supplied at detection resolution, AS-39 transform metadata maps
it to the requested original image dimensions before fitting. Border-touching
quads are accepted when the geometry gate permits them, so clipped-sheet cases
can proceed to later refinement.

## Scope

AS-44 provides mask-only, contour-only, line-only and mixed-source candidate
construction plus `maskQuadIoU` evidence. Final ranking/fusion, edge refinement,
perspective warp and PDF output remain later stages.

## Validation

Tests cover mask-to-quad IoU, contour-only and Hough-only sources, mixed-source
blending, duplicate/cap behavior, border-compatible geometry, and
detection-size-to-original-size mapping.
