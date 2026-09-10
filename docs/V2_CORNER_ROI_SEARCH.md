# V2 Local Corner ROI Search

AS-47 adds local refinement evidence around the selected candidate corners. It
does not commit a refined corner or change perspective/export behavior.

## Strategy

`search_corner_rois(image, corners, config)` computes a radius from image
diagonal, clamped by configured min/max pixels, then processes exactly four
local regions. Each ROI runs grayscale/blur, Sobel gradient and Canny evidence.
For each corner, two expected side directions (toward the previous and next
canonical corners) define ray samples and a narrow normal band. This keeps the
search close to the predicted boundary and reduces the chance that a deep grid
line is selected as the paper edge.

The result contains clipped `roi_xywh`, bounded per-side `EdgeSample` points,
gradient/edge scores, best local evidence point, confidence, neighbor priors and
sample counts. A best point is evidence only; AS-48 decides whether/how to fit
lines and commit refined intersections.

## Border and runtime policy

ROI bounds are clipped to `[0,width) x [0,height)` before slicing, so corners at
or near a frame border cannot index outside the image. The source is never
mutated. Work is bounded to four local crops; there is no second full-image
scan. Radius, band, Canny and blur limits are versioned in
`CornerSearchConfig`.

When `debug_directory` is enabled, one collision-safe ROI overlay per corner is
written with detected samples highlighted. Debug files are artifacts, not
sidecar diagnostics or final output.

## Validation

Tests cover shifted-corner edge recovery, clipped border ROIs, grid-heavy
direction/position priors, bounded sample counts, debug overlays and source
immutability. Robust edge-line fitting and refined corner quality remain AS-48.
