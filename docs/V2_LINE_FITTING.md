# V2 Robust Document Edge-Line Fitting

AS-48 fits the four document boundary lines from AS-47 local edge samples and
AS-43 Hough evidence. It does not intersect lines into final corners; AS-49
owns that operation and quality gate.

## Fitting policy

For each `top/right/bottom/left` edge, the fitter combines the `next` samples
from the starting corner with `previous` samples from the ending corner. The
selected quad supplies the expected direction. A PCA line fit is followed by a
residual trim (`max_residual_px`) and one refit from inliers. Direction
deviation, inlier ratio/count, residual mean/P95 and support length become
confidence evidence.

If local support is insufficient, nearby direction-compatible Hough segments
can contribute points. If the fit remains weak, the original candidate edge is
returned as a finite, low-confidence `original_edge` fallback when enabled.
No error path throws for weak image evidence; it remains recoverable and
diagnostic.

## Safety

All line models store finite anchor/direction/normalized line coefficients and
bounded support metrics. No line is extended to an unbounded intersection in
this task. Direction tolerance prevents an internal grid line with a different
orientation from replacing the expected document edge.

## Validation

Tests cover four-edge local fitting, partial/noisy support, Hough fallback,
original-edge fallback, direction preservation, residual confidence and finite
line coefficients. Corner intersection/refinement and final acceptance remain
AS-49/AS-50.
