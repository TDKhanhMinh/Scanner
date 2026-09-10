# V2 Hough-Line Evidence

AS-43 adds a bounded Hough line evidence source for cases where a document edge
is broken, occluded, or touches the image border. It does not decide a final
quadrilateral; later fusion/refinement stages consume its segments and clusters.

## Pipeline

`detect_hough_lines(image, config)` reuses the detection-resolution edge policy:

1. grayscale, blur, Canny;
2. bounded morphology close/dilate;
3. probabilistic `HoughLinesP` with configured threshold, minimum length and
   maximum gap;
4. endpoint normalization to original pixels;
5. angle normalization to `[0, 180)`, orientation classification and edge
   support measurement;
6. deterministic top-segment limit and near-parallel/near-distance clustering.

`HoughLineEvidence` records source/detection dimensions, scale factor, config,
segments, clusters and raw/kept counts. Each `HoughLineSegment` includes source
endpoints, normalized angle, length, support, orientation, border-contact edges
and cluster ID. Segment and cluster limits prevent grid-heavy forms from
creating an unbounded evidence set.

## Border extrapolation

`extrapolate_line_to_bounds()` extends a segment along its normalized direction
to the image rectangle when intersections are available. Extension is capped by
`max_extrapolation_ratio`; a limited or already-border segment remains safe and
is marked with `extrapolated` evidence. The function returns line evidence only;
it does not pair lines or form a final quad.

## Validation

Tests cover horizontal/vertical lines, rotated diagonal lines, border-touching
evidence, bounded segment/cluster counts, deterministic serialization, support
diagnostics and bounded extrapolation. Hough output is intentionally not wired
into final selection until the later fusion/refinement tasks.
