# V2 Corner Intersection and Refinement Quality

AS-49 converts AS-48 finite edge lines into candidate refined corners and keeps
the original selected quad when the refinement is unsafe. It does not modify
the source image or export a PDF.

## Intersection and partial policy

The four intersections are computed from normalized line coefficients in
canonical order: left/top → TL, top/right → TR, right/bottom → BR, and
bottom/left → BL. Near-parallel or non-finite intersections are recorded as
reasons rather than allowed to produce a far-away point.

Lines below `minimum_line_confidence`, marked `source="none"`, or marked as
fallback are replaced by the corresponding original candidate edge only when
`allow_partial_refinement` permits it. At least
`minimum_refined_edges` strong fitted lines are required. A disabled/weak fit
therefore remains recoverable and explicit instead of silently changing the
quad.

## Quality gates

The refined quad must pass AS-40 geometry validation. Every corner displacement
is compared with both an absolute pixel limit and an image-diagonal fraction.
Optional before/after edge-support values reject a refinement that drops beyond
the configured tolerance. Rejected refinement returns `selected_corners` equal
to the original candidate and records before/after corners, displacement,
line confidence, fallback labels and reason codes.

## Benchmark evidence

AS-35 predictions may carry both `raw_corners` and final `corners`. The report
keeps separate raw/final pixel and normalized distributions, including mean and
P95, so a refinement change is measurable rather than assumed to be beneficial.

## Validation

Tests cover perfect intersections, near-parallel/outlier rejection, partial line
fallback, support-drop rejection, displacement guards and raw-vs-final report
metrics. The actual detector wiring and perspective transform remain later
integration gates.
