# Grid-guided dewarp

The scanner contains a fail-safe dewarp stage for photographed forms whose rows
and columns are locally curved after the planar perspective warp.

## Pipeline behavior

The stage runs after the existing document warp and orientation step and before
enhancement:

1. Detect long horizontal and vertical grid evidence at a bounded analysis
   resolution.
2. Track each line as a smooth curve without using document-specific pixel
   coordinates.
3. Regularize neighboring curves as one smooth displacement field so a single
   noisy row or column cannot shear nearby content.
4. Build an inverse mesh with identity anchors at every canvas boundary. The
   stage does not apply a fixed crop or synthesize missing pixels.
5. Reduce column-correction strength progressively when the full two-axis map
   is unsafe; if necessary, keep only the independently validated row map.
6. Reject the map when evidence is weak, curves cross, pixels leave the source,
   local steps fold, cross-axis movement tears, the local Jacobian becomes
   degenerate, or measured row/column straightness does not improve.

When a map is rejected, the original planar-warped image is returned unchanged
and a scalar reason is recorded in `detection_quality_summary`.

## Activation

The desktop product pipeline enables the guarded stage by default. Direct
library calls to `dewarp_document_grid` remain opt-in and can enable it with an
explicit configuration:

```python
from attendance_scanner import GridDewarpConfig, PipelineConfig, scan_one

config = PipelineConfig(
    grid_dewarp=GridDewarpConfig(enabled=True),
)
result = scan_one(source, config=config)
```

The stage changes production output only after every safety and quality gate
passes. Review both the image and diagnostics, especially
`gridDewarpApplied`, `gridDewarpReason`, `gridDewarpMinimumJacobian`, and
`gridDewarpMaximumCrossStep`. The straightness scores provide before/after
evidence for both axes. `gridDewarpAppliedAxes` and `gridDewarpColumnBlend`
show whether the accepted result used both axes, a reduced column correction,
or rows only.

## Current verification boundary

Automated tests cover identity and boundary-anchored maps, mild curved
synthetic grids, out-of-bounds maps, folds, cross-axis tearing, weak evidence,
side-content retention, and invalid configuration. The current real-image
probe accepts only pages that satisfy all gates and otherwise preserves the
planar-warped image unchanged. This conservative fallback is intentional: a
page that cannot be corrected reliably is safer than a silently distorted one.
