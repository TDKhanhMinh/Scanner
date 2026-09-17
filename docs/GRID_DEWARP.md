# Grid-guided dewarp

The scanner now contains an opt-in, fail-safe dewarp stage for photographed
forms whose rows and columns are locally curved after the planar perspective
warp.

## Pipeline behavior

The stage runs after the existing document warp and orientation step and before
enhancement:

1. Detect long horizontal and vertical grid evidence at a bounded analysis
   resolution.
2. Track each line as a smooth curve without using document-specific pixel
   coordinates.
3. Build one inverse mesh and remap the complete image canvas. The stage does
   not apply a fixed crop or synthesize missing pixels.
4. Reject the map when evidence is weak, curves cross, pixels leave the source,
   local steps fold, cross-axis movement tears, or the local Jacobian becomes
   degenerate.

When a map is rejected, the original planar-warped image is returned unchanged
and a scalar reason is recorded in `detection_quality_summary`.

## Activation

The feature is deliberately disabled by default until a representative,
anonymized corpus passes visual review. It can be enabled by an explicit
pipeline configuration:

```python
from attendance_scanner import GridDewarpConfig, PipelineConfig, scan_one

config = PipelineConfig(
    grid_dewarp=GridDewarpConfig(enabled=True),
)
result = scan_one(source, config=config)
```

The default-disabled behavior prevents an experimental curve tracker from
changing existing production output. The activation decision should be made
only after checking both the image and the diagnostics, especially
`gridDewarpApplied`, `gridDewarpReason`, `gridDewarpMinimumJacobian`, and
`gridDewarpMaximumCrossStep`.

## Current verification boundary

Automated tests cover identity maps, mild curved synthetic grids, out-of-bounds
maps, folds, cross-axis tearing, weak evidence, and invalid configuration.
The current real-image probe keeps the stage in fallback for unsafe maps. This
is an intentional safety result, not evidence that the photographed pages are
already fully dewarped. A future rollout needs anonymized real fixtures and a
visual acceptance gate proving that row/column straightness improves without
losing side content or introducing seams.
