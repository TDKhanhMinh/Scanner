# V2 Segmentation Postprocessing and Coordinate Mapping

AS-39 adds the independent mask postprocessing boundary between the AS-38
segmentation adapter and later quadrilateral fitting. It makes resize policy,
thresholding, morphology, component selection, and coordinate mapping explicit
and testable without changing PDF/export behavior.

## Resize and transform policy

`SegmentationConfig.resize_mode` is either:

- `stretch`: resize directly to the model tensor dimensions; `scale_x` and
  `scale_y` are recorded separately;
- `letterbox`: preserve source aspect ratio, center the resized pixels, and
  record uniform scale plus `pad_x`/`pad_y`.

`SegmentationTransform` stores source and model dimensions, resized content
dimensions, padding, and scale factors. It maps model-input points back to
EXIF-normalized source pixels and removes letterbox padding before resizing a
probability mask to source size. Rounding is explicit and synthetic round-trip
tests require at most one pixel error.

No width/height swap is inferred from orientation. `load_image` first applies
EXIF orientation; all subsequent source dimensions refer to those normalized
pixels, including portrait, landscape, odd-size, and EXIF-rotated fixtures.

## Mask cleanup and component policy

`postprocess_document_mask()` accepts only a finite 2D probability mask in
`[0,1]`. It applies configurable thresholding and bounded elliptical open/close
morphology. Kernels are disabled with `0`, otherwise must be odd; iterations are
limited to `0..2` so thin document edges are not aggressively eroded.

Connected components expose area/ratio, bounding box, centroid, confidence,
centrality, border-contact edges, and deterministic score. The selected target
component combines area, centrality, confidence, and border-contact signals. If
the top two candidates are within `ambiguity_margin`, the best candidate is
still deterministic but `ambiguous=true` and both component records remain
available for review; no competition is silently discarded.

The adapter's `SegmentationOutput.binary_mask` is the selected component mask,
while `postprocess.cleaned_mask` and the full component list remain internal
evidence. Generic detector diagnostics contain only scalar summaries such as
component count, selected label, ambiguity, border contact, threshold, and
padding.

## End-to-end use

```python
segmentation = adapter.segment(loaded_image)
mask = segmentation.binary_mask
source_points = segmentation.transform.map_points_to_source(model_points)
```

The segmentation adapter still returns AS-36 `mask_only` and never creates the
final quad. AS-44 consumes the transform/candidate contract for quadrilateral
mapping and refinement.

## Validation

Tests cover threshold and bounded morphology behavior, portrait/landscape/odd
dimensions, EXIF-normalized dimensions, letterbox point round-trip, border
contact, multi-component ambiguity, deterministic component selection, and
invalid probability/kernel settings. The AS-35 benchmark runner can consume
mask evidence through the adapter path; real model/corpus accuracy remains a
later gate.
