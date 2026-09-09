# V2 Document Detector Contract

AS-36 defines the provider-neutral boundary between document detection and the
existing perspective/export pipeline. The contract lives in
`attendance_scanner.detector`; it does not change PDF rendering or output
behavior.

## Result model

`DocumentDetectionResult` always contains detector/model/pipeline versions and
the coordinate space. A successful result has validated canonical corners in
`TL -> TR -> BR -> BL` order. An expected no-document result has
`detected=false`, `corners=null`, a structured `failure_code`, and optional
warnings; providers should not throw for this normal outcome.

```python
from attendance_scanner.detector import DocumentDetectionResult

result = provider.detect(loaded_image)
if result.detected:
    corners = result.canonical_points()
    # Existing perspective API can consume this plain canonical point list.
```

The result also carries:

- `candidate_corners`: model-agnostic candidates from segmentation, keypoint,
  heatmap, or CV providers, each with source, confidence, coordinate space, and
  diagnostics;
- normalized confidence and `confidence_source`;
- typed geometry summary and extensible detector evidence;
- `fallback_used`, warnings, failure code, and stage timing breakdown;
- optional mask confidence/area/component/IoU summaries. Raw probability or
  binary masks are not part of the serialized contract and must remain an
  internal/debug artifact or be referenced through a separate artifact path.

## Coordinate rule

`original_pixels` means coordinates are in the EXIF-normalized source image
space used by `load_image`. A provider that emits detection-size coordinates
must label them `detection_pixels` and map them to original coordinates before
setting final `corners`. Downstream perspective and export code receives only
the canonical point list and does not import model internals.

## Provider interface

Any provider implementing:

```python
def detect(self, image: LoadedImage) -> DocumentDetectionResult: ...
```

satisfies the `DocumentDetector` protocol. This supports segmentation,
keypoint, and classical CV implementations without changing perspective or
export APIs. `adapt_v1_detector()` returns the compatibility adapter around the
existing OpenCV detector for benchmark A/B comparisons.

The V1 adapter preserves the existing detector implementation and maps its
accepted result into canonical corners. Rejected/clipped candidates remain in
`candidate_corners` with structured `failure_code` and are not exposed as final
corners. This keeps expected detection failure data-driven and prevents a
clipped candidate from being warped accidentally.

## Serialization

All contract models inherit the project's camelCase wire aliases while accepting
snake_case field names for Python callers. `diagnostics_extension()` returns a
`documentDetection` object that can be merged into existing sidecar diagnostics;
the JSONL protocol and existing PDF/export behavior remain unchanged.

## Tests

`scanner/tests/test_detector_contracts.py` covers canonical ordering and
serialization, structured no-document failure, mock segmentation/keypoint/CV
providers, the existing perspective bridge, and the V1 adapter. Providers can
be swapped in the test without changing the perspective/export call shape.
