# V2 ONNX Document Segmentation Adapter

AS-38 adds `OnnxSegmentationAdapter`, a generic segmentation provider built on
the reusable AS-37 ONNX service. It binds image preprocessing, output decoding,
probability/binary mask creation, evidence summary, and optional debug artifacts.
It intentionally does not fit final four corners; that belongs to the next
postprocessing stage.

## Model and provenance

The adapter accepts a required `SegmentationModelCard` containing model version,
license, attribution, and optional URL. The AS-37 service supplies the actual
ONNX file SHA-256 after load. This repository does not bundle an approved
MobileNetV3 + DeepLabV3 artifact, so a model owner must provide the licensed
artifact and verify its checksum before a production run. The adapter is named
generically so the provider can be replaced without downstream coupling to
MobileNetV3 or DeepLabV3.

## Input contract

`LoadedImage` is already EXIF-normalized BGR. The adapter converts it to RGB,
resizes to the configured input dimensions, scales to `[0,1]`, applies the
configured mean/std, and emits contiguous `float32` NCHW (or explicitly
configured NHWC) input with an optional batch dimension. The transform records
source/input dimensions and scale factors for later coordinate mapping.

```python
config = SegmentationConfig(
    input_width=224,
    input_height=224,
    input_layout="nchw",
    output_layout="nchw",
    activation="auto",
    threshold=0.5,
)
adapter = OnnxSegmentationAdapter(
    service,
    model_card=SegmentationModelCard(
        version="approved-model-version",
        license="actual-license",
        attribution="actual-attribution",
    ),
    config=config,
)
segmentation = adapter.segment(loaded_image)
```

## Output decoding and boundary

The decoder supports one-channel sigmoid/probability output and multi-class
logits/probabilities in NCHW/NHWC/CHW/HWC forms. `auto` detects conventional
layouts when unambiguous; ambiguous small tensors must configure
`output_layout`. It validates finite numeric outputs, class index, and
probability ranges. The probability mask is resized back to the original image
dimensions and thresholded into a boolean binary mask.

`SegmentationOutput` contains the raw probability/binary arrays only as an
internal Python artifact. Its `DocumentDetectionResult` is `detected=false` with
`failure_code="mask_only"`, no final corners, and only generic mask confidence,
area ratio, component count, model checksum, and timing evidence. Its
`diagnostics_extension()` therefore never serializes the raw mask to JSONL or a
manifest. `adapter.detect()` satisfies the AS-36 provider interface while the
later geometry task can call `segment()` to consume the internal mask.

When `debug_artifact_dir` is configured, the adapter writes `.probability.npy`,
normalized `.probability.png`, and thresholded `.binary.png` artifacts for local
debugging. These are explicitly outside the sidecar diagnostics contract.

## Failure and reuse policy

ONNX model/input/output failures are mapped to the typed AS-36 detector failure
contract by `adapter.detect()`. Expected no-document/failure outcomes remain
structured; no model binary or raw mask is printed to stdout. The adapter uses
the same AS-37 service instance across images, so the ONNX session is reused and
not recreated per file.

## Validation status

Tests cover BGR-to-RGB/channel order, transform metadata, logits/probability
decoding, ambiguous layout and bad probability rejection, original-size mask
and dtype/range, debug artifacts, mask-only evidence, service reuse, and model
failure mapping. The bundled ORT sample model proves the infrastructure path;
an approved MobileNetV3/DeepLabV3 artifact plus representative easy,
overlap, and low-contrast fixtures remain an external data/runtime gate.
