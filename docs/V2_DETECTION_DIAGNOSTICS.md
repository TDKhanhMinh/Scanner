# V2 Detection Diagnostics and Failure Taxonomy

AS-53 provides a bounded, typed diagnostics boundary for troubleshooting
detection failures without bloating manifest state or the V1 JSONL parser.

## Machine taxonomy

`DetectionFailureReason` is versioned (`1.0`) and covers:

- `SEGMENTATION_LOW_CONFIDENCE`;
- `MASK_INVALID` / `MASK_AMBIGUOUS_COMPONENTS`;
- `QUAD_FIT_FAILED`;
- `CV_NO_CANDIDATE`;
- `HYBRID_AMBIGUOUS`;
- `REFINEMENT_REJECTED`;
- `PERSPECTIVE_INVALID`;
- `FALLBACK_FULL_IMAGE`.

`summarize_detection_failure()` maps internal warning/failure state to typed
reason codes and a separate user-safe message. Developer diagnostics remain in
the existing structured log path, bounded to avoid binary/raw-mask payloads.

## Persistence

`ManifestEntry` and `ManifestArtifact` store only primary/reason code lists,
user-safe summary message, detector/model provenance, fallback state and the
small scalar quality summary from AS-51. Full candidate pools, probability
masks, debug images and stack traces stay outside the manifest in the opt-in
debug directory/log.

`FileCompletedEvent` adds reason fields as optional additive properties, so old
V1 consumers can continue parsing the event's existing fields. File/group
status remains failed when processing truly fails; no group is marked complete
just because a diagnostic artifact exists.

## Validation

Tests cover V1 no-document/fallback mapping, user-safe versus machine reason
separation, additive event serialization, persisted per-image/group summaries,
mixed grouped aggregation and legacy state compatibility.
