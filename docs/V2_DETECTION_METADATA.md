# V2 Detector Metadata in Incremental State

AS-51 extends the existing V2 manifest state without changing source-file or
document-group identity. Every newly processed `ManifestEntry` now records:

- `pipeline_version`;
- detector name and selected mode;
- detector model version and optional checksum;
- final detection status (`detected`, `fallback`, or `failed`);
- fallback-used flag;
- a small scalar quality summary for audit/reprocess decisions.

Grouped `ManifestArtifact` records carry the same provenance at artifact level,
plus a page/fallback count summary. Their existing source and artifact
dependencies are unchanged, so both `PER_IMAGE` and `GROUPED` export modes keep
their current mapping behavior.

## Compatibility

All new fields are optional/defaulted when reading old manifests. The existing
V1-to-V2 migration still preserves file/group/artifact identity and adds no
source-filename-derived document identity. Debug-heavy masks, overlays and
candidate pools are not copied into the persistent manifest; only small scalar
quality summaries are stored.

AS-51 records provenance for the current V1 OpenCV path (`v1_cv` plus the scan
mode and `opencv-classical` model version). Later detector integration can fill
the same fields with hybrid/model metadata. AS-52 owns the policy that decides
when a version mismatch requires reprocessing.

## Validation

Tests cover actual per-image persistence, grouped artifact provenance and source
dependency preservation, plus round-trip compatibility for legacy entries and
artifacts with absent optional metadata.
