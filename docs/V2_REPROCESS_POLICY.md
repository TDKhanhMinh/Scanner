# V2 Controlled Reprocess Policy

AS-52 prevents a detector/model upgrade from silently rebuilding every old PDF.
It compares the persisted AS-51 pipeline/detector/model metadata only after
normal source/output checks have run.

## Classification

- new source → `new`;
- changed source → existing `modified` rule;
- missing/stale output → existing `rebuild` rule;
- source/artifact unchanged and metadata equal → `unchanged`;
- source/artifact unchanged and pipeline/detector/model metadata differs →
  `needs_reprocess` by default;
- the same version mismatch with explicit opt-in → `rebuild`.

Version reasons include `PIPELINE_VERSION_CHANGED`,
`MODEL_VERSION_CHANGED`, `MODEL_CHECKSUM_CHANGED`, and detector name/mode
changes when the prior entry contains those optional metadata fields. Legacy
entries without AS-51 detector metadata remain compatible when their pipeline
version is still current; they are not silently rebuilt. Multiple reasons are
preserved in the plan counts. `needs_reprocess` is excluded from the process
set until opt-in, so an upgrade cannot silently overwrite V1 outputs.

## CLI opt-in

The sidecar `plan` and `scan-batch` commands accept `--reprocess`:

```powershell
attendance-scanner-sidecar plan --input C:\attendance --mode gray
attendance-scanner-sidecar plan --input C:\attendance --mode gray --reprocess
attendance-scanner-sidecar scan-batch --input C:\attendance --mode gray --reprocess
```

Without the flag, JSONL plan output reports `needsReprocess` and
`reprocessReasons` separately while `filesToProcess` remains unchanged. With
the flag, only version-affected files become rebuild work. For GROUPED export,
only groups containing opted-in affected sources are expanded; unrelated groups
remain untouched.

## Compatibility and safety

The policy preserves source-file, document-group, artifact dependency and
manual-order identities. It does not delete or bulk-overwrite old artifacts.
V1/V2 manifests with missing optional metadata are readable and remain current
when the pipeline version matches; AS-51 still
owns persisted provenance; AS-52 only decides classification/process intent.

Tests cover same-version, pipeline/model/detector mismatch, explicit opt-in,
source modification, missing output, unrelated files and CLI plan reporting.
