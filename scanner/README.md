# Attendance Scanner Engine

Core Python scanner engine running as a local sidecar for the Attendance Scanner Desktop application.

## Incremental planning

The `plan` command compares the discovered image inventory with the local manifest
stored in AppData. Each discovered file is classified as `new`, `modified`,
`unchanged`, or `rebuild`; only the first three processable states (`new`,
`modified`, and `rebuild`) are selected for the next batch. A missing recorded
PDF always produces `rebuild`, while a metadata-only change is verified with
SHA-256 before deciding whether the source needs processing.

Manifest entries for source files that are no longer present are retained as
historical state. The MVP never deletes PDFs automatically. A pipeline-version
mismatch is reported through `outdatedPipelineCount` but does not force a full
reprocess during the default “Scan New Files” flow.

## CLI Usage

```bash
# Plan scan without processing
python -m attendance_scanner.cli plan --input "<path_to_input_images>" --output "<path_to_output_pdf>" --mode gray

# Run batch processing
python -m attendance_scanner.cli scan-batch --input "<path_to_input_images>" --output "<path_to_output_pdf>" --mode gray --workers 3
```

`scan-batch` processes only files selected by the incremental plan. It emits
`file_started`, `file_completed`/`file_failed`, and `scan_completed` JSONL events
and checkpoints the manifest after every terminal file state. Exit code `0`
means all selected files completed, `2` means the batch completed with one or
more file failures, and `1` is reserved for fatal configuration/runtime errors.

## Diagnostics

Fatal and per-file errors use stable `errorCode` values and actionable messages.
The sidecar writes one safe structured JSON diagnostic per error to `stderr` and
keeps detailed exception text and tracebacks in
`%APPDATA%/attendance-scanner/attendance-scanner.log` on Windows (or the
platform equivalent from the configured application data directory). Image
bytes are never logged; file-level records use relative source paths.
