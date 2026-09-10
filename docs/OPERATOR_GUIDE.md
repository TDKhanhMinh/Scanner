# Attendance Scanner Desktop — Operator Guide

This guide describes the monthly local-first workflow for scanning attendance images.

## 1. Expected folders

Keep one direct employee folder below the input root:

```text
Attendance Input/
├── Nguyen Van A/
│   ├── 01.jpg
│   └── 02.png
└── Tran Thi B/
    └── 01.webp
```

Supported source formats are `.jpg`, `.jpeg`, `.png`, and `.webp`. Hidden/system
files and files outside direct employee folders are ignored. Do not put real
attendance data in repository fixtures or bug reports.

## 2. Monthly workflow

1. Add new images to the correct employee folders.
2. Open Attendance Scanner Desktop.
3. Select the input root, for example `D:\Attendance Input`.
4. Confirm or choose the output folder. The default is a sibling folder named
   `<input-root>_pdf`.
5. Select Gray, B&W, Color Enhanced, or Smart Document mode and choose a worker count from 1 to 4.
6. Wait for the scan plan to finish.
7. Select **Scan New Files**. The progress card shows the current employee/file,
   counters, warnings, failures, and skipped files.
8. Open the PDF output folder from the completion card, or inspect the Results tab.

The app works offline. Images and PDFs stay on the local machine.

## 3. Scan modes

- **Gray**: grayscale output suitable for ordinary printed forms and the smallest
  output size.
- **B&W**: adaptive binary output for high-contrast black-and-white documents.
- **Color Enhanced**: preserves color information while applying the scanner's
  color enhancement pipeline.
- **Smart Document**: whitens paper, reduces shadow gradients, improves contrast,
  and preserves colored handwriting and stamps.

The selected mode is part of the processing identity. Changing the mode rebuilds
the affected source output so an older enhancement result is not reused.

## 4. Plan and output semantics

The plan classifies each source as:

- **New**: no successful output is recorded.
- **Modified**: source metadata/hash changed or a previous attempt failed.
- **Rebuild**: a recorded output is missing.
- **Unchanged**: the source and recorded output are still valid.

The current MVP writes one PDF per processed image, preserving employee-relative
folders. A successful output is committed atomically. The scanner does not delete
PDFs when a source image is deleted.

## 5. Result meanings

- **Success**: image processed and PDF committed.
- **Warning**: PDF committed, but a non-fatal condition such as
  `DOCUMENT_NOT_DETECTED` occurred; review the source image.
- **Failure**: this image was isolated from the batch. Correct the source/output
  issue and run **Scan New Files** again.
- **Skipped**: the plan found the source unchanged, so no image processing was needed.

Failure rows show a technical code and a user-safe message. The technical code is
the value to include when contacting support; do not send image bytes.

## 6. Troubleshooting

### `INVALID_INPUT_ROOT`

The selected path does not exist or is not a directory. Select the parent folder
that directly contains employee folders and confirm the Windows account can read it.

### `OUTPUT_NOT_WRITABLE` or `PDF_WRITE_FAILED`

Choose a writable output folder, close any program locking the PDF, check free
disk space, and retry. Existing input/output documents are not removed by the
uninstaller.

### `IMAGE_DECODE_FAILED`

The source may be corrupt, empty, unsupported, or still being copied. Replace or
re-export the image, then retry the scan.

### `DOCUMENT_NOT_DETECTED`

The PDF was still produced using the full normalized image. Check lighting,
contrast, framing, and document edges. This is a warning, not a batch failure.

### Retry a failed file

Fix the source image or output permission, then run **Scan New Files** with the
same input/output roots. Failed entries are retryable; unchanged successful and
warning entries remain skipped. If a PDF was deleted, the plan selects that source
as `rebuild`.

### Support diagnostics

Record the technical error code, employee-relative source path, app version, and
the approximate time of the failure. Detailed Python diagnostics are stored in
the application log under `%APPDATA%\attendance-scanner\attendance-scanner.log`
on Windows. Logs do not contain image bytes.

## 7. AppData and uninstall safety

Scanner state is stored under `%APPDATA%\attendance-scanner\state`. Installer
uninstall may remove application state according to Windows installer defaults,
but must never remove the user-selected input root or PDF output root. Back up
important PDFs independently of application state.

## 8. V2 detector controls and diagnostics

- **AI Enhanced** is the default document detector. It combines configured AI
  segmentation evidence with CV fitting/refinement and falls back safely when
  evidence is unavailable.
- **Classic** uses the OpenCV-only detector and is the operator fallback. The
  image enhancement choice (Gray, B&W, Color Enhanced, Smart Document) is a
  separate setting and continues to work as before.
- The selected detector mode and diagnostics toggle are remembered locally for
  the next app session. **Cho phép reprocess** is reset when the app starts and
  must be explicitly enabled after reviewing the `needs_reprocess` count.
- When diagnostics or a warning is available, select the preview icon in the
  result list. The overlay shows the bounded preview, candidate/final corners,
  fallback reason and quality score. A quality score is not an absolute
  probability. Missing preview data never blocks the batch.

If the detector cannot safely identify the paper, the PDF uses the full
normalized image and displays a warning such as `DOCUMENT_NOT_DETECTED` or
`FALLBACK_FULL_IMAGE`; the system does not silently crop a doubtful candidate.
