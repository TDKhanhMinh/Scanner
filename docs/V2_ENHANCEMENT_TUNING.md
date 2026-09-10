# V2 Timesheet Enhancement Tuning

AS-56 keeps document detection preprocessing separate from final enhancement.
The detector receives the normalized source image; only the selected warp result
is passed to Gray, B&W, Color Enhanced or Smart Document.

## Comparison workflow

Generate artifacts for one curated timesheet page:

```powershell
.\scanner\.venv\Scripts\python scripts\compare-enhancement-modes.py `
  --input C:\path\to\timesheet.jpg `
  --output C:\path\to\enhancement-comparison
```

The output directory contains `gray.png`, `bw.png`, `color.png`,
`smart_document.png`, and `comparison.json`. The report records output shape,
luminance contrast, edge density, source-edge/fine-stroke recall proxy, and
color-ink retention where color is available.

The recommendation is based on readability proxies with a stable mode-order
tie-break. File size is deliberately excluded. These metrics are comparative
signals, not a calibrated handwriting-recognition score; final approval still
requires visual review on the curated handwriting/grid corpus.

## Mode guidance

- Gray remains the conservative default for faint blue/black handwriting and
  grid lines; sharpening/denoise must not create halos or break cells.
- B&W is a high-contrast option and must be rejected for a corpus if the
  fine-stroke recall proxy or visual review shows mass loss.
- Color Enhanced applies contrast to luminance and is the preservation path for
  red stamps, blue signatures and colored handwriting.
- Smart Document is a color-preserving paper/background cleanup option, not a
  binary default.

Output dimensions must remain the warped dimensions (or a proportional
downscale from the later resize stage); enhancement never upscales an image.
