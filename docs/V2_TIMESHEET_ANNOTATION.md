# V2 Timesheet Annotation Workflow

AS-34 provides a standalone, resumable workflow for creating the timesheet
ground-truth corpus used by the AS-32 benchmark contract. It is intentionally
separate from the desktop scanner UI and does not change the V1 scan path.

## Data policy

- Use only local, public, synthetic, or anonymized images. Do not commit
  employee names, signatures, IDs, or other attendance PII.
- Record the real dataset name, license, attribution, and source URL when a
  public dataset is used. The tool requires these fields during session setup;
  it never invents provenance.
- Keep the image corpus outside the repository when licensing or privacy makes
  redistribution unsuitable. The session and manifest store relative paths,
  while the image root remains a local operator setting.

SmartDoc is one suitable public source for generic document geometry: its
dataset page describes the corner ground truth and CC BY 4.0 terms, and the
automation-friendly repository provides metadata and downloadable frame/model
archives:

- <https://sites.google.com/site/icdar15smartdoc/challenge-1/dataset>
- <https://github.com/jchazalon/smartdoc15-ch1-dataset>

## Coordinate and visibility policy

Coordinates are recorded after the scanner's EXIF orientation normalization and
always use `TL -> TR -> BR -> BL` in image pixel coordinates. A fully visible
document must have all four corners inside the image bounds. If the document is
cut by the frame or obscured, mark it `--not-visible` and provide one of:
`border_touching`, `out_of_frame`, or `occluded`. Such samples are useful for
negative/robustness evaluation, but finalization still requires a valid
four-corner geometric annotation; points may be outside the image in that case.

`document_polygon` and `mask_path` are optional. Use them for the difficult
subset (for example paper overlapping another sheet or an irregular visible
boundary), not as a requirement for every image. AS-32 validates polygon
geometry, relative paths, and mask dimensions.

## Workflow

### 1. Create a session

Place a curated flat set of images in a local folder, then run:

```powershell
python scripts/annotate-timesheet-benchmark.py init `
  --image-root D:\datasets\timesheet-images `
  --session D:\datasets\timesheet-annotation.json `
  --dataset timesheet-domain-v1 `
  --split acceptance `
  --source-name "Internal anonymized timesheet corpus" `
  --license "Internal use" `
  --attribution "Attendance Scanner team"
```

`init` discovers supported direct-child images in deterministic filename order,
reads their EXIF-normalized dimensions, and writes all records as `pending`.
The source fields must be replaced with the actual provenance for a real
corpus. Use `--force` only when intentionally recreating the session.

### 2. Annotate and save progress

For a local interactive workbench:

```powershell
python scripts/annotate-timesheet-benchmark.py gui `
  --session D:\datasets\timesheet-annotation.json
```

Click corners in canonical order. The workbench supports right-click undo,
`r` reset, `s` save, `k` skip, `n`/Enter next, `p` previous, and `q` save and
quit. `skip` is retained for triage but prevents finalization, so it must be
resolved before producing a benchmark manifest.

For automation, CI, or an occlusion case, update one record directly:

```powershell
python scripts/annotate-timesheet-benchmark.py set `
  --session D:\datasets\timesheet-annotation.json `
  --sample-id timesheet-domain-v1:sheet-001.jpg `
  --corners-json '[[10,10],[1190,12],[1188,1680],[12,1678]]' `
  --visible `
  --tag desk `
  --tag shadow `
  --sequence-id capture-001
```

An occluded/out-of-frame record uses the same command with `--not-visible` and
an explicit reason. Optional polygon and mask references are relative to the
image root:

```powershell
python scripts/annotate-timesheet-benchmark.py set `
  --session D:\datasets\timesheet-annotation.json `
  --sample-id timesheet-domain-v1:sheet-002.jpg `
  --corners-json '[[8,8],[1192,8],[1200,1700],[0,1700]]' `
  --not-visible `
  --visibility-reason border_touching `
  --polygon-json '[[8,8],[1192,8],[1200,1700],[0,1700]]' `
  --tag overlap
```

### 3. Review overlays and coverage

Render a non-destructive overlay that a reviewer can open quickly:

```powershell
python scripts/annotate-timesheet-benchmark.py preview `
  --session D:\datasets\timesheet-annotation.json `
  --sample-id timesheet-domain-v1:sheet-001.jpg `
  --output D:\datasets\review\sheet-001-overlay.png
```

Inspect progress and scenario coverage at any time:

```powershell
python scripts/annotate-timesheet-benchmark.py summary `
  --session D:\datasets\timesheet-annotation.json
```

The reviewer should open overlays for every scenario tag, verify that labels
are TL/TR/BR/BL and sit on the paper boundary rather than an inner table/grid,
then confirm that the split and sequence assignments are appropriate. The
summary reports pending/annotated/skipped records, split counts, tag coverage,
and optional polygon/mask coverage.

### 4. Finalize the AS-32 manifest

The acceptance target is 200–300 images when source material is available. The
finalizer fails closed if the count is outside that range, any record is
pending/skipped, a visible corner is out of bounds, references are missing, or
sequence IDs leak across splits:

```powershell
python scripts/annotate-timesheet-benchmark.py finalize `
  --session D:\datasets\timesheet-annotation.json `
  --output-manifest D:\datasets\timesheet-domain-v1\manifest.json
```

For synthetic/unit-test sessions only, use explicit smaller bounds. Do not use
that override as evidence that the production-sized corpus exists.

## Current handoff gap

This repository contains the session model, GUI/CLI workflow, overlay renderer,
coverage summary, and AS-32 manifest finalizer, but it does not bundle a
200–300-image timesheet corpus. A corpus owner still needs to provide approved
anonymized/public images, provenance, reviewer sign-off, and the materialized
manifest. This is an explicit data-availability gate; no images or labels are
fabricated by the tooling.
