# V2 OpenCV Candidate Generator

AS-41 turns the existing contour-based OpenCV detector into a candidate source.
It does not decide the final document and does not perform perspective warp.
The generator lives in `attendance_scanner.cv_candidates` and keeps
`V1CvDocumentDetector` available for baseline A/B benchmark runs.

## Candidate contract

`generate_cv_candidates(loaded_image, config)` returns a `CvCandidateSet` with:

- source/detection dimensions and scale factor;
- the exact preprocessing configuration;
- deterministic top-N `CvCandidate` records;
- canonical candidate corners in original pixel coordinates;
- contour area/ratio, perimeter, convexity, confidence, source contour index,
  approximation epsilon, and scalar diagnostics.

Contours are fully enumerated and approximated over configurable epsilon values;
the function never returns from the first four-point approximation. Candidates
are de-duplicated, ranked deterministically by geometry confidence/area/source
order, then limited to `max_candidates`.

## Scope and safety

The generator only prepares an edge map, extracts contours, approximates and
orders quadrilaterals, validates basic geometry, and returns evidence. It does
not crop, warp, enhance, or mutate the source image. Border/candidate quality
decisions are available to later geometry/fusion stages rather than being
hidden in export behavior.

`V1CvDocumentDetector` remains the compatibility adapter for the existing
single-result V1 path. AS-41 therefore supplies a new CV candidate pool without
changing current PDF/export behavior.

## Validation

Tests cover no-candidate, one-document, nested/internal-rectangle multiple
candidate, deterministic ordering, top-N configuration, diagnostics and source
immutability. Hough lines, fusion, refinement, and final ranking are intentionally
left to subsequent V2 tasks.
