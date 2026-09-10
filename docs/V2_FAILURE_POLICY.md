# V2 Failure and Fallback Policy

AS-57 closes the no-silent-failure boundary for the V2 detector.

## Invariants

- No usable candidate: keep the normalized full image and emit
  `FALLBACK_FULL_IMAGE` plus an actionable detection reason.
- Ambiguous candidates: do not auto-warp; expose the ambiguity and keep the
  full image for review.
- Invalid perspective: discard the attempted crop and keep the source/full
  normalized image. A corrupted or partial crop is never committed.
- Segmentation provider missing or failing: try the configured CV rescue path;
  if no safe candidate remains, use full-image fallback and expose
  `SEGMENTATION_LOW_CONFIDENCE`/`CV_NO_CANDIDATE` evidence.
- A single file failure remains isolated. Grouped export keeps the existing
  atomicity rule and does not commit a grouped PDF when any required page fails.

User-facing event messages contain only stable, actionable text. Technical
exception details and tracebacks remain in the diagnostics log. Warning,
failure, skipped and fallback counters are kept separate in the desktop UI;
`scan_completed` remains authoritative for terminal batch totals.

Curated regression coverage includes provider unavailable, no candidate,
ambiguous ranking, perspective failure, refinement-rejected taxonomy mapping,
and grouped atomicity. Production model accuracy and packaged runtime remain
later release gates.
