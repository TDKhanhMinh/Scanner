# V2 Segmentation-First Hybrid Decision Policy

AS-46 adds `HybridDocumentDetector`, which coordinates the AS-38/39
segmentation output, AS-41/43 CV/Hough evidence, AS-44 candidate construction,
AS-42 edge support, AS-40 geometry validation, and AS-45 scoring. It returns the
AS-36 `DocumentDetectionResult` contract and keeps the existing PDF/export API
unchanged.

## Decision flow

1. Try the configured segmentation provider once for the image. Its mask is
   treated as evidence, not as an automatic crop.
2. Generate bounded CV contour and optional Hough candidates for rescue and
   disagreement analysis.
3. Build the multi-source AS-44 pool, attach edge-support evidence, then rank
   through the configurable AS-45 policy.
4. Return a canonical quad only when the top candidate clears
   `minimum_final_score` and is outside `ambiguity_margin` from the runner-up.
5. Return structured `ambiguous` or `no_document` failure with a decision trace
   when evidence is too close/weak. When no candidate is usable and the policy
   allows it, set `fallback_used=true` and emit `FALLBACK_FULL_IMAGE`; the
   detector never silently warps a low-confidence candidate.

## Decision trace

The result records a compact `DetectorDecisionTrace` containing path
(`segmentation_first`, `agreement`, `disagreement`, `cv_fallback`, `ambiguous`,
or `full_image_fallback`), segmentation state, valid/rejected candidate counts,
source counts, ranking status, selected ID, fallback flag, and reason codes.
This trace is safe to serialize; raw masks remain internal artifacts.

`mask_fit`/`mixed` selection indicates segmentation support, a CV/Hough winner
after segmentation failure is `cv_fallback`, and a CV/Hough winner while
segmentation was available is `disagreement`. A mixed source is the explicit
agreement path; corners are never averaged blindly outside AS-44's bounded
candidate construction.

## Modes

`create_detector()` keeps A/B selection explicit:

- `v1_cv`: existing V1 OpenCV adapter;
- `segmentation_only`: caller-supplied segmentation `DocumentDetector`;
- `cv_v2`: hybrid orchestration with segmentation disabled;
- `hybrid`: segmentation-first policy;
- `docaligner_reference`: caller-supplied reference provider.

Operators see product modes such as `AI Enhanced`/`Classic`; model names remain
diagnostic metadata rather than UI coupling.

## Validation

Tests cover segmentation success, CV rescue after provider failure, agreement/
disagreement source traces, ambiguous ranking without final corners, total
failure full-image fallback, and explicit A/B factory errors. Model accuracy,
threshold tuning, packaged runtime, and production corpus remain later gates.
