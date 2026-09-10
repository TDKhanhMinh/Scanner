# V2 Candidate Scoring and Ranking

AS-45 scores the valid candidate pool from AS-44 with an explainable,
benchmark-tunable policy. It does not mutate candidates, fit geometry, or
perform perspective warp.

## Score components

`score_candidate()` stores every component with raw value, configured weight,
normalized weight, contribution, applicability, and policy:

- mask-to-quad IoU;
- mask coverage and mask confidence when supplied;
- edge support;
- AS-40 geometry quality and convexity;
- border behavior;
- optional soft aspect hint;
- source reliability (`mask_fit`, `mixed`, `contour`, `hough`).

Missing evidence is not treated as failure: CV/Hough candidates use explicit
neutral values (`neutral_mask_score`, `neutral_edge_score`, and
`neutral_aspect_score`) and mark those components `applicable=false`. Border
touching is allowed and uses a configurable soft score.

## Ranking decision

`rank_candidate_pool()` sorts by score, geometry quality, source, and candidate
ID in a deterministic order. It returns one of:

- `selected`: top score is at least `minimum_final_score` and clearly ahead;
- `ambiguous`: top two scores are within `ambiguity_margin`, so no candidate is
  silently promoted;
- `below_threshold`: top score is below the minimum;
- `no_candidate`: the valid pool is empty.

Rejected AS-44 candidates are not passed to ranking; their reason-coded evidence
remains available for diagnostics. A high mask IoU cannot override a poor
geometry score when the candidate is degenerate or tiny.

## Benchmark sweep

Use `sweep_candidate_scores()` with a list of `CandidateScoringConfig` values to
compare weights and thresholds without changing detector code. The returned
ranking reports preserve the full breakdown, making benchmark tuning auditable.

## Validation

Tests cover complete breakdown/neutral policy, score direction for mask/edge
evidence, tiny-vs-document ranking, minimum/ambiguity decisions, deterministic
ordering and weight sweeps. Threshold defaults are initial policy only; AS-35
benchmark results and the approved production corpus must tune/revalidate them.
