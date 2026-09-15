# V4 Workflow SLA Benchmark

The v4 benchmark measures the two execution boundaries that matter for a desktop
scanner:

- `end_to_end`: a fresh `scan-one` process, including interpreter/sidecar startup;
- `pipeline`: repeated in-process `scan_one` execution.

The fixture is deterministic and exactly `4000x3000` pixels (12MP). It contains
only synthetic geometry and does not use employee documents. When `--fixture`
is omitted, the script keeps a generated fixture next to the report and records
its SHA-256 plus generation recipe. A supplied fixture is recorded as
`source=provided` and is not labeled synthetic.

## Run

From the repository root:

```powershell
python scripts/benchmark-workflow-sla.py `
  --output build/benchmarks/v4-workflow-sla.json `
  --detector-mode classic `
  --detector-mode ai_enhanced `
  --repetitions 5
```

To measure a packaged sidecar at the end-to-end boundary, add
`--packaged-artifact` with the target-triple executable. AI measurement for a
packaged artifact requires an explicit `--packaged-model` declaration (or it is
reported as skipped; host model availability is not used as proof). Add `--enforce` when a
CI or release step should return a non-zero exit code for a measured SLA miss.

The thresholds are:

| Mode | Boundary | P50 | P95 |
| --- | --- | ---: | ---: |
| Classical OpenCV | Fresh process | 1.2s | 1.8s |
| Classical OpenCV | In-process | 0.4s | 0.7s |
| AI Enhanced | Fresh process | 2.8s | 3.8s |
| AI Enhanced | In-process | 1.2s | 1.8s |

The JSON report records measured values, the fixture dimensions, model
availability and whether enforcement was requested. A passing benchmark does
not replace clean Windows packaged-app, model checksum or manual document
quality approval.
