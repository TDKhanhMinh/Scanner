# Attendance Scanner V2.0 Release Runbook

AS-60 is the final release gate. A passing developer test suite is not enough
to mark the V2 release complete.

## Release sequence

1. Run the full Python, frontend and Rust regression suites, typechecks, builds,
   lint/format checks and `git diff --check`.
2. Run `scripts/run-v2-decision.py` with the approved benchmark manifests and
   prediction bundles. Require model checksums and a non-blocked decision.
3. Build the sidecar and Tauri Windows installers with
   `scripts/build-windows.ps1`.
4. Verify generated artifact size/SHA-256 and offline metadata:

   ```powershell
   .\scripts\verify-release-manifest.ps1
   ```

5. With a clean Windows test account/VM, install MSI or NSIS, launch the app,
   run both PER_IMAGE and GROUPED smoke cases, close/relaunch, verify state and
   `needs_reprocess`, then uninstall. Confirm input/PDF folders are untouched.
6. Review model, ONNX Runtime, OpenCV and benchmark/reference licenses before
   publishing the installer.

## Current gate policy

The release manifest records `offline=true`, target triple
`x86_64-pc-windows-msvc`, detector product modes and an explicit V2
`productionReady=false` until the model artifact is supplied. Running the
verifier without `-RequireV2Model` checks all generated artifact hashes. Running
it with `-RequireV2Model` is the V2 release gate and must fail until an approved
segmentation model path, checksum and packaged startup test are recorded.

The current package/scanner version is `0.1.0`; no production model version is
claimed while the V2 gate is blocked. Do not relabel the package as a V2 release
only because the source-level tests pass.

## Rollback

1. Stop the desktop app and preserve the current input/output folders.
2. Use the known-good V1 tag/artifact documented in `docs/V1_BASELINE.md`, or
   rebuild it from the locked dependencies.
3. Use a separate output root when comparing V1 and V2; do not delete PDFs or
   rewrite shared Git history.
4. If a V2 manifest/model version is newer, let incremental planning rebuild
   affected outputs under the V1 pipeline rather than reusing V2 artifacts.
5. Record the failed release gate, corpus, installer hash and rollback decision
   for the next remediation cycle.

## Gate evidence template

Record the exact date, Git commit, benchmark manifest hash, model checksum,
sidecar/installer hashes, OS build, CPU, test account, smoke input class and
results for every release candidate. Keep “static pass”, “packaged runtime
pass” and “production approval” as separate statuses.
