# Tauri + React + Typescript

This template should help get you started developing with Tauri, React and Typescript in Vite.

## Scanner sidecar bridge

The desktop shell exposes `plan_scan` and `start_scan` Tauri commands. Both pass
the user paths as an argument array to the bundled
`attendance-scanner-sidecar`; no shell command string is constructed. Scanner
JSONL events are forwarded on `scanner://event`, while stderr diagnostics use
`scanner://stderr`. The Rust bridge prevents concurrent batches and attempts to
terminate the child process when the app exits.

The sidecar artifact is produced by the Python packaging task and must be placed
under `src-tauri/binaries/` with the target-triple suffix before a Tauri dev or
release run. The current AS-14 bridge reports a typed launch error until that
artifact is available.

## Recommended IDE Setup

- [VS Code](https://code.visualstudio.com/) + [Tauri](https://marketplace.visualstudio.com/items?itemName=tauri-apps.tauri-vscode) + [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer)
