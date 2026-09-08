// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use tauri::{Emitter, Manager, State};
use tauri_plugin_shell::{
    process::{CommandChild, CommandEvent},
    ShellExt,
};

const SIDECAR_NAME: &str = "attendance-scanner-sidecar";
const SCANNER_EVENT_CHANNEL: &str = "scanner://event";
const SCANNER_STDERR_CHANNEL: &str = "scanner://stderr";
const PROTOCOL_VERSION: u64 = 1;

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ScannerDiagnostic {
    pub stream: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum ScannerBridgeError {
    AlreadyRunning {
        message: String,
    },
    InvalidRequest {
        message: String,
    },
    LaunchFailed {
        message: String,
    },
    StreamFailed {
        message: String,
    },
    InvalidEvent {
        message: String,
    },
    MissingPlan {
        message: String,
    },
    SidecarExited {
        code: i32,
        #[serde(skip_serializing_if = "Option::is_none")]
        error_code: Option<String>,
        message: String,
    },
    Internal {
        message: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScanPlanPayload {
    pub protocol_version: u64,
    #[serde(rename = "type")]
    pub event_type: String,
    pub timestamp: String,
    pub input_root: String,
    pub output_root: String,
    pub employees: u64,
    pub total_images: u64,
    pub new: u64,
    pub modified: u64,
    #[serde(default)]
    pub rebuild: u64,
    pub unchanged: u64,
    #[serde(default)]
    pub files_to_process: u64,
    #[serde(default)]
    pub outdated_pipeline_count: u64,
    #[serde(default)]
    pub unsupported_count: u64,
    pub collisions: Vec<String>,
    #[serde(default)]
    pub year: Option<u32>,
    #[serde(default)]
    pub month: Option<u32>,
    #[serde(default)]
    pub export_mode: Option<String>,
    #[serde(default)]
    pub document_groups: u64,
    #[serde(default)]
    pub expected_artifacts: u64,
    #[serde(default)]
    pub complete_groups: u64,
    #[serde(default)]
    pub incomplete_groups: u64,
    #[serde(default)]
    pub ambiguous_groups: u64,
    #[serde(default)]
    pub pages_needing_review: u64,
    #[serde(default)]
    pub review_groups: Vec<Value>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ScanRunOutcome {
    pub exit_code: i32,
}

#[derive(Clone, Default)]
struct ScannerState {
    active: Arc<Mutex<bool>>,
    child: Arc<Mutex<Option<CommandChild>>>,
}

impl ScannerState {
    fn begin(&self) -> Result<(), ScannerBridgeError> {
        let mut active = self
            .active
            .lock()
            .map_err(|_| ScannerBridgeError::Internal {
                message: "Scanner state lock is poisoned".to_string(),
            })?;
        if *active {
            return Err(ScannerBridgeError::AlreadyRunning {
                message: "A scanner batch is already running".to_string(),
            });
        }
        *active = true;
        Ok(())
    }

    fn attach_child(&self, child: CommandChild) -> Result<(), ScannerBridgeError> {
        let mut slot = self
            .child
            .lock()
            .map_err(|_| ScannerBridgeError::Internal {
                message: "Scanner child lock is poisoned".to_string(),
            })?;
        *slot = Some(child);
        Ok(())
    }

    fn clear_child(&self, terminate: bool) {
        let child = self.child.lock().ok().and_then(|mut slot| slot.take());
        if terminate {
            if let Some(child) = child {
                let _ = child.kill();
            }
        }
    }

    fn finish(&self) {
        if let Ok(mut active) = self.active.lock() {
            *active = false;
        }
    }

    fn terminate_child(&self) {
        self.clear_child(true);
        self.finish();
    }
}

#[derive(Debug)]
struct SidecarCapture {
    exit_code: i32,
    plan: Option<ScanPlanPayload>,
    stderr: Vec<CapturedDiagnostic>,
}

#[derive(Debug, Clone)]
struct CapturedDiagnostic {
    error_code: Option<String>,
    message: String,
}

fn validate_request(
    input_root: &str,
    output_root: Option<&str>,
    mode: Option<&str>,
    workers: Option<u32>,
    year: Option<u32>,
    month: Option<u32>,
    export_mode: Option<&str>,
) -> Result<(), ScannerBridgeError> {
    if input_root.trim().is_empty() {
        return Err(ScannerBridgeError::InvalidRequest {
            message: "inputRoot must be a non-empty path".to_string(),
        });
    }
    if output_root.is_some_and(|path| path.trim().is_empty()) {
        return Err(ScannerBridgeError::InvalidRequest {
            message: "outputRoot must be a non-empty path when provided".to_string(),
        });
    }
    if let Some(value) = mode {
        if !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "gray" | "bw" | "color"
        ) {
            return Err(ScannerBridgeError::InvalidRequest {
                message: format!("Unsupported scan mode: {value}"),
            });
        }
    }
    if let Some(value) = workers {
        if !(1..=4).contains(&value) {
            return Err(ScannerBridgeError::InvalidRequest {
                message: "workers must be between 1 and 4".to_string(),
            });
        }
    }
    if year.is_some() != month.is_some() {
        return Err(ScannerBridgeError::InvalidRequest {
            message: "year and month must be provided together".to_string(),
        });
    }
    if let Some(value) = year {
        if !(1..=9999).contains(&value) {
            return Err(ScannerBridgeError::InvalidRequest {
                message: "year must be between 1 and 9999".to_string(),
            });
        }
    }
    if let Some(value) = month {
        if !(1..=12).contains(&value) {
            return Err(ScannerBridgeError::InvalidRequest {
                message: "month must be between 1 and 12".to_string(),
            });
        }
    }
    if let Some(value) = export_mode {
        if !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "per-image" | "grouped"
        ) {
            return Err(ScannerBridgeError::InvalidRequest {
                message: format!("Unsupported export mode: {value}"),
            });
        }
    }
    Ok(())
}

fn validate_manual_order(value: Option<&Value>) -> Result<(), ScannerBridgeError> {
    let Some(value) = value else {
        return Ok(());
    };
    let object = value
        .as_object()
        .ok_or_else(|| ScannerBridgeError::InvalidRequest {
            message: "manualOrder must be an object mapping group ids to source paths".to_string(),
        })?;
    for (group_id, paths) in object {
        if group_id.trim().is_empty() {
            return Err(ScannerBridgeError::InvalidRequest {
                message: "manualOrder group ids must be non-empty".to_string(),
            });
        }
        let paths = paths
            .as_array()
            .ok_or_else(|| ScannerBridgeError::InvalidRequest {
                message: format!("manualOrder value for {group_id} must be an array"),
            })?;
        let mut seen = HashSet::new();
        for path in paths {
            let path = path
                .as_str()
                .ok_or_else(|| ScannerBridgeError::InvalidRequest {
                    message: format!("manualOrder paths for {group_id} must be strings"),
                })?;
            if path.trim().is_empty() || !seen.insert(path.replace('\\', "/")) {
                return Err(ScannerBridgeError::InvalidRequest {
                    message: format!(
                        "manualOrder paths for {group_id} must be non-empty and unique"
                    ),
                });
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_sidecar_args(
    command: &str,
    input_root: &str,
    output_root: Option<&str>,
    mode: Option<&str>,
    workers: Option<u32>,
    year: Option<u32>,
    month: Option<u32>,
    export_mode: Option<&str>,
    manual_order_json: Option<&str>,
) -> Vec<String> {
    let mut args = vec![
        command.to_string(),
        "--input".to_string(),
        input_root.to_string(),
    ];
    if let Some(output_root) = output_root {
        args.extend(["--output".to_string(), output_root.to_string()]);
    }
    if let Some(mode) = mode {
        args.extend(["--mode".to_string(), mode.to_string()]);
    }
    if let Some(workers) = workers {
        args.extend(["--workers".to_string(), workers.to_string()]);
    }
    if let Some(year) = year {
        args.extend(["--year".to_string(), year.to_string()]);
    }
    if let Some(month) = month {
        args.extend(["--month".to_string(), month.to_string()]);
    }
    if let Some(export_mode) = export_mode {
        args.extend(["--export-mode".to_string(), export_mode.to_string()]);
    }
    if let Some(manual_order_json) = manual_order_json {
        args.extend([
            "--manual-order-json".to_string(),
            manual_order_json.to_string(),
        ]);
    }
    args
}

fn validate_wire_event(value: &Value) -> Result<(), ScannerBridgeError> {
    let object = value
        .as_object()
        .ok_or_else(|| ScannerBridgeError::InvalidEvent {
            message: "Sidecar stdout event must be a JSON object".to_string(),
        })?;

    if object.get("protocolVersion").and_then(Value::as_u64) != Some(PROTOCOL_VERSION) {
        return Err(ScannerBridgeError::InvalidEvent {
            message: "Unsupported or missing protocolVersion in sidecar event".to_string(),
        });
    }
    if object
        .get("timestamp")
        .and_then(Value::as_str)
        .is_none_or(|timestamp| timestamp.trim().is_empty())
    {
        return Err(ScannerBridgeError::InvalidEvent {
            message: "Missing timestamp in sidecar event".to_string(),
        });
    }
    if object
        .get("type")
        .and_then(Value::as_str)
        .is_none_or(|event_type| event_type.trim().is_empty())
    {
        return Err(ScannerBridgeError::InvalidEvent {
            message: "Missing type in sidecar event".to_string(),
        });
    }
    Ok(())
}

fn parse_scan_plan(value: &Value) -> Result<ScanPlanPayload, ScannerBridgeError> {
    validate_wire_event(value)?;
    let has_process_count = value
        .as_object()
        .is_some_and(|object| object.contains_key("filesToProcess"));
    let mut plan: ScanPlanPayload = serde_json::from_value(value.clone()).map_err(|error| {
        ScannerBridgeError::InvalidEvent {
            message: format!("Invalid scan_plan event: {error}"),
        }
    })?;
    let expected = plan.new + plan.modified + plan.rebuild;
    if !has_process_count {
        plan.files_to_process = expected;
    }
    if plan.files_to_process != expected {
        return Err(ScannerBridgeError::InvalidEvent {
            message: format!(
                "filesToProcess must equal new + modified + rebuild ({expected}), got {}",
                plan.files_to_process
            ),
        });
    }
    Ok(plan)
}

fn drain_lines(buffer: &mut Vec<u8>, flush_remainder: bool) -> Vec<Vec<u8>> {
    let mut lines = Vec::new();
    let mut consumed = 0;
    for (index, byte) in buffer.iter().enumerate() {
        if *byte == b'\n' {
            lines.push(buffer[consumed..index].to_vec());
            consumed = index + 1;
        }
    }
    if consumed > 0 {
        buffer.drain(..consumed);
    }
    if flush_remainder && !buffer.is_empty() {
        lines.push(std::mem::take(buffer));
    }
    lines
}

fn parse_stderr_diagnostic(raw_message: &str) -> CapturedDiagnostic {
    serde_json::from_str::<Value>(raw_message)
        .ok()
        .and_then(|value| {
            let object = value.as_object()?;
            let message = object
                .get("message")
                .and_then(Value::as_str)
                .filter(|message| !message.trim().is_empty())?
                .to_string();
            let error_code = object
                .get("errorCode")
                .and_then(Value::as_str)
                .filter(|code| !code.trim().is_empty())
                .map(str::to_string);
            Some(CapturedDiagnostic {
                error_code,
                message,
            })
        })
        .unwrap_or_else(|| CapturedDiagnostic {
            error_code: None,
            message: "Scanner sidecar emitted an unstructured diagnostic.".to_string(),
        })
}

fn forward_stdout_line<R: tauri::Runtime>(
    app: &tauri::AppHandle<R>,
    bytes: Vec<u8>,
    plan: &mut Option<ScanPlanPayload>,
) -> Result<(), ScannerBridgeError> {
    let line = String::from_utf8(bytes).map_err(|error| ScannerBridgeError::StreamFailed {
        message: format!("Sidecar emitted non-UTF-8 output: {error}"),
    })?;
    let line = line.trim_end_matches('\r').trim();
    if line.is_empty() {
        return Ok(());
    }
    let value: Value =
        serde_json::from_str(line).map_err(|error| ScannerBridgeError::InvalidEvent {
            message: format!("Invalid JSONL from scanner sidecar: {error}"),
        })?;
    validate_wire_event(&value)?;
    if value.get("type").and_then(Value::as_str) == Some("scan_plan") {
        *plan = Some(parse_scan_plan(&value)?);
    }
    app.emit(SCANNER_EVENT_CHANNEL, value)
        .map_err(|error| ScannerBridgeError::StreamFailed {
            message: format!("Unable to forward scanner event: {error}"),
        })
}

fn forward_stderr_line<R: tauri::Runtime>(
    app: &tauri::AppHandle<R>,
    bytes: Vec<u8>,
    stderr: &mut Vec<CapturedDiagnostic>,
) -> Result<(), ScannerBridgeError> {
    let raw_message = String::from_utf8_lossy(&bytes)
        .trim_end_matches(['\r', '\n'])
        .trim()
        .to_string();
    if raw_message.is_empty() {
        return Ok(());
    }

    let diagnostic = parse_stderr_diagnostic(&raw_message);

    stderr.push(diagnostic.clone());
    app.emit(
        SCANNER_STDERR_CHANNEL,
        ScannerDiagnostic {
            stream: "stderr".to_string(),
            message: diagnostic.message,
            error_code: diagnostic.error_code,
        },
    )
    .map_err(|error| ScannerBridgeError::StreamFailed {
        message: format!("Unable to forward scanner diagnostics: {error}"),
    })
}

async fn spawn_sidecar<R: tauri::Runtime>(
    app: &tauri::AppHandle<R>,
    state: &ScannerState,
    args: Vec<String>,
) -> Result<tauri::async_runtime::Receiver<CommandEvent>, ScannerBridgeError> {
    let command =
        app.shell()
            .sidecar(SIDECAR_NAME)
            .map_err(|error| ScannerBridgeError::LaunchFailed {
                message: format!("Unable to resolve scanner sidecar: {error}"),
            })?;
    let (receiver, child) = command
        .args(args)
        .set_raw_out(true)
        .spawn()
        .map_err(|error| ScannerBridgeError::LaunchFailed {
            message: format!("Unable to launch scanner sidecar: {error}"),
        })?;
    state.attach_child(child)?;
    Ok(receiver)
}

async fn consume_sidecar<R: tauri::Runtime>(
    app: tauri::AppHandle<R>,
    mut receiver: tauri::async_runtime::Receiver<CommandEvent>,
) -> Result<SidecarCapture, ScannerBridgeError> {
    let mut exit_code = None;
    let mut plan = None;
    let mut stderr = Vec::new();
    let mut stdout_buffer = Vec::new();
    let mut stderr_buffer = Vec::new();

    while let Some(event) = receiver.recv().await {
        match event {
            CommandEvent::Stdout(bytes) => {
                stdout_buffer.extend(bytes);
                for line in drain_lines(&mut stdout_buffer, false) {
                    forward_stdout_line(&app, line, &mut plan)?;
                }
            }
            CommandEvent::Stderr(bytes) => {
                stderr_buffer.extend(bytes);
                for line in drain_lines(&mut stderr_buffer, false) {
                    forward_stderr_line(&app, line, &mut stderr)?;
                }
            }
            CommandEvent::Error(message) => {
                return Err(ScannerBridgeError::StreamFailed { message });
            }
            CommandEvent::Terminated(payload) => {
                exit_code = payload.code;
            }
            _ => {}
        }
    }

    for line in drain_lines(&mut stdout_buffer, true) {
        forward_stdout_line(&app, line, &mut plan)?;
    }
    for line in drain_lines(&mut stderr_buffer, true) {
        forward_stderr_line(&app, line, &mut stderr)?;
    }

    let exit_code = exit_code.ok_or_else(|| ScannerBridgeError::StreamFailed {
        message: "Scanner sidecar stream ended without an exit status".to_string(),
    })?;
    Ok(SidecarCapture {
        exit_code,
        plan,
        stderr,
    })
}

async fn stream_sidecar<R: tauri::Runtime>(
    app: tauri::AppHandle<R>,
    state: ScannerState,
    args: Vec<String>,
) -> Result<SidecarCapture, ScannerBridgeError> {
    let receiver = spawn_sidecar(&app, &state, args).await;
    let result = match receiver {
        Ok(receiver) => consume_sidecar(app, receiver).await,
        Err(error) => Err(error),
    };
    state.clear_child(result.is_err());
    result
}

fn sidecar_exit_error(capture: &SidecarCapture) -> ScannerBridgeError {
    let diagnostic = capture.stderr.last();
    let message = diagnostic.map_or_else(
        || "Scanner sidecar exited without a diagnostic.".to_string(),
        |item| item.message.clone(),
    );
    let error_code = diagnostic.and_then(|item| item.error_code.clone());
    ScannerBridgeError::SidecarExited {
        code: capture.exit_code,
        error_code,
        message,
    }
}

#[tauri::command]
#[allow(clippy::too_many_arguments)]
async fn plan_scan(
    app: tauri::AppHandle,
    state: State<'_, ScannerState>,
    input_root: String,
    output_root: Option<String>,
    mode: Option<String>,
    year: Option<u32>,
    month: Option<u32>,
    export_mode: Option<String>,
) -> Result<ScanPlanPayload, ScannerBridgeError> {
    validate_request(
        input_root.as_str(),
        output_root.as_deref(),
        mode.as_deref(),
        None,
        year,
        month,
        export_mode.as_deref(),
    )?;
    let args = build_sidecar_args(
        "plan",
        input_root.as_str(),
        output_root.as_deref(),
        mode.as_deref(),
        None,
        year,
        month,
        export_mode.as_deref(),
        None,
    );
    let scanner_state = state.inner().clone();
    scanner_state.begin()?;
    let stream_result = stream_sidecar(app, scanner_state.clone(), args).await;
    let result = match stream_result {
        Ok(capture) if capture.exit_code == 0 => {
            capture.plan.ok_or_else(|| ScannerBridgeError::MissingPlan {
                message: "Scanner sidecar completed without a scan_plan event".to_string(),
            })
        }
        Ok(capture) => Err(sidecar_exit_error(&capture)),
        Err(error) => Err(error),
    };
    scanner_state.finish();
    result
}

#[tauri::command]
#[allow(clippy::too_many_arguments)]
async fn start_scan(
    app: tauri::AppHandle,
    state: State<'_, ScannerState>,
    input_root: String,
    output_root: Option<String>,
    mode: Option<String>,
    workers: Option<u32>,
    year: Option<u32>,
    month: Option<u32>,
    export_mode: Option<String>,
    manual_order: Option<Value>,
) -> Result<ScanRunOutcome, ScannerBridgeError> {
    validate_request(
        input_root.as_str(),
        output_root.as_deref(),
        mode.as_deref(),
        workers,
        year,
        month,
        export_mode.as_deref(),
    )?;
    validate_manual_order(manual_order.as_ref())?;
    let manual_order_json = manual_order
        .as_ref()
        .map(serde_json::to_string)
        .transpose()
        .map_err(|error| ScannerBridgeError::InvalidRequest {
            message: format!("manualOrder could not be serialized: {error}"),
        })?;
    let args = build_sidecar_args(
        "scan-batch",
        input_root.as_str(),
        output_root.as_deref(),
        mode.as_deref(),
        workers,
        year,
        month,
        export_mode.as_deref(),
        manual_order_json.as_deref(),
    );
    let scanner_state = state.inner().clone();
    scanner_state.begin()?;
    let stream_result = stream_sidecar(app, scanner_state.clone(), args).await;
    let result = match stream_result {
        Ok(capture) if matches!(capture.exit_code, 0 | 2) => Ok(ScanRunOutcome {
            exit_code: capture.exit_code,
        }),
        Ok(capture) => Err(sidecar_exit_error(&capture)),
        Err(error) => Err(error),
    };
    scanner_state.finish();
    result
}

#[cfg(target_os = "windows")]
fn hide_helper_windows() {
    use windows_sys::Win32::Foundation::{BOOL, HWND, LPARAM};
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        EnumWindows, GetClassNameW, GetWindowThreadProcessId, ShowWindow, SW_HIDE,
    };

    unsafe extern "system" fn enum_windows_proc(hwnd: HWND, lparam: LPARAM) -> BOOL {
        let target_pid = lparam as u32;
        let mut pid: u32 = 0;
        GetWindowThreadProcessId(hwnd, &mut pid);
        if pid == target_pid {
            let mut class_name = [0u16; 256];
            let len = GetClassNameW(hwnd, class_name.as_mut_ptr(), 256);
            if len > 0 {
                let class_str = String::from_utf16_lossy(&class_name[..len as usize]);
                if class_str.contains("UAC") || class_str.contains("Tao Thread") {
                    ShowWindow(hwnd, SW_HIDE);
                }
            }
        }
        1
    }

    let pid = std::process::id();
    unsafe {
        EnumWindows(Some(enum_windows_proc), pid as isize);
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .manage(ScannerState::default())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![greet, plan_scan, start_scan])
        .setup(|app| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.show();
                let _ = window.set_focus();
            }

            #[cfg(target_os = "windows")]
            {
                // Run window sanitizer to guarantee Process.MainWindowTitle matches primary window
                hide_helper_windows();
                std::thread::spawn(|| {
                    std::thread::sleep(std::time::Duration::from_millis(150));
                    hide_helper_windows();
                    std::thread::sleep(std::time::Duration::from_millis(500));
                    hide_helper_windows();
                });
            }

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|app_handle, event| {
        if let tauri::RunEvent::ExitRequested { .. } = event {
            app_handle.state::<ScannerState>().terminate_child();
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sidecar_args_keep_user_paths_as_individual_arguments() {
        let args = build_sidecar_args(
            "scan-batch",
            r"D:\Attendance Input\Nguyễn Văn A",
            Some(r"D:\Attendance Output"),
            Some("gray"),
            Some(3),
            None,
            None,
            None,
            None,
        );

        assert_eq!(
            args,
            vec![
                "scan-batch",
                "--input",
                r"D:\Attendance Input\Nguyễn Văn A",
                "--output",
                r"D:\Attendance Output",
                "--mode",
                "gray",
                "--workers",
                "3",
            ]
        );
    }

    #[test]
    fn legacy_scan_plan_without_process_count_is_derived() {
        let value = serde_json::json!({
            "protocolVersion": 1,
            "type": "scan_plan",
            "timestamp": "2026-09-07T00:00:00Z",
            "inputRoot": "/input",
            "outputRoot": "/output",
            "employees": 1,
            "totalImages": 2,
            "new": 1,
            "modified": 0,
            "rebuild": 1,
            "unchanged": 0,
            "collisions": []
        });

        let plan = parse_scan_plan(&value).expect("legacy plan should be accepted");
        assert_eq!(plan.files_to_process, 2);
    }

    #[test]
    fn inconsistent_scan_plan_is_rejected() {
        let value = serde_json::json!({
            "protocolVersion": 1,
            "type": "scan_plan",
            "timestamp": "2026-09-07T00:00:00Z",
            "inputRoot": "/input",
            "outputRoot": "/output",
            "employees": 1,
            "totalImages": 1,
            "new": 1,
            "modified": 0,
            "rebuild": 0,
            "unchanged": 0,
            "filesToProcess": 0,
            "collisions": []
        });

        assert!(parse_scan_plan(&value).is_err());
    }

    #[test]
    fn scanner_state_rejects_concurrent_batches_and_releases_after_finish() {
        let state = ScannerState::default();

        assert!(state.begin().is_ok());
        assert!(matches!(
            state.begin(),
            Err(ScannerBridgeError::AlreadyRunning { .. })
        ));
        state.finish();
        assert!(state.begin().is_ok());
        state.finish();
    }

    #[test]
    fn request_validation_rejects_worker_counts_outside_contract() {
        assert!(validate_request("input", None, Some("gray"), Some(0), None, None, None).is_err());
        assert!(validate_request("input", None, Some("gray"), Some(5), None, None, None).is_err());
        assert!(validate_request("input", None, Some("gray"), Some(4), None, None, None).is_ok());
    }

    #[test]
    fn request_validation_and_args_support_period_and_export_mode() {
        assert!(validate_request(
            "input",
            None,
            Some("gray"),
            Some(2),
            Some(2026),
            Some(9),
            Some("grouped")
        )
        .is_ok());
        assert!(validate_request(
            "input",
            None,
            Some("gray"),
            Some(2),
            Some(2026),
            None,
            Some("grouped")
        )
        .is_err());
        assert!(validate_request(
            "input",
            None,
            Some("gray"),
            Some(2),
            Some(2026),
            Some(13),
            Some("grouped")
        )
        .is_err());

        let args = build_sidecar_args(
            "plan",
            "D:\\Input",
            Some("D:\\Output"),
            Some("gray"),
            None,
            Some(2026),
            Some(9),
            Some("grouped"),
            None,
        );
        assert_eq!(
            args,
            vec![
                "plan",
                "--input",
                "D:\\Input",
                "--output",
                "D:\\Output",
                "--mode",
                "gray",
                "--year",
                "2026",
                "--month",
                "9",
                "--export-mode",
                "grouped",
            ]
        );
    }

    #[test]
    fn manual_order_is_validated_and_forwarded_as_one_json_argument() {
        let valid = serde_json::json!({
            "NV01:2026-09": ["NV01/page-2.png", "NV01/page-1.png"]
        });
        assert!(validate_manual_order(Some(&valid)).is_ok());
        assert!(validate_manual_order(Some(&serde_json::json!({
            "NV01:2026-09": ["NV01/page-1.png", "NV01/page-1.png"]
        })))
        .is_err());

        let encoded = serde_json::to_string(&valid).expect("manual order JSON should serialize");
        let args = build_sidecar_args(
            "scan-batch",
            "D:\\Input",
            None,
            Some("gray"),
            Some(2),
            Some(2026),
            Some(9),
            Some("grouped"),
            Some(&encoded),
        );
        assert_eq!(args[args.len() - 2], "--manual-order-json");
        assert_eq!(args.last(), Some(&encoded));
    }

    #[test]
    fn drain_lines_reassembles_fragmented_jsonl_chunks() {
        let mut buffer = br#"{"#.to_vec();
        assert!(drain_lines(&mut buffer, false).is_empty());

        buffer.extend(br#""type":"scan_plan"}"#);
        assert!(drain_lines(&mut buffer, false).is_empty());

        buffer.extend(b"\nnext");
        assert_eq!(
            drain_lines(&mut buffer, false),
            vec![br#"{"type":"scan_plan"}"#.to_vec()]
        );
        assert_eq!(drain_lines(&mut buffer, true), vec![b"next".to_vec()]);
    }

    #[test]
    fn structured_stderr_keeps_user_message_and_error_code_only() {
        let diagnostic = parse_stderr_diagnostic(
            r#"{"event":"scanner_error","errorCode":"IMAGE_DECODE_FAILED","message":"Không thể đọc ảnh này.","traceback":"private diagnostics"}"#,
        );

        assert_eq!(
            diagnostic.error_code.as_deref(),
            Some("IMAGE_DECODE_FAILED")
        );
        assert_eq!(diagnostic.message, "Không thể đọc ảnh này.");
        assert!(!diagnostic.message.contains("private diagnostics"));
    }

    #[test]
    fn unstructured_stderr_uses_safe_fallback_message() {
        let diagnostic = parse_stderr_diagnostic("raw path and traceback");

        assert_eq!(diagnostic.error_code, None);
        assert_eq!(
            diagnostic.message,
            "Scanner sidecar emitted an unstructured diagnostic."
        );
    }
}
