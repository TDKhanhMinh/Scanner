//! Unit tests for the Tauri scanner bridge and sidecar protocol helpers.

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
        Some(9),
        Some("GROUPED")
    )
    .is_ok());
    assert!(validate_request(
        "input",
        None,
        Some("gray"),
        Some(2),
        Some(2026),
        Some(9),
        Some("PER_IMAGE")
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
        Some("PER_IMAGE"),
        None,
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
            "per-image",
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
        None,
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
