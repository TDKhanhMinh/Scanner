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
fn quick_scan_args_forward_temp_root_and_orientation() {
    let args = build_quick_scan_args(
        r"D:\Attendance Input\page one.jpg",
        Path::new(r"C:\Users\tester\AppData\Local\Attendance\temp\quick_scan"),
        "smart_document",
        "ai_enhanced",
        "portrait",
        true,
    );

    assert_eq!(
        args,
        vec![
            "scan-one",
            "--input",
            r"D:\Attendance Input\page one.jpg",
            "--temp-root",
            r"C:\Users\tester\AppData\Local\Attendance\temp\quick_scan",
            "--mode",
            "smart_document",
            "--detector-mode",
            "ai_enhanced",
            "--orientation",
            "portrait",
            "--debug-diagnostics",
        ]
    );
}

#[test]
fn quick_scan_completed_event_is_parsed_and_validated() {
    let value = serde_json::json!({
        "protocolVersion": 1,
        "type": "quick_scan_completed",
        "timestamp": "2026-09-14T00:00:00Z",
        "success": true,
        "inputPath": "D:\\Input\\page.jpg",
        "tempPdfPath": "C:\\Temp\\quick_scan\\page.pdf",
        "documentDetected": true,
        "durationMs": 1200,
        "detectionPreview": null,
        "processedPreviewDataUrl": null,
        "errorCode": null,
        "message": null,
        "warning": null
    });

    let parsed = parse_quick_scan(&value).expect("quick scan event should be valid");
    assert_eq!(parsed.event_type, "quick_scan_completed");
    assert!(parsed.success);
    assert_eq!(
        parsed.temp_pdf_path.as_deref(),
        Some("C:\\Temp\\quick_scan\\page.pdf")
    );
}

#[test]
fn automatic_pdf_target_uses_a_unique_suffix_without_overwriting() {
    let root = std::env::temp_dir().join(format!(
        "attendance-scanner-auto-save-test-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be valid")
            .as_nanos()
    ));
    fs::create_dir_all(&root).expect("test directory should be created");
    let target = root.join("page.pdf");
    fs::write(&target, b"existing").expect("existing PDF should be created");

    let first_candidate = resolve_non_overwriting_target(&target).expect("suffix is available");
    assert_eq!(first_candidate, root.join("page_1.pdf"));
    fs::write(&first_candidate, b"existing").expect("first suffix should be created");

    let second_candidate =
        resolve_non_overwriting_target(&target).expect("second suffix is available");
    assert_eq!(second_candidate, root.join("page_2.pdf"));

    fs::remove_dir_all(root).expect("test directory should be removed");
}

#[test]
fn no_replace_move_never_overwrites_an_existing_target() {
    let root = std::env::temp_dir().join(format!(
        "attendance-scanner-no-replace-test-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be valid")
            .as_nanos()
    ));
    fs::create_dir_all(&root).expect("test directory should be created");
    let staging = root.join("staging.pdf");
    let target = root.join("page.pdf");
    fs::write(&staging, b"new").expect("staging PDF should be created");
    fs::write(&target, b"old").expect("existing PDF should be created");

    let error = move_file_without_replace(&staging, &target)
        .expect_err("no-replace move must reject an existing target");

    assert!(is_existing_target_error(&error));
    assert_eq!(
        fs::read(&target).expect("target should remain readable"),
        b"old"
    );
    assert!(staging.is_file());
    fs::remove_dir_all(root).expect("test directory should be removed");
}

#[test]
fn flat_scan_args_forward_export_mode_orientation_and_workers() {
    let args = build_flat_scan_args(
        r"D:\Flat Input",
        r"D:\Flat Output",
        "merged",
        "gray",
        "classic",
        "auto",
        2,
    );
    assert_eq!(
        args,
        vec![
            "scan-flat",
            "--input",
            r"D:\Flat Input",
            "--output",
            r"D:\Flat Output",
            "--export-mode",
            "merged",
            "--mode",
            "gray",
            "--detector-mode",
            "classic",
            "--orientation",
            "auto",
            "--workers",
            "2",
        ]
    );
    assert_eq!(normalize_flat_export_mode("MERGED"), Some("merged"));
    assert_eq!(normalize_flat_export_mode("per-image"), Some("per-image"));
    assert_eq!(normalize_flat_export_mode("grouped"), None);
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
    assert!(validate_request(
        "input",
        None,
        Some("gray"),
        Some(0),
        None,
        None,
        None,
        None,
        None,
        None
    )
    .is_err());
    assert!(validate_request(
        "input",
        None,
        Some("gray"),
        Some(5),
        None,
        None,
        None,
        None,
        None,
        None
    )
    .is_err());
    assert!(validate_request(
        "input",
        None,
        Some("gray"),
        Some(4),
        None,
        None,
        None,
        None,
        None,
        None
    )
    .is_ok());
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
        Some("grouped"),
        Some("ai_enhanced"),
        Some(false),
        Some(false),
    )
    .is_ok());
    assert!(validate_request(
        "input",
        None,
        Some("smart_document"),
        Some(2),
        Some(2026),
        Some(9),
        Some("per-image"),
        Some("classic"),
        Some(false),
        Some(false),
    )
    .is_ok());
    assert!(validate_request(
        "input",
        None,
        Some("gray"),
        Some(2),
        Some(2026),
        Some(9),
        Some("GROUPED"),
        None,
        None,
        None,
    )
    .is_ok());
    assert!(validate_request(
        "input",
        None,
        Some("gray"),
        Some(2),
        Some(2026),
        Some(9),
        Some("PER_IMAGE"),
        None,
        None,
        None,
    )
    .is_ok());
    assert!(validate_request(
        "input",
        None,
        Some("gray"),
        Some(2),
        Some(2026),
        None,
        Some("grouped"),
        None,
        None,
        None,
    )
    .is_err());
    assert!(validate_request(
        "input",
        None,
        Some("gray"),
        Some(2),
        Some(2026),
        Some(13),
        Some("grouped"),
        None,
        None,
        None,
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
        Some("ai_enhanced"),
        Some(true),
        Some(true),
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
            "--detector-mode",
            "ai_enhanced",
            "--debug-diagnostics",
            "--reprocess",
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
        Some("classic"),
        Some(true),
        Some(true),
    );
    let manual_index = args
        .iter()
        .position(|value| value == "--manual-order-json")
        .expect("manual order flag should be forwarded");
    assert_eq!(args.get(manual_index + 1), Some(&encoded));
    assert!(args
        .windows(2)
        .any(|window| window == ["--detector-mode", "classic"]));
    assert!(args.contains(&"--debug-diagnostics".to_string()));
    assert!(args.contains(&"--reprocess".to_string()));
}

#[test]
fn request_validation_rejects_unknown_detector_mode_before_launch() {
    assert!(validate_request(
        "input",
        None,
        Some("gray"),
        None,
        None,
        None,
        None,
        Some("unknown"),
        None,
        None,
    )
    .is_err());
    assert!(validate_request(
        "input",
        None,
        Some("gray"),
        None,
        None,
        None,
        None,
        Some("ai_enhanced"),
        Some(true),
        Some(false),
    )
    .is_ok());
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
