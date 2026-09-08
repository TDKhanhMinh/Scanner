import { describe, expect, it } from "vitest";
import {
  initialScanExecutionState,
  MAX_RECENT_ACTIVITY,
  scanExecutionReducer,
} from "@/lib/scanExecutionReducer";
import type { ScannerEvent } from "@/types/scanner";

function started(relativePath: string, index: number): ScannerEvent {
  return {
    protocolVersion: 1,
    type: "file_started",
    timestamp: `2026-09-08T00:00:0${index}Z`,
    relativePath,
    employeeName: "NV01",
    index,
    total: 2,
  };
}

function completed(
  relativePath: string,
  timestamp: string,
  warning: string | null = null,
): ScannerEvent {
  return {
    protocolVersion: 1,
    type: "file_completed",
    timestamp,
    relativePath,
    employeeName: "NV01",
    outputRelativePath: relativePath.replace(/\.[^.]+$/, ".pdf"),
    documentDetected: !warning,
    warning,
    durationMs: 10,
  };
}

describe("scanExecutionReducer", () => {
  it("keeps aggregate counts accurate for out-of-order completion and duplicates", () => {
    let state = scanExecutionReducer(initialScanExecutionState, {
      type: "scan_started",
      totalToProcess: 2,
    });
    state = scanExecutionReducer(state, {
      type: "scanner_event",
      event: started("NV01/1.jpg", 1),
    });
    state = scanExecutionReducer(state, {
      type: "scanner_event",
      event: started("NV01/2.jpg", 2),
    });
    state = scanExecutionReducer(state, {
      type: "scanner_event",
      event: completed("NV01/2.jpg", "2026-09-08T00:00:12Z", "DOCUMENT_NOT_DETECTED"),
    });
    state = scanExecutionReducer(state, {
      type: "scanner_event",
      event: completed("NV01/1.jpg", "2026-09-08T00:00:11Z"),
    });
    const afterDuplicate = scanExecutionReducer(state, {
      type: "scanner_event",
      event: completed("NV01/2.jpg", "2026-09-08T00:00:12Z", "DOCUMENT_NOT_DETECTED"),
    });

    expect(afterDuplicate.processed).toBe(2);
    expect(afterDuplicate.success).toBe(1);
    expect(afterDuplicate.warning).toBe(1);
    expect(afterDuplicate.failed).toBe(0);
    expect(afterDuplicate.results).toHaveLength(2);
  });

  it("uses scan_completed as the authoritative terminal summary", () => {
    let state = scanExecutionReducer(initialScanExecutionState, {
      type: "scan_started",
      totalToProcess: 1,
    });
    state = scanExecutionReducer(state, {
      type: "scanner_event",
      event: {
        protocolVersion: 1,
        type: "scan_completed",
        timestamp: "2026-09-08T00:00:10Z",
        totalProcessed: 1,
        success: 0,
        failed: 1,
        warning: 0,
        skipped: 4,
        durationMs: 200,
      },
    });

    expect(state.phase).toBe("completed");
    expect(state.processed).toBe(1);
    expect(state.failed).toBe(1);
    expect(state.skipped).toBe(4);
    expect(state.currentFile).toBe("");
  });

  it("caps recent activity without changing processed totals", () => {
    let state = scanExecutionReducer(initialScanExecutionState, {
      type: "scan_started",
      totalToProcess: MAX_RECENT_ACTIVITY + 5,
    });
    for (let index = 0; index < MAX_RECENT_ACTIVITY + 5; index += 1) {
      state = scanExecutionReducer(state, {
        type: "scanner_event",
        event: completed(
          `NV01/${index}.jpg`,
          `2026-09-08T00:01:${String(index).padStart(2, "0")}Z`,
        ),
      });
    }

    expect(state.processed).toBe(MAX_RECENT_ACTIVITY + 5);
    expect(state.results).toHaveLength(MAX_RECENT_ACTIVITY);
  });

  it("represents transport failures as an error phase", () => {
    const state = scanExecutionReducer(initialScanExecutionState, {
      type: "scan_error",
      message: "Sidecar unavailable",
    });

    expect(state.phase).toBe("error");
    expect(state.errorMessage).toBe("Sidecar unavailable");
  });
});
