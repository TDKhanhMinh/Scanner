import { beforeEach, describe, expect, it, vi } from "vitest";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import {
  listenScannerEvents,
  planScan,
  scannerDiagnosticMessage,
  scannerErrorMessage,
  startScan,
} from "@/lib/scannerBridge";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
}));

vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn(),
}));

describe("scannerBridge", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("passes sidecar request fields as structured invoke arguments", async () => {
    vi.mocked(invoke).mockResolvedValue({
      protocolVersion: 1,
      type: "scan_plan",
      timestamp: "2026-09-07T00:00:00Z",
      inputRoot: "D:/Attendance Input",
      outputRoot: "D:/Attendance Output",
      employees: 1,
      totalImages: 1,
      new: 1,
      modified: 0,
      rebuild: 0,
      unchanged: 0,
      filesToProcess: 1,
      collisions: [],
      outdatedPipelineCount: 0,
    });

    await planScan({
      inputRoot: "D:/Attendance Input",
      outputRoot: "D:/Attendance Output",
      mode: "gray",
    });
    await startScan({
      inputRoot: "D:/Attendance Input",
      outputRoot: "D:/Attendance Output",
      mode: "gray",
      workers: 3,
    });

    expect(invoke).toHaveBeenNthCalledWith(1, "plan_scan", {
      inputRoot: "D:/Attendance Input",
      outputRoot: "D:/Attendance Output",
      mode: "gray",
      detectorMode: null,
      debugDiagnostics: false,
      reprocess: false,
      year: null,
      month: null,
      exportMode: null,
    });
    expect(invoke).toHaveBeenNthCalledWith(2, "start_scan", {
      inputRoot: "D:/Attendance Input",
      outputRoot: "D:/Attendance Output",
      mode: "gray",
      detectorMode: null,
      debugDiagnostics: false,
      reprocess: false,
      workers: 3,
      year: null,
      month: null,
      exportMode: null,
      manualOrder: null,
      skipGroups: null,
    });
  });

  it("passes the selected period and export mode to both bridge commands", async () => {
    vi.mocked(invoke).mockResolvedValue({
      protocolVersion: 1,
      type: "scan_plan",
      timestamp: "2026-09-07T00:00:00Z",
      inputRoot: "D:/Attendance Input",
      outputRoot: "D:/Attendance Output",
      employees: 1,
      totalImages: 2,
      new: 2,
      modified: 0,
      rebuild: 0,
      unchanged: 0,
      filesToProcess: 2,
      collisions: [],
      outdatedPipelineCount: 0,
      unsupportedCount: 0,
      period: { year: 2026, month: 9 },
      exportMode: "GROUPED",
    });

    await planScan({
      inputRoot: "D:/Attendance Input",
      outputRoot: "D:/Attendance Output",
      mode: "gray",
      period: { year: 2026, month: 9 },
      exportMode: "GROUPED",
    });
    await startScan({
      inputRoot: "D:/Attendance Input",
      outputRoot: "D:/Attendance Output",
      mode: "gray",
      workers: 2,
      period: { year: 2026, month: 9 },
      exportMode: "GROUPED",
    });

    expect(invoke).toHaveBeenNthCalledWith(1, "plan_scan", {
      inputRoot: "D:/Attendance Input",
      outputRoot: "D:/Attendance Output",
      mode: "gray",
      detectorMode: null,
      debugDiagnostics: false,
      reprocess: false,
      year: 2026,
      month: 9,
      exportMode: "GROUPED",
    });
    expect(invoke).toHaveBeenNthCalledWith(2, "start_scan", {
      inputRoot: "D:/Attendance Input",
      outputRoot: "D:/Attendance Output",
      mode: "gray",
      detectorMode: null,
      debugDiagnostics: false,
      reprocess: false,
      workers: 2,
      year: 2026,
      month: 9,
      exportMode: "GROUPED",
      manualOrder: null,
      skipGroups: null,
    });
  });

  it("parses forwarded scanner events and ignores invalid payloads", async () => {
    let handler: ((event: { payload: unknown }) => void) | undefined;
    vi.mocked(listen).mockImplementation(async (_channel, callback) => {
      handler = callback as (event: { payload: unknown }) => void;
      const unlisten: UnlistenFn = () => undefined;
      return unlisten;
    });
    const onEvent = vi.fn();
    await listenScannerEvents(onEvent);

    handler?.({
      payload: {
        protocolVersion: 1,
        type: "file_started",
        timestamp: "2026-09-07T00:00:00Z",
        relativePath: "NV01/01.jpg",
        employeeName: "NV01",
        index: 1,
        total: 1,
      },
    });
    handler?.({ payload: { type: "invalid" } });

    expect(onEvent).toHaveBeenCalledTimes(1);
    expect(onEvent.mock.calls[0][0]).toMatchObject({
      type: "file_started",
      relativePath: "NV01/01.jpg",
    });
  });

  it("extracts typed bridge errors for the UI", () => {
    expect(scannerErrorMessage({ kind: "launchFailed", message: "Missing sidecar" })).toBe(
      "SIDECAR_LAUNCH_FAILED: Không thể khởi động scanner sidecar. Hãy kiểm tra bản cài đặt và thử lại.",
    );
    expect(scannerErrorMessage({ errorCode: "IMAGE_DECODE_FAILED", message: "raw detail" })).toBe(
      "IMAGE_DECODE_FAILED: Không thể đọc ảnh này. Hãy kiểm tra file có bị hỏng và thuộc định dạng được hỗ trợ.",
    );
    expect(scannerErrorMessage("sidecar failed")).toBe(
      "UNEXPECTED_ERROR: Đã xảy ra lỗi không xác định khi quét. Vui lòng thử lại; nếu lỗi lặp lại, gửi mã cho bộ phận hỗ trợ.",
    );
  });

  it("maps structured diagnostics without exposing traceback details", () => {
    expect(
      scannerDiagnosticMessage({
        stream: "stderr",
        errorCode: "OUTPUT_NOT_WRITABLE",
        message: "raw diagnostic detail",
      }),
    ).toBe(
      "OUTPUT_NOT_WRITABLE: Không thể ghi vào thư mục xuất PDF. Hãy chọn thư mục khác hoặc kiểm tra quyền truy cập.",
    );

    expect(
      scannerDiagnosticMessage({
        stream: "stderr",
        message: "C:/private/traceback and technical details",
      }),
    ).toBe(
      "UNEXPECTED_ERROR: Đã xảy ra lỗi không xác định khi quét. Vui lòng thử lại; nếu lỗi lặp lại, gửi mã cho bộ phận hỗ trợ.",
    );

    expect(
      scannerDiagnosticMessage({
        stream: "stderr",
        message: { privatePath: "C:/private/image.jpg" },
      } as unknown),
    ).toBe(
      "UNEXPECTED_ERROR: Đã xảy ra lỗi không xác định khi quét. Vui lòng thử lại; nếu lỗi lặp lại, gửi mã cho bộ phận hỗ trợ.",
    );

    expect(scannerErrorMessage({ errorCode: "constructor" })).toBe(
      "UNEXPECTED_ERROR: Đã xảy ra lỗi không xác định khi quét. Vui lòng thử lại; nếu lỗi lặp lại, gửi mã cho bộ phận hỗ trợ.",
    );
    expect(scannerDiagnosticMessage({ errorCode: "toString" })).toBe(
      "UNEXPECTED_ERROR: Đã xảy ra lỗi không xác định khi quét. Vui lòng thử lại; nếu lỗi lặp lại, gửi mã cho bộ phận hỗ trợ.",
    );
    expect(scannerErrorMessage({ kind: "hasOwnProperty" })).toBe(
      "UNEXPECTED_ERROR: Đã xảy ra lỗi không xác định khi quét. Vui lòng thử lại; nếu lỗi lặp lại, gửi mã cho bộ phận hỗ trợ.",
    );
  });
});
