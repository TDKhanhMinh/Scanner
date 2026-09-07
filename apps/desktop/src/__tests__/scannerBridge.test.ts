import { beforeEach, describe, expect, it, vi } from "vitest";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import {
  listenScannerEvents,
  planScan,
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
    });
    expect(invoke).toHaveBeenNthCalledWith(2, "start_scan", {
      inputRoot: "D:/Attendance Input",
      outputRoot: "D:/Attendance Output",
      mode: "gray",
      workers: 3,
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
      "Missing sidecar",
    );
    expect(scannerErrorMessage("sidecar failed")).toBe("sidecar failed");
  });
});
