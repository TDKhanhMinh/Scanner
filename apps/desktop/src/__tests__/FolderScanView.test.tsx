import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AppShell } from "@/components/scanner/AppShell";
import { invokeFlatScan, listenScannerEvents } from "@/lib/scannerBridge";
import { open } from "@tauri-apps/plugin-dialog";
import type { ScannerEvent } from "@/types/scanner";

vi.mock("@tauri-apps/plugin-dialog", () => ({
  open: vi.fn(),
  save: vi.fn(),
}));

vi.mock("@tauri-apps/plugin-opener", () => ({
  openPath: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn().mockResolvedValue(() => undefined),
}));

vi.mock("@/lib/scannerBridge", () => ({
  invokeFlatScan: vi.fn(),
  listenScannerEvents: vi.fn().mockResolvedValue(() => undefined),
  listenScannerDiagnostics: vi.fn().mockResolvedValue(() => undefined),
  planScan: vi.fn(),
  startScan: vi.fn(),
  scannerErrorMessage: () => "Sidecar unavailable",
  scannerDiagnosticMessage: () => "Scanner diagnostic",
}));

describe("FolderScanView", () => {
  let eventHandler: ((event: ScannerEvent) => void) | undefined;

  beforeEach(() => {
    window.localStorage.clear();
    vi.clearAllMocks();
    vi.mocked(open).mockResolvedValue("D:/input");
    vi.mocked(invokeFlatScan).mockResolvedValue({ exitCode: 0 });
    eventHandler = undefined;
    vi.mocked(listenScannerEvents).mockImplementation(async (callback) => {
      eventHandler = callback as (event: ScannerEvent) => void;
      return () => undefined;
    });
  });

  it("starts a flat scan with selected export, orientation, and worker settings", async () => {
    render(<AppShell />);
    fireEvent.click(screen.getByRole("tab", { name: "Thư mục tự do" }));
    fireEvent.click(screen.getByRole("button", { name: "Chọn thư mục" }));

    await waitFor(() => expect(screen.getByDisplayValue("D:/input")).toBeInTheDocument());
    act(() => {
      eventHandler?.({
        protocolVersion: 1,
        type: "flat_scan_plan",
        timestamp: "2026-09-14T00:00:00Z",
        inputRoot: "D:/input",
        outputRoot: "D:/input_pdf",
        exportMode: "MERGED",
        orientation: "auto",
        totalFiles: 5,
        filesToProcess: 5,
         unchangedFiles: 0,
         expectedArtifacts: 1,
         unsupportedCount: 0,
      });
    });

    fireEvent.click(screen.getByRole("radio", { name: "Gộp thành một PDF" }));
    fireEvent.change(screen.getByRole("combobox", { name: "Chiều tài liệu" }), {
      target: { value: "portrait" },
    });
    fireEvent.change(screen.getByRole("combobox", { name: "Số worker" }), {
      target: { value: "1" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Bắt đầu quét thư mục" }));

    await waitFor(() => expect(invokeFlatScan).toHaveBeenCalledWith({
      inputRoot: "D:/input",
      outputRoot: "D:/input_pdf",
      exportMode: "MERGED",
      mode: "gray",
      detectorMode: "ai_enhanced",
      orientation: "portrait",
      workers: 1,
    }));
  });
});
