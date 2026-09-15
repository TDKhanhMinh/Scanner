import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AppShell } from "@/components/scanner/AppShell";
import { invokeFlatScan, listenScannerEvents, openOutputFolder } from "@/lib/scannerBridge";
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
  openOutputFolder: vi.fn(),
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
    vi.mocked(openOutputFolder).mockResolvedValue(undefined);
    eventHandler = undefined;
    vi.mocked(listenScannerEvents).mockImplementation(async (callback) => {
      eventHandler = callback as (event: ScannerEvent) => void;
      return () => undefined;
    });
  });

  it("opens the completed output directory through the Tauri command", async () => {
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
        exportMode: "PER_IMAGE",
        orientation: "auto",
        totalFiles: 1,
        filesToProcess: 1,
        unchangedFiles: 0,
        expectedArtifacts: 1,
        unsupportedCount: 0,
      });
      eventHandler?.({
        protocolVersion: 1,
        type: "flat_scan_completed",
        timestamp: "2026-09-14T00:00:01Z",
        inputRoot: "D:/input",
        outputRoot: "D:/input_pdf",
        exportMode: "PER_IMAGE",
        totalProcessed: 1,
        success: 1,
        warning: 0,
        failed: 0,
        skipped: 0,
        unsupportedCount: 0,
        durationMs: 100,
        exitCode: 0,
        artifactStatus: "committed",
        artifactRelativePath: "page.pdf",
        artifactErrorCode: null,
        artifactMessage: null,
      });
    });

    fireEvent.click(screen.getByRole("button", { name: "Mở thư mục xuất" }));
    await waitFor(() => expect(openOutputFolder).toHaveBeenCalledWith("D:/input_pdf"));
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

  it("places the start action after all folder configuration controls", () => {
    render(<AppShell />);
    fireEvent.click(screen.getByRole("tab", { name: "Thư mục tự do" }));

    const detectorHeading = screen.getByRole("heading", { name: "Chế độ nhận diện tài liệu" });
    const orientation = screen.getByRole("combobox", { name: "Chiều tài liệu" });
    const workers = screen.getByRole("combobox", { name: "Số worker" });
    const startButton = screen.getByRole("button", { name: "Bắt đầu quét thư mục" });
    const follows = (element: Element) =>
      Boolean(element.compareDocumentPosition(startButton) & Node.DOCUMENT_POSITION_FOLLOWING);

    expect(follows(detectorHeading)).toBe(true);
    expect(follows(orientation)).toBe(true);
    expect(follows(workers)).toBe(true);
  });
});
