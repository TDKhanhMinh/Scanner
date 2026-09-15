import { fireEvent, render, screen, waitFor, act } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AppShell } from "@/components/scanner/AppShell";
import { invokeQuickScan, listenScannerEvents } from "@/lib/scannerBridge";
import { open } from "@tauri-apps/plugin-dialog";

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
  invokeQuickScan: vi.fn(),
  listenScannerEvents: vi.fn().mockResolvedValue(() => undefined),
  listenScannerDiagnostics: vi.fn().mockResolvedValue(() => undefined),
  planScan: vi.fn(),
  startScan: vi.fn(),
  invokeFlatScan: vi.fn(),
  scannerErrorMessage: () => "Sidecar unavailable",
  scannerDiagnosticMessage: () => "Scanner diagnostic",
}));

describe("cross-workflow isolation", () => {
  beforeEach(() => {
    window.localStorage.clear();
    vi.clearAllMocks();
    vi.mocked(listenScannerEvents).mockResolvedValue(() => undefined);
  });

  it("locks every workflow while Quick Scan is executing", async () => {
    vi.mocked(open).mockResolvedValue("D:/input/page.jpg");
    let releaseScan!: (value: Awaited<ReturnType<typeof invokeQuickScan>>) => void;
    vi.mocked(invokeQuickScan).mockImplementation(
      () => new Promise((resolve) => {
        releaseScan = resolve;
      }),
    );

    render(<AppShell />);
    fireEvent.click(screen.getByRole("tab", { name: "Quét nhanh" }));
    fireEvent.click(screen.getByRole("button", { name: "Chọn file ảnh" }));

    await waitFor(() => expect(invokeQuickScan).toHaveBeenCalled());
    expect(screen.getByRole("tab", { name: "Hồ sơ nhân sự" })).toBeDisabled();
    expect(screen.getByRole("tab", { name: "Thư mục tự do" })).toBeDisabled();
    expect(screen.getByRole("tab", { name: "Quét nhanh" })).toBeDisabled();
    const attendanceScanButton = screen
      .getAllByRole("button", { hidden: true })
      .find((button) => button.textContent?.includes("Quét các file mới"));
    expect(attendanceScanButton).toBeDefined();
    expect(attendanceScanButton).toBeDisabled();

    await act(async () => {
      releaseScan({
        success: true,
        inputPath: "D:/input/page.jpg",
        tempPdfPath: null,
        savedPdfPath: null,
        isSaved: false,
        documentDetected: true,
        durationMs: 100,
        detectionPreview: null,
        processedPreviewDataUrl: null,
        errorCode: null,
        message: null,
        warning: null,
      });
    });

    await waitFor(() => expect(screen.getByRole("tab", { name: "Hồ sơ nhân sự" })).toBeEnabled());
    fireEvent.click(screen.getByRole("tab", { name: "Hồ sơ nhân sự" }));
    fireEvent.click(screen.getByRole("tab", { name: "Quét nhanh" }));
    expect(screen.getByText("D:/input/page.jpg")).toBeInTheDocument();
  });

  it("keeps Folder Scan settings when switching tabs", async () => {
    vi.mocked(open).mockResolvedValue("D:/flat-input");

    render(<AppShell />);
    fireEvent.click(screen.getByRole("tab", { name: "Thư mục tự do" }));
    fireEvent.click(screen.getByRole("button", { name: "Chọn thư mục" }));
    await waitFor(() => expect(screen.getByDisplayValue("D:/flat-input")).toBeInTheDocument());

    fireEvent.click(screen.getByRole("tab", { name: "Quét nhanh" }));
    fireEvent.click(screen.getByRole("tab", { name: "Thư mục tự do" }));
    expect(screen.getByDisplayValue("D:/flat-input")).toBeInTheDocument();
  });
});
