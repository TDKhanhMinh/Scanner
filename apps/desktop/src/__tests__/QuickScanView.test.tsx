import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { listen } from "@tauri-apps/api/event";
import { AppShell } from "@/components/scanner/AppShell";
import { parentDirectory } from "@/components/scanner/QuickScanView";
import { invokeQuickScan, openOutputFolder, saveQuickScanPdf } from "@/lib/scannerBridge";
import { open, save } from "@tauri-apps/plugin-dialog";
import { openPath } from "@tauri-apps/plugin-opener";

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
  openOutputFolder: vi.fn(),
  saveQuickScanPdf: vi.fn(),
  listenScannerEvents: vi.fn().mockResolvedValue(() => undefined),
  listenScannerDiagnostics: vi.fn().mockResolvedValue(() => undefined),
  planScan: vi.fn(),
  startScan: vi.fn(),
  scannerErrorMessage: () => "Sidecar unavailable",
  scannerDiagnosticMessage: () => "Scanner diagnostic",
}));

const quickResult = {
  success: true,
  inputPath: "D:/input/page.jpg",
  tempPdfPath: "C:/Temp/quick_scan/quick-page.pdf",
  savedPdfPath: null,
  isSaved: false,
  documentDetected: true,
  durationMs: 1200,
  detectionPreview: {
    sourceWidth: 200,
    sourceHeight: 100,
    coordinateSpace: "original_pixels" as const,
    finalCorners: [
      { x: 5, y: 5 },
      { x: 195, y: 5 },
      { x: 195, y: 95 },
      { x: 5, y: 95 },
    ],
    candidateCorners: [],
    maskAvailable: false,
    previewImageDataUrl: "data:image/jpeg;base64,AA==",
    confidence: 0.9,
    confidenceIsCalibrated: false,
    fallbackUsed: false,
    reasonCodes: [],
    warningCodes: [],
  },
  processedPreviewDataUrl: "data:image/jpeg;base64,AA==",
  errorCode: null,
  message: null,
  warning: null,
};

describe("QuickScanView", () => {
  let dragDropHandler: ((event: { payload: { paths: string[] } }) => void) | undefined;

  beforeEach(() => {
    window.localStorage.clear();
    vi.clearAllMocks();
    vi.mocked(open).mockResolvedValue("D:/input/page.jpg");
    vi.mocked(invokeQuickScan).mockResolvedValue(quickResult);
    vi.mocked(save).mockResolvedValue("D:/output/page.pdf");
    vi.mocked(saveQuickScanPdf).mockResolvedValue(undefined);
    vi.mocked(openOutputFolder).mockResolvedValue(undefined);
    dragDropHandler = undefined;
    vi.mocked(listen).mockImplementation(async (channel, callback) => {
      if (channel === "tauri://drag-drop") {
        dragDropHandler = callback as (event: { payload: { paths: string[] } }) => void;
      }
      return () => undefined;
    });
  });

  it("resolves relative saved PDF paths to their containing directories", () => {
    expect(parentDirectory("page.pdf")).toBe(".");
    expect(parentDirectory("C:page.pdf")).toBe("C:.");
    expect(parentDirectory("C:\\page.pdf")).toBe("C:\\");
  });

  it("ignores native drag-drop while Quick Scan is hidden", () => {
    render(<AppShell />);

    dragDropHandler?.({ payload: { paths: ["D:/input/page.jpg"] } });

    expect(invokeQuickScan).not.toHaveBeenCalled();
  });

  it("rejects unsupported native drag-drop files with a safe message", async () => {
    render(<AppShell />);
    fireEvent.click(screen.getByRole("tab", { name: "Quét nhanh" }));

    dragDropHandler?.({ payload: { paths: ["D:/input/notes.txt"] } });

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Quick Scan chỉ hỗ trợ JPG, JPEG, PNG hoặc WebP.",
    );
    expect(invokeQuickScan).not.toHaveBeenCalled();
  });

  it("processes one picked image, renders both previews, and saves then opens the PDF", async () => {
    render(<AppShell />);
    fireEvent.click(screen.getByRole("tab", { name: "Quét nhanh" }));
    fireEvent.click(screen.getByRole("button", { name: "Chọn file ảnh" }));

    await waitFor(() => expect(invokeQuickScan).toHaveBeenCalledWith(
      expect.objectContaining({
        inputPath: "D:/input/page.jpg",
        mode: "gray",
        detectorMode: "ai_enhanced",
        orientation: "auto",
        debugDiagnostics: false,
      }),
    ));
    expect(screen.getByAltText("Ảnh gốc Quick Scan")).toBeInTheDocument();
    expect(screen.getByAltText("Ảnh đã xử lý Quick Scan")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Mở PDF" })).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: "Lưu PDF" }));
    await waitFor(() => expect(saveQuickScanPdf).toHaveBeenCalledWith(
      "C:/Temp/quick_scan/quick-page.pdf",
      "D:/output/page.pdf",
    ));
    expect(screen.getByRole("button", { name: "Mở PDF" })).toBeEnabled();

    fireEvent.click(screen.getByRole("button", { name: "Mở thư mục" }));
    await waitFor(() => expect(openOutputFolder).toHaveBeenCalledWith("D:/output"));

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Mở PDF" }));
    });
    expect(openPath).toHaveBeenCalledWith("D:/output/page.pdf");
  });
});
