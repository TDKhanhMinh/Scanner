import { act, fireEvent, render, screen } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { planScan } from "@/lib/scannerBridge";
import { BatchProgressCard } from "@/components/scanner/BatchProgressCard";

vi.mock("@/lib/scannerBridge", () => ({
  planScan: vi.fn(),
  startScan: vi.fn(),
  listenScannerEvents: vi.fn(),
  listenScannerDiagnostics: vi.fn(),
  scannerErrorMessage: (error: unknown) =>
    error instanceof Error ? error.message : "Sidecar unavailable",
}));

vi.mock("@tauri-apps/plugin-dialog", () => ({
  open: vi.fn(),
}));

const validPlan = {
  protocolVersion: 1,
  type: "scan_plan" as const,
  timestamp: "2026-09-08T00:00:00Z",
  inputRoot: "C:/Attendance Input",
  outputRoot: "C:/Attendance Input_pdf",
  employees: 2,
  totalImages: 5,
  new: 2,
  modified: 1,
  rebuild: 1,
  unchanged: 1,
  filesToProcess: 4,
  collisions: ["NV01/card"],
  outdatedPipelineCount: 0,
  unsupportedCount: 2,
};

async function enterInputPath(path = "C:/Attendance Input") {
  const input = screen.getByPlaceholderText(/Nhập hoặc chọn đường dẫn thư mục/i);
  fireEvent.change(input, { target: { value: path } });
  await act(async () => {
    await vi.advanceTimersByTimeAsync(250);
  });
}

describe("App scan plan states", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("keeps Scan New Files disabled until a folder has a valid plan", () => {
    render(<App />);
    expect(screen.getByRole("button", { name: /Quét các file mới/i })).toBeDisabled();
    expect(screen.getByLabelText("Thư mục ảnh chấm công gốc")).toBeInTheDocument();
    expect(screen.getByText(/Vui lòng nhập hoặc chọn thư mục gốc/i)).toBeInTheDocument();
  });

  it("shows loading while the plan request is pending", async () => {
    let resolvePlan: ((value: typeof validPlan) => void) | undefined;
    vi.mocked(planScan).mockReturnValue(
      new Promise((resolve) => {
        resolvePlan = resolve;
      }),
    );
    render(<App />);

    await enterInputPath();
    expect(screen.getByText(/Đang phân tích thư mục/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Quét các file mới/i })).toBeDisabled();

    await act(async () => {
      resolvePlan?.(validPlan);
    });
  });

  it("renders a valid plan with warnings and enables scanning", async () => {
    vi.mocked(planScan).mockResolvedValue(validPlan);
    render(<App />);

    await enterInputPath();

    expect(screen.getByText("5")).toBeInTheDocument();
    expect(screen.getByText("+2")).toBeInTheDocument();
    expect(screen.getByText(/Bỏ qua 2 file không thuộc định dạng/i)).toBeInTheDocument();
    expect(screen.getByText(/Có 1 tên PDF có nguy cơ trùng/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Quét các file mới \(4\)/i })).toBeEnabled();
    expect(planScan).toHaveBeenCalledWith({
      inputRoot: "C:/Attendance Input",
      outputRoot: "C:/Attendance Input_pdf",
      mode: "gray",
    });
  });

  it("shows a plan error and keeps scanning disabled", async () => {
    vi.mocked(planScan).mockRejectedValue(new Error("Không tìm thấy sidecar"));
    render(<App />);

    await enterInputPath();

    expect(screen.getByRole("alert")).toHaveTextContent("Không tìm thấy sidecar");
    expect(screen.getByRole("button", { name: /Quét các file mới/i })).toBeDisabled();
  });

  it("keeps the completion summary visible when every file was skipped", () => {
    const onOpenOutputFolder = vi.fn();
    render(
      <BatchProgressCard
        currentFile=""
        currentEmployee=""
        processedCount={0}
        totalCount={0}
        successCount={0}
        warningCount={0}
        failedCount={0}
        skippedCount={3}
        isScanning={false}
        isComplete
        outputReady
        onOpenOutputFolder={onOpenOutputFolder}
      />,
    );

    expect(screen.getByText("3")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Mở thư mục kết quả PDF/i })).toBeEnabled();
  });
});
