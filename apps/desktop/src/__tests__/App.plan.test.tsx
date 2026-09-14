import { act, fireEvent, render, screen } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { planScan, startScan } from "@/lib/scannerBridge";
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

const ambiguousGroup = {
  key: { employeeRelativeDir: "NV01", year: 2026, month: 9 },
  sourcePages: [
    {
      sourceRelativePath: "NV01/page-one.png",
      identity: { pageType: "UNKNOWN" as const, pageOrder: null, confidence: 0.4 },
    },
    {
      sourceRelativePath: "NV01/page-two.png",
      identity: { pageType: "UNKNOWN" as const, pageOrder: null, confidence: 0.4 },
    },
  ],
  reasons: ["unknown_page_identity"],
  completenessStatus: "AMBIGUOUS" as const,
  reviewRequired: true,
  manualOrder: [],
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
    window.localStorage.clear();
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
      detectorMode: "ai_enhanced",
      debugDiagnostics: false,
      reprocess: false,
      period: {
        year: new Date().getFullYear(),
        month: new Date().getMonth() + 1,
      },
      exportMode: "PER_IMAGE",
    });
  });

  it("marks plan stale when scan mode changes and replans when synchronizing", async () => {
    vi.mocked(planScan).mockResolvedValue(validPlan);
    render(<App />);

    await enterInputPath();
    expect(planScan).toHaveBeenCalledTimes(1);

    // Changing scan mode updates settings without immediate planScan or results reset
    fireEvent.click(screen.getByRole("button", { name: /Smart Document/i }));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(250);
    });

    // planScan was NOT called again immediately
    expect(planScan).toHaveBeenCalledTimes(1);
    expect(screen.getByText(/Thiết lập đã thay đổi/i)).toBeInTheDocument();

    // Scan button now says "Đồng bộ & Quét"
    const syncScanButton = screen.getByRole("button", { name: /Đồng bộ & Quét/i });
    expect(syncScanButton).toBeInTheDocument();

    // Clicking "Đồng bộ & Quét" triggers plan refresh with new settings
    fireEvent.click(syncScanButton);
    await act(async () => {
      await Promise.resolve();
    });

    expect(planScan).toHaveBeenLastCalledWith({
      inputRoot: "C:/Attendance Input",
      outputRoot: "C:/Attendance Input_pdf",
      mode: "smart_document",
      detectorMode: "ai_enhanced",
      debugDiagnostics: false,
      reprocess: false,
      period: {
        year: new Date().getFullYear(),
        month: new Date().getMonth() + 1,
      },
      exportMode: "PER_IMAGE",
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

  it("shows editable period/export options and sends the selected values to planning on sync", async () => {
    vi.mocked(planScan).mockResolvedValue(validPlan);
    render(<App />);
    await enterInputPath();

    const month = screen.getByLabelText("Tháng xử lý");
    const year = screen.getByLabelText("Năm xử lý");
    expect(month).toHaveValue("9");
    expect(year).toHaveValue(2026);
    expect(screen.getByRole("radio", { name: /Nhiều ảnh → một PDF/i })).toBeInTheDocument();

    fireEvent.change(month, { target: { value: "8" } });
    fireEvent.change(year, { target: { value: "2025" } });
    fireEvent.click(screen.getByRole("radio", { name: /Nhiều ảnh → một PDF/i }));

    // Verify amber banner is shown
    expect(screen.getByText(/Thiết lập đã thay đổi/i)).toBeInTheDocument();

    // Click "Đồng bộ" button in banner
    const syncButton = screen.getByRole("button", { name: /^Đồng bộ$/i });
    fireEvent.click(syncButton);
    await act(async () => {
      await Promise.resolve();
    });

    expect(planScan).toHaveBeenLastCalledWith({
      inputRoot: "C:/Attendance Input",
      outputRoot: "C:/Attendance Input_pdf",
      mode: "gray",
      detectorMode: "ai_enhanced",
      debugDiagnostics: false,
      reprocess: false,
      period: { year: 2025, month: 8 },
      exportMode: "GROUPED",
    });
  });

  it("blocks grouped scanning until every ambiguous group is resolved or skipped", async () => {
    vi.mocked(planScan)
      .mockResolvedValueOnce(validPlan)
      .mockResolvedValueOnce({
        ...validPlan,
        period: { year: 2026, month: 9 },
        exportMode: "GROUPED" as const,
        reviewGroups: [ambiguousGroup],
      });
    vi.mocked(startScan).mockResolvedValue({ exitCode: 0 });
    render(<App />);
    await enterInputPath();

    fireEvent.click(screen.getByRole("radio", { name: /Nhiều ảnh → một PDF/i }));

    // User clicks "Đồng bộ & Quét" to synchronize with GROUPED mode
    const syncScanButton = screen.getByRole("button", { name: /Đồng bộ & Quét/i });
    fireEvent.click(syncScanButton);
    await act(async () => {
      await Promise.resolve();
    });

    // Scan was halted before startScan because ambiguousGroup needs review
    expect(startScan).not.toHaveBeenCalled();
    expect(screen.getByText(/Cần xác nhận thứ tự page/i)).toBeInTheDocument();

    const scanButton = screen.getByRole("button", { name: /Quét các file mới \(4\)/i });
    expect(scanButton).toBeDisabled();

    // User resolves the review group
    fireEvent.click(screen.getByRole("button", { name: /Xác nhận thứ tự page/i }));
    expect(scanButton).toBeEnabled();

    // User clicks scan now that group is resolved
    fireEvent.click(scanButton);
    await act(async () => {
      await Promise.resolve();
    });

    expect(startScan).toHaveBeenCalledWith(
      expect.objectContaining({
        exportMode: "GROUPED",
        manualOrder: {
          "NV01:2026-09": ["NV01/page-one.png", "NV01/page-two.png"],
        },
      }),
    );
  });

  it("drops stale plan response when settings change while plan request is in flight", async () => {
    let resolveFirstPlan: ((value: typeof validPlan) => void) | undefined;
    vi.mocked(planScan).mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveFirstPlan = resolve;
        }),
    );
    render(<App />);

    // Start first plan request
    const input = screen.getByPlaceholderText(/Nhập hoặc chọn đường dẫn thư mục/i);
    fireEvent.change(input, { target: { value: "C:/Attendance Input" } });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(250);
    });

    // While request is in flight, change scan mode
    fireEvent.click(screen.getByRole("button", { name: /Smart Document/i }));

    // Now resolve the first plan request (which had mode: "gray")
    await act(async () => {
      resolveFirstPlan?.(validPlan);
      await Promise.resolve();
    });

    // Verify stale plan response was dropped: plan is NOT marked ready with stale fingerprint
    expect(screen.getByRole("button", { name: /Quét các file mới/i })).toBeDisabled();
  });

  it("releases loading spinner when plan request is invalidated by setting change and allows new plan to run", async () => {
    let resolveFirstPlan: ((value: typeof validPlan) => void) | undefined;
    vi.mocked(planScan).mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveFirstPlan = resolve;
        }),
    );
    render(<App />);

    // Start first plan request
    const input = screen.getByPlaceholderText(/Nhập hoặc chọn đường dẫn thư mục/i);
    fireEvent.change(input, { target: { value: "C:/Attendance Input" } });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(250);
    });

    // Verify loading spinner is active
    expect(screen.getByText(/Đang phân tích thư mục/i)).toBeInTheDocument();

    // While request is in flight, user changes mode
    fireEvent.click(screen.getByRole("button", { name: /Smart Document/i }));

    // Finding P1 #1 Fix: Spinner must be immediately released!
    expect(screen.queryByText(/Đang phân tích thư mục/i)).not.toBeInTheDocument();

    // Now resolve old request
    await act(async () => {
      resolveFirstPlan?.(validPlan);
      await Promise.resolve();
    });

    // Still not loading
    expect(screen.queryByText(/Đang phân tích thư mục/i)).not.toBeInTheDocument();

    // User can now refresh plan with new settings successfully
    vi.mocked(planScan).mockResolvedValueOnce({
      ...validPlan,
      mode: "smart_document",
    });

    const refreshButton = screen.getByRole("button", { name: /Làm mới/i });
    fireEvent.click(refreshButton);
    await act(async () => {
      await Promise.resolve();
    });

    expect(planScan).toHaveBeenLastCalledWith(
      expect.objectContaining({
        mode: "smart_document",
      }),
    );
  });

  it("reschedules plan with latest settings when option changes during input debounce", async () => {
    vi.mocked(planScan).mockResolvedValue(validPlan);
    render(<App />);

    // Type folder path (triggers 250ms debounce)
    const input = screen.getByPlaceholderText(/Nhập hoặc chọn đường dẫn thư mục/i);
    fireEvent.change(input, { target: { value: "C:/Attendance Input" } });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(100);
    });

    // Before 250ms elapses, user changes mode to Smart Document
    fireEvent.click(screen.getByRole("button", { name: /Smart Document/i }));

    // Now advance timers past debounce window
    await act(async () => {
      await vi.advanceTimersByTimeAsync(250);
    });

    // Plan must have been requested with Smart Document (the latest setting)
    expect(planScan).toHaveBeenCalledWith(
      expect.objectContaining({
        mode: "smart_document",
      }),
    );
  });

  it("blocks grouped scan when plan refreshed from stale even if old plan had resolved group with same key", async () => {
    vi.mocked(planScan)
      .mockResolvedValueOnce(validPlan)
      .mockResolvedValueOnce({
        ...validPlan,
        exportMode: "GROUPED" as const,
        reviewGroups: [ambiguousGroup],
      })
      .mockResolvedValueOnce({
        ...validPlan,
        exportMode: "GROUPED" as const,
        reviewGroups: [ambiguousGroup],
      });
    vi.mocked(startScan).mockResolvedValue({ exitCode: 0 });

    render(<App />);
    await enterInputPath();

    // Switch to GROUPED mode
    fireEvent.click(screen.getByRole("radio", { name: /Nhiều ảnh → một PDF/i }));
    const initialSyncButton = screen.getByRole("button", { name: /Đồng bộ & Quét/i });
    fireEvent.click(initialSyncButton);
    await act(async () => {
      await Promise.resolve();
    });

    // In GROUPED plan, user resolves the review group
    expect(screen.getByText(/Cần xác nhận thứ tự page/i)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Xác nhận thứ tự page/i }));

    const scanButton = screen.getByRole("button", { name: /Quét các file mới \(4\)/i });
    expect(scanButton).toBeEnabled();

    // Now user changes detector mode -> makes plan stale!
    fireEvent.click(screen.getByRole("button", { name: /^Classic/i }));

    // Finding P2 #3 Fix: Stale review panel must be hidden!
    expect(screen.queryByText(/Cần xác nhận thứ tự page/i)).not.toBeInTheDocument();

    // User clicks "Đồng bộ & Quét"
    const syncScanButton = screen.getByRole("button", { name: /Đồng bộ & Quét/i });
    fireEvent.click(syncScanButton);
    await act(async () => {
      await Promise.resolve();
    });

    // Finding P1 #2 Fix: startScan must NOT be called! Group in new plan requires review again!
    expect(startScan).not.toHaveBeenCalled();
    expect(screen.getByText(/Cần xác nhận thứ tự page/i)).toBeInTheDocument();
  });

  it("keeps grouped review state when debug diagnostics changes", async () => {
    vi.mocked(planScan).mockResolvedValue({
      ...validPlan,
      exportMode: "GROUPED" as const,
      reviewGroups: [ambiguousGroup],
    });
    render(<App />);
    await enterInputPath();

    fireEvent.click(screen.getByRole("radio", { name: /Nhiều ảnh → một PDF/i }));
    fireEvent.click(screen.getByRole("button", { name: /Đồng bộ & Quét/i }));
    await act(async () => {
      await Promise.resolve();
    });

    expect(screen.getByText(/Cần xác nhận thứ tự page/i)).toBeInTheDocument();
    const planCallCount = vi.mocked(planScan).mock.calls.length;

    fireEvent.click(screen.getByRole("checkbox", { name: /Bật diagnostics/i }));

    expect(screen.getByText(/Cần xác nhận thứ tự page/i)).toBeInTheDocument();
    expect(vi.mocked(planScan)).toHaveBeenCalledTimes(planCallCount);
  });

  it("keeps the newest plan loading while an invalidated older request finishes", async () => {
    let resolveFirstPlan: ((value: typeof validPlan) => void) | undefined;
    let resolveSecondPlan: ((value: typeof validPlan) => void) | undefined;
    vi.mocked(planScan)
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveFirstPlan = resolve;
          }),
      )
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveSecondPlan = resolve;
          }),
      );
    render(<App />);

    const input = screen.getByPlaceholderText(/Nhập hoặc chọn đường dẫn thư mục/i);
    fireEvent.change(input, { target: { value: "C:/Attendance Input" } });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(250);
    });

    fireEvent.click(screen.getByRole("button", { name: /Smart Document/i }));
    fireEvent.click(screen.getByRole("button", { name: /Làm mới/i }));
    expect(screen.getByText(/Đang phân tích thư mục/i)).toBeInTheDocument();

    await act(async () => {
      resolveFirstPlan?.(validPlan);
      await Promise.resolve();
    });
    expect(screen.getByText(/Đang phân tích thư mục/i)).toBeInTheDocument();

    await act(async () => {
      resolveSecondPlan?.(validPlan);
      await Promise.resolve();
    });
    expect(screen.queryByText(/Đang phân tích thư mục/i)).not.toBeInTheDocument();
  });
});
