import { useEffect, useReducer, useRef, useState } from "react";
import { Sparkles, Scan, FileText, Info } from "lucide-react";
import { open } from "@tauri-apps/plugin-dialog";
import { openPath } from "@tauri-apps/plugin-opener";
import { AppHeader } from "@/components/scanner/AppHeader";
import { FolderSelectorCard } from "@/components/scanner/FolderSelectorCard";
import { ScanModeSelector, type ScanFilterMode } from "@/components/scanner/ScanModeSelector";
import { WorkerSettingCard } from "@/components/scanner/WorkerSettingCard";
import { ScanPlanSummaryCard, type ScanPlanStats } from "@/components/scanner/ScanPlanSummaryCard";
import { BatchProgressCard } from "@/components/scanner/BatchProgressCard";
import { FileResultList } from "@/components/scanner/FileResultList";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  listenScannerDiagnostics,
  listenScannerEvents,
  planScan,
  scannerDiagnosticMessage,
  scannerErrorMessage,
  startScan,
} from "@/lib/scannerBridge";
import type {
  ScanPlanEvent,
  ScannerEvent,
} from "@/types/scanner";
import {
  initialScanExecutionState,
  scanExecutionReducer,
} from "@/lib/scanExecutionReducer";

function toPlanStats(plan: ScanPlanEvent): ScanPlanStats {
  return {
    totalEmployees: plan.employees,
    totalImages: plan.totalImages,
    newFiles: plan.new,
    modifiedFiles: plan.modified,
    unchangedFiles: plan.unchanged,
    rebuildFiles: plan.rebuild,
    unsupportedFiles: plan.unsupportedCount,
    collisions: plan.collisions,
  };
}

interface ScannerSettings {
  mode: ScanFilterMode;
  workers: number | null;
}

export function App() {
  const [inputPath, setInputPath] = useState<string>("");
  const [outputPath, setOutputPath] = useState<string>("");
  const [settings, setSettings] = useState<ScannerSettings>({
    mode: "gray",
    workers: null,
  });
  const [activeTab, setActiveTab] = useState<string>("config");

  // Incremental scan plan state loaded from the Tauri scanner bridge.
  const [planStats, setPlanStats] = useState<ScanPlanStats>({
    totalEmployees: 0,
    totalImages: 0,
    newFiles: 0,
    modifiedFiles: 0,
    unchangedFiles: 0,
    rebuildFiles: 0,
    unsupportedFiles: 0,
  });

  // Batch progress state
  const [isPlanning, setIsPlanning] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string>("");
  const [hasPlanError, setHasPlanError] = useState<boolean>(false);
  const [isPlanReady, setIsPlanReady] = useState<boolean>(false);
  const [execution, dispatchExecution] = useReducer(
    scanExecutionReducer,
    initialScanExecutionState,
  );
  const planRequestId = useRef(0);
  const scanRequestId = useRef(0);
  const planDebounceTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const scanMode = settings.mode;
  const isScanning = execution.phase === "running";
  const { currentFile, currentEmployee, processed: processedCount, results } = execution;
  const successCount = execution.success;
  const warningCount = execution.warning;
  const failedCount = execution.failed;

  useEffect(() => {
    return () => {
      if (planDebounceTimer.current !== undefined) {
        clearTimeout(planDebounceTimer.current);
      }
      planRequestId.current += 1;
    };
  }, []);

  const requestPlan = (path: string, nextOutputPath: string) => {
    const requestId = ++planRequestId.current;
    setIsPlanning(true);
    setErrorMessage("");
    setHasPlanError(false);
    setIsPlanReady(false);

    void planScan({
      inputRoot: path,
      outputRoot: nextOutputPath,
      mode: scanMode,
    })
      .then((plan) => {
        if (requestId === planRequestId.current) {
          setPlanStats(toPlanStats(plan));
          setInputPath(plan.inputRoot);
          setOutputPath(plan.outputRoot);
          setIsPlanReady(true);
        }
      })
      .catch((error: unknown) => {
        if (requestId === planRequestId.current) {
          setPlanStats({
            totalEmployees: 0,
            totalImages: 0,
            newFiles: 0,
            modifiedFiles: 0,
            unchangedFiles: 0,
            rebuildFiles: 0,
            unsupportedFiles: 0,
          });
          setErrorMessage(scannerErrorMessage(error));
          setHasPlanError(true);
          setIsPlanReady(false);
        }
      })
      .finally(() => {
        if (requestId === planRequestId.current) {
          setIsPlanning(false);
        }
      });
  };

  const schedulePlan = (path: string, nextOutputPath: string) => {
    if (planDebounceTimer.current !== undefined) {
      clearTimeout(planDebounceTimer.current);
    }
    planDebounceTimer.current = setTimeout(() => {
      planDebounceTimer.current = undefined;
      requestPlan(path, nextOutputPath);
    }, 250);
  };

  // Cập nhật thư mục nhập từ người dùng và tính toán kế hoạch quét
  const handleInputChange = (path: string) => {
    setInputPath(path);
    dispatchExecution({ type: "reset" });
    planRequestId.current += 1;
    setIsPlanReady(false);
    const trimmed = path.trim();
    const nextOutputPath = trimmed ? `${trimmed}_pdf` : "";
    setOutputPath(nextOutputPath);

    if (trimmed) {
      schedulePlan(trimmed, nextOutputPath);
    } else {
      if (planDebounceTimer.current !== undefined) {
        clearTimeout(planDebounceTimer.current);
        planDebounceTimer.current = undefined;
      }
      planRequestId.current += 1;
      setIsPlanning(false);
      setErrorMessage("");
      setHasPlanError(false);
      setIsPlanReady(false);
      setPlanStats({
        totalEmployees: 0,
        totalImages: 0,
        newFiles: 0,
        modifiedFiles: 0,
        unchangedFiles: 0,
        rebuildFiles: 0,
        unsupportedFiles: 0,
      });
    }
  };

  const chooseDirectory = async (title: string): Promise<string | null> => {
    try {
      const selected = await open({
        directory: true,
        multiple: false,
        title,
      });
      return typeof selected === "string" ? selected : null;
    } catch (error: unknown) {
      setErrorMessage(scannerErrorMessage(error));
      return null;
    }
  };

  const handleSelectInputFolder = () => {
    void chooseDirectory("Chọn thư mục ảnh chấm công gốc").then((selected) => {
      if (selected) {
        handleInputChange(selected);
      }
    });
  };

  const handleOutputChange = (path: string) => {
    const input = inputPath.trim();
    const nextOutputPath = path.trim() || (input ? `${input}_pdf` : "");
    dispatchExecution({ type: "reset" });
    planRequestId.current += 1;
    setIsPlanReady(false);
    setHasPlanError(false);
    setOutputPath(nextOutputPath);
    if (input) {
      schedulePlan(input, nextOutputPath);
    }
  };

  const handleSelectOutputFolder = () => {
    void chooseDirectory("Chọn thư mục xuất PDF").then((selected) => {
      if (selected) {
        handleOutputChange(selected);
      }
    });
  };

  const handleResetOutputFolder = () => {
    if (!inputPath.trim()) return;
    handleOutputChange(`${inputPath.trim()}_pdf`);
  };

  const handleRefreshPlan = () => {
    if (!inputPath) return;
    if (planDebounceTimer.current !== undefined) {
      clearTimeout(planDebounceTimer.current);
      planDebounceTimer.current = undefined;
    }
    requestPlan(inputPath, outputPath || `${inputPath}_pdf`);
  };

  const handleScannerEvent = (event: ScannerEvent) => {
    dispatchExecution({ type: "scanner_event", event });
    if (event.type === "scan_plan") {
      setPlanStats(toPlanStats(event));
    }
  };

  const handleStartScan = () => {
    if (!inputPath || isScanning) return;

    if (planDebounceTimer.current !== undefined) {
      clearTimeout(planDebounceTimer.current);
      planDebounceTimer.current = undefined;
    }
    const requestId = ++scanRequestId.current;
    dispatchExecution({
      type: "scan_started",
      totalToProcess: planStats.newFiles + planStats.modifiedFiles + (planStats.rebuildFiles ?? 0),
    });
    setErrorMessage("");

    void (async () => {
      let unlistenEvents: (() => void) | undefined;
      let unlistenDiagnostics: (() => void) | undefined;
      try {
        unlistenEvents = await listenScannerEvents(handleScannerEvent);
        unlistenDiagnostics = await listenScannerDiagnostics((diagnostic) => {
          if (requestId === scanRequestId.current) {
            setErrorMessage(scannerDiagnosticMessage(diagnostic));
          }
        });
        await startScan({
          inputRoot: inputPath,
          outputRoot: outputPath || `${inputPath}_pdf`,
          mode: scanMode,
          workers: settings.workers,
        });
      } catch (error: unknown) {
        if (requestId === scanRequestId.current) {
          const message = scannerErrorMessage(error);
          dispatchExecution({ type: "scan_error", message });
          setErrorMessage(message);
        }
      } finally {
        unlistenEvents?.();
        unlistenDiagnostics?.();
      }
    })();
  };

  const handleOpenOutputFolder = () => {
    const target = outputPath || `${inputPath}_pdf`;
    void openPath(target).catch((error: unknown) => {
      setErrorMessage(scannerErrorMessage(error));
    });
  };

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col antialiased selection:bg-primary/20 selection:text-primary">
      {/* Header Bar */}
      <AppHeader />

      {/* Main Body */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8 space-y-6">
        {/* Banner Informational Notice */}
        <div className="rounded-2xl border border-primary/20 bg-primary/5 p-4 flex items-start gap-3 backdrop-blur-sm">
          <Sparkles className="w-5 h-5 text-primary shrink-0 mt-0.5" />
          <div className="text-xs sm:text-sm text-foreground/90 leading-relaxed">
            <strong className="text-primary font-semibold">Cơ chế quét thông minh (Incremental Scan): </strong>
            Chỉ xử lý các ảnh mới thêm hoặc ảnh nguồn đã bị chỉnh sửa, tự động bỏ qua các ảnh đã tạo PDF thành công trước đó để tối ưu thời gian.
          </div>
        </div>

        {errorMessage && (
          <div
            role="alert"
            className="rounded-2xl border border-destructive/30 bg-destructive/10 p-4 text-sm text-destructive"
          >
            {errorMessage}
          </div>
        )}

        {/* Tab Navigation */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full sm:w-auto grid-cols-2 max-w-md">
            <TabsTrigger value="config" className="flex items-center gap-2">
              <Scan className="w-4 h-4" />
              Cấu hình & Quét
            </TabsTrigger>
            <TabsTrigger value="results" className="flex items-center gap-2">
              <FileText className="w-4 h-4" />
              Kết quả chi tiết
              {results.length > 0 && (
                <span className="ml-1 px-1.5 py-0.2 rounded-full text-[10px] bg-primary text-primary-foreground font-mono">
                  {results.length}
                </span>
              )}
            </TabsTrigger>
          </TabsList>

          {/* TAB 1: CẤU HÌNH & TIẾN ĐỘ QUÉT */}
          <TabsContent value="config" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
              {/* Cột trái (2 cột trên lg/desktop): Thư mục + Chế độ quét */}
              <div className="lg:col-span-2 space-y-6 min-w-0">
                <FolderSelectorCard
                  inputPath={inputPath}
                  outputPath={outputPath}
                  onInputChange={handleInputChange}
                  onOutputChange={handleOutputChange}
                  onSelectInputFolder={handleSelectInputFolder}
                  onSelectOutputFolder={handleSelectOutputFolder}
                  onResetOutputFolder={handleResetOutputFolder}
                  disabled={isScanning}
                />

                <ScanModeSelector
                  mode={scanMode}
                  onSelectMode={(mode) => setSettings((previous) => ({ ...previous, mode }))}
                  disabled={isScanning}
                />

                <WorkerSettingCard
                  workers={settings.workers}
                  onWorkersChange={(workers) =>
                    setSettings((previous) => ({ ...previous, workers }))
                  }
                  disabled={isScanning}
                />

                {(isScanning || processedCount > 0 || execution.phase === "completed") && (
                  <BatchProgressCard
                    currentFile={currentFile}
                    currentEmployee={currentEmployee}
                    processedCount={processedCount}
                    totalCount={
                      planStats.newFiles +
                      planStats.modifiedFiles +
                      (planStats.rebuildFiles ?? 0)
                    }
                    successCount={successCount}
                    warningCount={warningCount}
                    failedCount={failedCount}
                    skippedCount={execution.skipped}
                    isScanning={isScanning}
                    isComplete={execution.phase === "completed"}
                    outputReady={
                      Boolean(outputPath) && isPlanReady && !isPlanning && !hasPlanError
                    }
                    onOpenOutputFolder={handleOpenOutputFolder}
                  />
                )}
              </div>

              {/* Cột phải (1 cột trên lg/desktop): Thống kê Scan Plan & CTA */}
              <div className="lg:col-span-1 min-w-0">
                <ScanPlanSummaryCard
                  stats={planStats}
                  isPlanning={isPlanning}
                  isScanning={isScanning}
                  canScan={Boolean(inputPath.trim()) && isPlanReady && !hasPlanError}
                  onRefreshPlan={handleRefreshPlan}
                  onStartScan={handleStartScan}
                />
              </div>
            </div>
          </TabsContent>

          {/* TAB 2: KẾT QUẢ QUÉT CHI TIẾT */}
          <TabsContent value="results" className="space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <h2 className="text-base font-semibold text-foreground flex items-center gap-2">
                <FileText className="w-4 h-4 text-primary" />
                Danh sách tài liệu đã quét
              </h2>
              <span className="text-xs text-muted-foreground">
                Tổng số đã xử lý: {results.length} file
              </span>
            </div>

            <FileResultList items={results} />
          </TabsContent>
        </Tabs>
      </main>

      {/* Footer */}
      <footer className="border-t border-border/80 bg-background/90 py-4 px-4 sm:px-6 mt-auto backdrop-blur-sm">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between text-xs text-muted-foreground gap-2">
          <div className="flex items-center gap-2">
            <Info className="w-3.5 h-3.5 text-primary" />
            <span>Attendance Scanner Desktop • Kiến trúc Tauri 2 + Python 3.11 Sidecar + OpenCV</span>
          </div>
          <span>Định dạng hỗ trợ: .jpg, .jpeg, .png, .webp</span>
        </div>
      </footer>
    </div>
  );
}

export default App;

// --- Hybrid Responsive Summary ---
// mobile  (default / sm):  layout 1 cột liền mạch, tabs full width, các card xếp chồng tự nhiên
// tablet  (md / lg):       layout 2 cột linh hoạt (65% cấu hình bên trái / 35% tóm tắt bên phải)
// desktop (xl / 2xl):      tối đa max-w-7xl căn giữa, bảng dữ liệu kết quả chi tiết, animation mượt mà
// Interaction:             touch target >= 44px trên tất cả các nút, focus ring rõ ràng, hover states đầy đủ
