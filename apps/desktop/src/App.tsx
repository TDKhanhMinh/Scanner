import { useRef, useState } from "react";
import { Sparkles, Scan, FileText, Info } from "lucide-react";
import { AppHeader } from "@/components/scanner/AppHeader";
import { FolderSelectorCard } from "@/components/scanner/FolderSelectorCard";
import { ScanModeSelector, type ScanFilterMode } from "@/components/scanner/ScanModeSelector";
import { ScanPlanSummaryCard, type ScanPlanStats } from "@/components/scanner/ScanPlanSummaryCard";
import { BatchProgressCard } from "@/components/scanner/BatchProgressCard";
import { FileResultList, type FileResultItem } from "@/components/scanner/FileResultList";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  listenScannerDiagnostics,
  listenScannerEvents,
  planScan,
  scannerErrorMessage,
  startScan,
} from "@/lib/scannerBridge";
import type {
  FileCompletedEvent,
  FileFailedEvent,
  ScanPlanEvent,
  ScannerEvent,
} from "@/types/scanner";

function toPlanStats(plan: ScanPlanEvent): ScanPlanStats {
  return {
    totalEmployees: plan.employees,
    totalImages: plan.totalImages,
    newFiles: plan.new,
    modifiedFiles: plan.modified,
    unchangedFiles: plan.unchanged,
    rebuildFiles: plan.rebuild,
  };
}

function sourceFileName(relativePath: string): string {
  return relativePath.split(/[\\/]/).pop() ?? relativePath;
}

function completedResultItem(event: FileCompletedEvent): FileResultItem {
  return {
    id: `${event.relativePath}:${event.timestamp}`,
    employeeName: event.employeeName,
    sourceFile: sourceFileName(event.relativePath),
    targetPdf: event.outputRelativePath,
    status: event.warning ? "warning" : "success",
    documentDetected: event.documentDetected,
    message: event.warning ?? undefined,
    timestamp: event.timestamp,
  };
}

function failedResultItem(event: FileFailedEvent): FileResultItem {
  return {
    id: `${event.relativePath}:${event.timestamp}`,
    employeeName: event.employeeName,
    sourceFile: sourceFileName(event.relativePath),
    targetPdf: "",
    status: "failed",
    documentDetected: false,
    message: `${event.errorCode}: ${event.message}`,
    timestamp: event.timestamp,
  };
}

export function App() {
  const [inputPath, setInputPath] = useState<string>("");
  const [outputPath, setOutputPath] = useState<string>("");
  const [scanMode, setScanMode] = useState<ScanFilterMode>("gray");
  const [activeTab, setActiveTab] = useState<string>("config");

  // Incremental scan plan state loaded from the Tauri scanner bridge.
  const [planStats, setPlanStats] = useState<ScanPlanStats>({
    totalEmployees: 0,
    totalImages: 0,
    newFiles: 0,
    modifiedFiles: 0,
    unchangedFiles: 0,
    rebuildFiles: 0,
  });

  // Batch progress state
  const [isScanning, setIsScanning] = useState<boolean>(false);
  const [isPlanning, setIsPlanning] = useState<boolean>(false);
  const [processedCount, setProcessedCount] = useState<number>(0);
  const [currentFile, setCurrentFile] = useState<string>("");
  const [results, setResults] = useState<FileResultItem[]>([]);
  const [errorMessage, setErrorMessage] = useState<string>("");
  const planRequestId = useRef(0);
  const scanRequestId = useRef(0);
  const planDebounceTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const requestPlan = (path: string, nextOutputPath: string) => {
    const requestId = ++planRequestId.current;
    setIsPlanning(true);
    setErrorMessage("");

    void planScan({
      inputRoot: path,
      outputRoot: nextOutputPath,
      mode: scanMode,
    })
      .then((plan) => {
        if (requestId === planRequestId.current) {
          setPlanStats(toPlanStats(plan));
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
          });
          setErrorMessage(scannerErrorMessage(error));
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
      setPlanStats({
        totalEmployees: 0,
        totalImages: 0,
        newFiles: 0,
        modifiedFiles: 0,
        unchangedFiles: 0,
        rebuildFiles: 0,
      });
    }
  };

  const handleSelectFolder = () => {
    const selected = prompt("Nhập đường dẫn thư mục ảnh nhân viên:", inputPath || "");
    if (selected !== null) {
      handleInputChange(selected);
    }
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
    switch (event.type) {
      case "scan_plan":
        setPlanStats(toPlanStats(event));
        break;
      case "file_started":
        setCurrentFile(event.relativePath);
        break;
      case "file_completed":
        setResults((previous) => [...previous, completedResultItem(event)]);
        setProcessedCount((count) => count + 1);
        break;
      case "file_failed":
        setResults((previous) => [...previous, failedResultItem(event)]);
        setProcessedCount((count) => count + 1);
        break;
      case "scan_completed":
        break;
    }
  };

  const handleStartScan = () => {
    if (!inputPath || isScanning) return;

    if (planDebounceTimer.current !== undefined) {
      clearTimeout(planDebounceTimer.current);
      planDebounceTimer.current = undefined;
    }
    const requestId = ++scanRequestId.current;
    setIsScanning(true);
    setProcessedCount(0);
    setResults([]);
    setCurrentFile("");
    setErrorMessage("");

    void (async () => {
      let unlistenEvents: (() => void) | undefined;
      let unlistenDiagnostics: (() => void) | undefined;
      try {
        unlistenEvents = await listenScannerEvents(handleScannerEvent);
        unlistenDiagnostics = await listenScannerDiagnostics((diagnostic) => {
          if (requestId === scanRequestId.current) {
            setErrorMessage(diagnostic.message);
          }
        });
        await startScan({
          inputRoot: inputPath,
          outputRoot: outputPath || `${inputPath}_pdf`,
          mode: scanMode,
          workers: 3,
        });
      } catch (error: unknown) {
        if (requestId === scanRequestId.current) {
          setErrorMessage(scannerErrorMessage(error));
        }
      } finally {
        unlistenEvents?.();
        unlistenDiagnostics?.();
        if (requestId === scanRequestId.current) {
          setIsScanning(false);
          setCurrentFile("");
        }
      }
    })();
  };

  const handleOpenOutputFolder = () => {
    alert(`Mở thư mục: ${outputPath || `${inputPath}_pdf`}`);
  };

  const successCount = results.filter((r) => r.status === "success").length;
  const warningCount = results.filter((r) => r.status === "warning").length;
  const failedCount = results.filter((r) => r.status === "failed").length;

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
                  onSelectInputFolder={handleSelectFolder}
                  disabled={isScanning}
                />

                <ScanModeSelector
                  mode={scanMode}
                  onSelectMode={setScanMode}
                  disabled={isScanning}
                />

                {(isScanning || processedCount > 0) && (
                  <BatchProgressCard
                    currentFile={currentFile}
                    processedCount={processedCount}
                    totalCount={
                      planStats.newFiles +
                      planStats.modifiedFiles +
                      (planStats.rebuildFiles ?? 0)
                    }
                    successCount={successCount}
                    warningCount={warningCount}
                    failedCount={failedCount}
                    isScanning={isScanning}
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
                  canScan={Boolean(inputPath)}
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
