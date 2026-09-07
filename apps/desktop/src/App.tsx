import { useState } from "react";
import { Sparkles, Scan, FileText, Info } from "lucide-react";
import { AppHeader } from "@/components/scanner/AppHeader";
import { FolderSelectorCard } from "@/components/scanner/FolderSelectorCard";
import { ScanModeSelector, type ScanFilterMode } from "@/components/scanner/ScanModeSelector";
import { ScanPlanSummaryCard, type ScanPlanStats } from "@/components/scanner/ScanPlanSummaryCard";
import { BatchProgressCard } from "@/components/scanner/BatchProgressCard";
import { FileResultList, type FileResultItem } from "@/components/scanner/FileResultList";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";

export function App() {
  const [inputPath, setInputPath] = useState<string>("");
  const [outputPath, setOutputPath] = useState<string>("");
  const [scanMode, setScanMode] = useState<ScanFilterMode>("gray");
  const [activeTab, setActiveTab] = useState<string>("config");

  // Mock initial scan plan state (sẽ tích hợp Tauri IPC ở Task AS-14/15)
  const [planStats, setPlanStats] = useState<ScanPlanStats>({
    totalEmployees: 0,
    totalImages: 0,
    newFiles: 0,
    modifiedFiles: 0,
    unchangedFiles: 0,
  });

  // Batch progress state
  const [isScanning, setIsScanning] = useState<boolean>(false);
  const [isPlanning, setIsPlanning] = useState<boolean>(false);
  const [processedCount, setProcessedCount] = useState<number>(0);
  const [currentFile, setCurrentFile] = useState<string>("");
  const [results, setResults] = useState<FileResultItem[]>([]);

  // Giả lập chọn thư mục demo cho frontend
  const handleSelectFolder = () => {
    const demoPath = "D:\\ChamCong\\all";
    setInputPath(demoPath);
    setOutputPath(`${demoPath}_pdf`);

    // Phân tích scan plan mẫu
    setIsPlanning(true);
    setTimeout(() => {
      setPlanStats({
        totalEmployees: 4,
        totalImages: 12,
        newFiles: 3,
        modifiedFiles: 1,
        unchangedFiles: 8,
      });
      setIsPlanning(false);
    }, 400);
  };

  const handleRefreshPlan = () => {
    if (!inputPath) return;
    setIsPlanning(true);
    setTimeout(() => {
      setIsPlanning(false);
    }, 300);
  };

  const handleStartScan = () => {
    setIsScanning(true);
    setProcessedCount(0);
    setResults([]);

    const demoItems: FileResultItem[] = [
      {
        id: "1",
        employeeName: "Nguyen Van A",
        sourceFile: "2026-09.jpg",
        targetPdf: "Nguyen Van A/2026-09.pdf",
        status: "success",
        documentDetected: true,
      },
      {
        id: "2",
        employeeName: "Tran Thi B",
        sourceFile: "2026-09.png",
        targetPdf: "Tran Thi B/2026-09.pdf",
        status: "warning",
        documentDetected: false,
        message: "Không nhận diện đủ 4 góc tài liệu, fallback sang xử lý toàn bộ ảnh nguồn.",
      },
      {
        id: "3",
        employeeName: "Le Van C",
        sourceFile: "2026-08_modified.jpg",
        targetPdf: "Le Van C/2026-08_modified.pdf",
        status: "success",
        documentDetected: true,
      },
      {
        id: "4",
        employeeName: "Pham Thi D",
        sourceFile: "2026-09.jpeg",
        targetPdf: "Pham Thi D/2026-09.pdf",
        status: "success",
        documentDetected: true,
      },
    ];

    let currentStep = 0;
    const interval = setInterval(() => {
      if (currentStep < demoItems.length) {
        const item = demoItems[currentStep];
        setCurrentFile(`${item.employeeName}/${item.sourceFile}`);
        setResults((prev) => [...prev, item]);
        setProcessedCount(currentStep + 1);
        currentStep++;
      } else {
        clearInterval(interval);
        setIsScanning(false);
        setCurrentFile("");
      }
    }, 500);
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
                    totalCount={planStats.newFiles + planStats.modifiedFiles}
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
