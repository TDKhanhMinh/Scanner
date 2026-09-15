import { useEffect, useMemo, useRef, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { FolderOpen, ScanLine } from "lucide-react";
import { DetectorModeSelector } from "@/components/scanner/DetectorModeSelector";
import { ScanModeSelector } from "@/components/scanner/ScanModeSelector";
import { useExecutionCoordinator } from "@/components/scanner/executionCoordinator";
import { invokeFlatScan, openOutputFolder, scannerErrorMessage } from "@/lib/scannerBridge";
import { loadUserPreferences, saveUserPreferences } from "@/lib/userPreferences";
import type {
  DocumentOrientation,
  FlatFileCompletedEvent,
  FlatScanCompletedEvent,
  FlatScanPlanEvent,
  FlatExportMode,
  ProductDetectorMode,
  ScanMode,
  ScannerEvent,
} from "@/types/scanner";

interface FlatScanStats {
  totalFiles: number;
  filesToProcess: number;
  unchangedFiles: number;
  expectedArtifacts: number;
  processed: number;
  sourceProcessed: number;
  success: number;
  warning: number;
  failed: number;
  skipped: number;
  unsupportedCount: number;
  artifactStatus: "not_required" | "committed" | "not_committed";
  artifactMessage: string | null;
  currentFile: string;
  completed: boolean;
}

interface FlatResult {
  relativePath: string;
  outputRelativePath: string | null;
  status: FlatFileCompletedEvent["status"];
  message: string;
  durationMs: number;
}

const EMPTY_STATS: FlatScanStats = {
  totalFiles: 0,
  filesToProcess: 0,
  unchangedFiles: 0,
  expectedArtifacts: 0,
  processed: 0,
  sourceProcessed: 0,
  success: 0,
  warning: 0,
  failed: 0,
  skipped: 0,
  unsupportedCount: 0,
  artifactStatus: "not_required",
  artifactMessage: null,
  currentFile: "",
  completed: false,
};

export function FolderScanView() {
  const preferences = useMemo(loadUserPreferences, []);
  const [inputRoot, setInputRoot] = useState("");
  const [outputRoot, setOutputRoot] = useState("");
  const [exportMode, setExportMode] = useState<FlatExportMode>("PER_IMAGE");
  const [mode, setMode] = useState<ScanMode>(preferences.mode);
  const [detectorMode, setDetectorMode] = useState<ProductDetectorMode>(preferences.detectorMode);
  const [orientation, setOrientation] = useState<DocumentOrientation>("auto");
  const [workers, setWorkers] = useState(3);
  const [stats, setStats] = useState<FlatScanStats>(EMPTY_STATS);
  const [results, setResults] = useState<FlatResult[]>([]);
  const [errorMessage, setErrorMessage] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const { isAnyExecuting, beginExecution, endExecution, registerEventHandler } =
    useExecutionCoordinator();
  const eventHandlerRef = useRef<(event: ScannerEvent) => void>(() => undefined);
  const isBusy = isProcessing || isAnyExecuting;

  useEffect(() => {
    eventHandlerRef.current = (event) => {
      if (event.type === "flat_scan_plan") {
        const plan: FlatScanPlanEvent = event;
        setOutputRoot(plan.outputRoot);
        setStats({
          totalFiles: plan.totalFiles,
          filesToProcess: plan.filesToProcess,
          unchangedFiles: plan.unchangedFiles,
          expectedArtifacts: plan.expectedArtifacts,
           processed: 0,
           sourceProcessed: 0,
          success: 0,
          warning: 0,
          failed: 0,
           skipped: plan.unchangedFiles,
           unsupportedCount: plan.unsupportedCount,
           artifactStatus: "not_required",
           artifactMessage: null,
           currentFile: "",
           completed: false,
        });
        setResults([]);
        return;
      }
      if (event.type === "flat_scan_progress") {
        setStats((current) => ({
          ...current,
          sourceProcessed: Math.max(current.sourceProcessed, event.completedSources),
          currentFile: event.relativePath,
        }));
        return;
      }
      if (event.type === "flat_file_completed") {
        const file: FlatFileCompletedEvent = event;
        setResults((current) => [
          ...current.filter((item) => item.relativePath !== file.relativePath),
          {
            relativePath: file.relativePath,
            outputRelativePath: file.outputRelativePath,
            status: file.status,
            message: file.message ?? file.warning ?? "",
            durationMs: file.durationMs,
          },
        ]);
        setStats((current) => ({
           ...current,
           processed: current.processed + (file.status === "failed" ? 1 : 1),
           sourceProcessed: Math.max(current.sourceProcessed, current.processed + 1),
          success: current.success + (file.status === "success" ? 1 : 0),
          warning: current.warning + (file.status === "warning" ? 1 : 0),
          failed: current.failed + (file.status === "failed" ? 1 : 0),
        }));
        return;
      }
      if (event.type === "flat_scan_completed") {
        const completed: FlatScanCompletedEvent = event;
        setStats((current) => ({
          ...current,
          processed: completed.totalProcessed,
          success: completed.success,
          warning: completed.warning,
           failed: completed.failed,
           skipped: completed.skipped,
           unsupportedCount: completed.unsupportedCount,
           artifactStatus: completed.artifactStatus,
           artifactMessage: completed.artifactMessage,
           currentFile: "",
           completed: true,
        }));
      }
    };
  }, []);

  useEffect(() => {
    return registerEventHandler("folder_scan", (event) => eventHandlerRef.current(event));
  }, [registerEventHandler]);

  useEffect(() => {
    saveUserPreferences({ mode, detectorMode });
  }, [detectorMode, mode]);

  const handleChooseFolder = async () => {
    if (isBusy) return;
    const selected = await open({ directory: true, multiple: false, title: "Chọn thư mục ảnh" });
    if (typeof selected === "string") {
      setInputRoot(selected);
      setOutputRoot(`${selected.replace(/[\\/]+$/, "")}_pdf`);
      setStats(EMPTY_STATS);
      setResults([]);
      setErrorMessage("");
    }
  };

  const handleStartScan = async () => {
    const input = inputRoot.trim();
    if (!input || isBusy || !beginExecution("folder_scan")) return;
    const output = outputRoot.trim() || `${input}_pdf`;
    setErrorMessage("");
    setStats(EMPTY_STATS);
    setResults([]);
    setIsProcessing(true);
    try {
      await invokeFlatScan({
        inputRoot: input,
        outputRoot: output,
        exportMode,
        mode,
        detectorMode,
        orientation,
        workers,
      });
    } catch (error: unknown) {
      setErrorMessage(scannerErrorMessage(error));
    } finally {
      setIsProcessing(false);
      endExecution("folder_scan");
    }
  };

  const handleOpenOutputFolder = async () => {
    const target = outputRoot.trim();
    if (!target || isBusy || !stats.completed) return;
    try {
      await openOutputFolder(target);
    } catch (error: unknown) {
      setErrorMessage(scannerErrorMessage(error));
    }
  };

  const progress = stats.filesToProcess > 0
    ? Math.min(100, Math.round((Math.max(stats.processed, stats.sourceProcessed) / stats.filesToProcess) * 100))
    : stats.completed ? 100 : 0;

  return (
    <main className="mx-auto max-w-[1440px] space-y-6 p-4 sm:p-6 lg:p-8">
      <section className="rounded-2xl border border-border/80 bg-card p-5 shadow-sm">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h1 className="flex items-center gap-2 text-xl font-semibold">
              <FolderOpen className="h-5 w-5 text-primary" />
              Quét thư mục tự do
            </h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Quét các ảnh nằm trực tiếp trong một thư mục, không cần cấu trúc nhân sự.
            </p>
          </div>
          <button
            type="button"
            onClick={() => void handleChooseFolder()}
            disabled={isBusy}
            className="inline-flex min-h-11 items-center justify-center gap-2 rounded-lg bg-primary px-4 text-sm font-semibold text-primary-foreground disabled:cursor-not-allowed disabled:opacity-50"
          >
            <FolderOpen className="h-4 w-4" />
            Chọn thư mục
          </button>
        </div>
        <div className="mt-5 grid gap-3 sm:grid-cols-2">
          <label className="text-sm">
            <span className="font-semibold">Thư mục nguồn</span>
            <input
              value={inputRoot}
              name="flat-input-root"
              autoComplete="off"
              onChange={(event) => setInputRoot(event.target.value)}
              disabled={isBusy}
              className="mt-2 min-h-11 w-full rounded-lg border border-border bg-background px-3 font-mono text-xs"
              placeholder="D:/Documents/images"
            />
          </label>
          <label className="text-sm">
            <span className="font-semibold">Thư mục xuất PDF (thư mục tự do)</span>
            <input
              aria-label="Thư mục xuất PDF thư mục tự do"
              value={outputRoot}
              name="flat-output-root"
              autoComplete="off"
              onChange={(event) => setOutputRoot(event.target.value)}
              disabled={isBusy}
              className="mt-2 min-h-11 w-full rounded-lg border border-border bg-background px-3 font-mono text-xs"
              placeholder="D:/Documents/images_pdf"
            />
          </label>
        </div>
      </section>

      <section className="space-y-6">
        <section className="space-y-4" aria-label="Thiết lập xử lý ảnh">
          <ScanModeSelector mode={mode} onSelectMode={setMode} disabled={isBusy} />
          <DetectorModeSelector mode={detectorMode} onSelectMode={setDetectorMode} disabled={isBusy} />
        </section>

        <div className="grid items-start gap-6 2xl:grid-cols-[minmax(0,1fr)_22rem]">
          <div className="min-w-0 space-y-4">
          <div className="rounded-2xl border border-border/80 bg-card p-5">
            <fieldset>
              <legend className="text-sm font-semibold">Cách xuất PDF</legend>
              <div className="mt-3 grid gap-2 sm:grid-cols-2">
                {([
                  ["PER_IMAGE", "Mỗi ảnh một PDF"],
                  ["MERGED", "Gộp thành một PDF"],
                ] as const).map(([value, label]) => (
                  <label key={value} className="flex min-h-14 items-center gap-3 rounded-xl border border-border/80 p-3 text-sm">
                    <input
                      type="radio"
                      name="flat-export-mode"
                      checked={exportMode === value}
                      onChange={() => setExportMode(value)}
                      disabled={isBusy}
                    />
                    {label}
                  </label>
                ))}
              </div>
            </fieldset>
            <div
              className="mt-5 h-3 overflow-hidden rounded-full bg-secondary"
              role="progressbar"
              aria-label="Tiến độ quét"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={progress}
            >
              <div className="h-full bg-primary transition-all" style={{ width: `${progress}%` }} />
            </div>
            <div className="mt-2 flex justify-between text-xs text-muted-foreground" aria-live="polite">
              <span>{Math.max(stats.processed, stats.sourceProcessed)}/{stats.filesToProcess} file cần xử lý</span>
              <span>{progress}%</span>
            </div>
            {stats.currentFile && (
              <p className="mt-2 truncate text-xs text-muted-foreground" aria-live="polite">
                Đang xử lý: <span className="font-mono">{stats.currentFile}</span>
              </p>
            )}
          </div>

          {errorMessage && (
            <div role="alert" className="rounded-xl border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
              {errorMessage}
            </div>
          )}

          {stats.completed && stats.artifactMessage && (
            <div role="status" className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-3 text-sm text-amber-800">
              {stats.artifactMessage}
              {stats.unsupportedCount > 0 && ` Đã bỏ qua ${stats.unsupportedCount} file không hỗ trợ.`}
            </div>
          )}

          <div className="rounded-2xl border border-border/80 bg-card p-5">
            <h2 className="text-sm font-semibold">Kết quả file</h2>
            {results.length === 0 ? (
              <p className="mt-3 text-xs text-muted-foreground">
                {stats.completed && stats.totalFiles === 0
                  ? stats.unsupportedCount > 0
                    ? `Không có ảnh được hỗ trợ. Đã bỏ qua ${stats.unsupportedCount} file.`
                    : "Thư mục không có ảnh để quét."
                  : "Chưa có file hoàn tất."}
              </p>
            ) : (
              <ul className="mt-3 divide-y divide-border/60 text-xs">
                {results.map((item) => (
                   <li key={item.relativePath} className="flex items-start justify-between gap-3 py-2">
                     <span className="min-w-0 flex-1 break-words font-mono">
                       {item.relativePath}
                       {item.outputRelativePath && (
                         <span className="ml-2 text-muted-foreground">→ {item.outputRelativePath}</span>
                       )}
                       {item.message && <span className="ml-2 font-sans text-muted-foreground">({item.message})</span>}
                     </span>
                     <span className={item.status === "failed" ? "text-destructive" : item.status === "warning" ? "text-amber-600" : "text-emerald-600"}>
                       {item.status}
                     </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
          </div>

          <aside className="min-w-0 space-y-4 2xl:sticky 2xl:top-24 2xl:self-start">
          <label className="block rounded-xl border border-border/80 bg-card p-4 text-sm">
            <span className="font-semibold">Chiều tài liệu</span>
            <select
              value={orientation}
              onChange={(event) => setOrientation(event.target.value as DocumentOrientation)}
              disabled={isBusy}
              className="mt-2 min-h-11 w-full rounded-lg border border-border bg-background px-3"
            >
              <option value="auto">Tự động</option>
              <option value="landscape">Khổ ngang</option>
              <option value="portrait">Khổ dọc</option>
            </select>
          </label>
          <label className="block rounded-xl border border-border/80 bg-card p-4 text-sm">
            <span className="font-semibold">Số worker</span>
            <select
              value={workers}
              onChange={(event) => setWorkers(Number(event.target.value))}
              disabled={isBusy}
              className="mt-2 min-h-11 w-full rounded-lg border border-border bg-background px-3"
            >
              {[1, 2, 3, 4].map((count) => <option key={count} value={count}>{count} worker</option>)}
            </select>
          </label>
          <button
            type="button"
            onClick={() => void handleOpenOutputFolder()}
            disabled={!outputRoot || !stats.completed || isBusy}
            className="inline-flex min-h-11 w-full items-center justify-center gap-2 rounded-lg border border-border bg-card px-4 text-sm font-semibold disabled:cursor-not-allowed disabled:opacity-50"
          >
            <FolderOpen className="h-4 w-4" />
            Mở thư mục xuất
          </button>
          </aside>
        </div>

        <section className="rounded-2xl border border-primary/20 bg-primary/5 p-5" aria-label="Bắt đầu quét">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="min-w-0">
              <h2 className="text-base font-semibold">Đã cấu hình xong?</h2>
              <p className="mt-1 text-sm text-muted-foreground">
                Kiểm tra lại các tùy chọn phía trên, sau đó bắt đầu xử lý thư mục ảnh.
              </p>
            </div>
            <button
              type="button"
              onClick={() => void handleStartScan()}
              disabled={!inputRoot.trim() || isBusy}
              className="inline-flex min-h-11 shrink-0 items-center justify-center gap-2 rounded-lg bg-primary px-5 text-sm font-semibold text-primary-foreground shadow-sm disabled:cursor-not-allowed disabled:opacity-50"
            >
              <ScanLine className="h-4 w-4" />
              {isProcessing ? "Đang quét..." : "Bắt đầu quét thư mục"}
            </button>
          </div>
        </section>
      </section>
    </main>
  );
}
