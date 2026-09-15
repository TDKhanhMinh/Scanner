import { useCallback, useEffect, useMemo, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { open, save } from "@tauri-apps/plugin-dialog";
import { openPath } from "@tauri-apps/plugin-opener";
import { ScanLine, Upload, FileDown, ExternalLink, RefreshCw, FolderOpen } from "lucide-react";
import { DetectorModeSelector } from "@/components/scanner/DetectorModeSelector";
import { ScanModeSelector } from "@/components/scanner/ScanModeSelector";
import { useExecutionCoordinator } from "@/components/scanner/executionCoordinator";
import {
  invokeQuickScan,
  openOutputFolder,
  saveQuickScanPdf,
  scannerErrorMessage,
} from "@/lib/scannerBridge";
import {
  DEFAULT_USER_PREFERENCES,
  loadUserPreferences,
  saveUserPreferences,
} from "@/lib/userPreferences";
import type { DocumentOrientation, ProductDetectorMode, ScanMode } from "@/types/scanner";
import type { QuickScanResult } from "@/types/workflow";

interface NativeDragDropPayload {
  paths?: unknown;
}

function isImagePath(path: string): boolean {
  return /\.(jpg|jpeg|png|webp)$/i.test(path);
}

function polygonPoints(
  points: Array<{ x: number; y: number }>,
  width: number,
  height: number,
): string {
  return points
    .map((point) => `${(point.x / width) * 100},${(point.y / height) * 100}`)
    .join(" ");
}

function sourceName(path: string): string {
  return path.split(/[\\/]/).pop() ?? path;
}

export function parentDirectory(path: string): string {
  const normalized = path.replace(/[\\/]+$/, "");
  const separator = Math.max(normalized.lastIndexOf("\\"), normalized.lastIndexOf("/"));
  if (separator < 0) {
    return /^[A-Za-z]:/.test(normalized) ? `${normalized.slice(0, 2)}.` : ".";
  }
  if (separator === 2 && /^[A-Za-z]:[\\/]$/.test(normalized.slice(0, 3))) {
    return normalized.slice(0, 3);
  }
  return separator > 0 ? normalized.slice(0, separator) : normalized;
}

export function QuickScanView() {
  const preferences = useMemo(loadUserPreferences, []);
  const [inputPath, setInputPath] = useState("");
  const [mode, setMode] = useState<ScanMode>(preferences.mode ?? DEFAULT_USER_PREFERENCES.mode);
  const [detectorMode, setDetectorMode] = useState<ProductDetectorMode>(
    preferences.detectorMode ?? DEFAULT_USER_PREFERENCES.detectorMode,
  );
  const [orientation, setOrientation] = useState<DocumentOrientation>("auto");
  const [debugDiagnostics, setDebugDiagnostics] = useState(preferences.debugDiagnostics);
  const [result, setResult] = useState<QuickScanResult | null>(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const {
    activeWorkflow,
    isAnyExecuting,
    beginExecution,
    endExecution,
  } = useExecutionCoordinator();
  const isBusy = isProcessing || isAnyExecuting;

  useEffect(() => {
    saveUserPreferences({ mode, detectorMode, debugDiagnostics });
  }, [debugDiagnostics, detectorMode, mode]);

  const processImage = useCallback(
    async (path: string) => {
      const trimmed = path.trim();
      if (!trimmed) return;
      if (!isImagePath(trimmed)) {
        setErrorMessage("Quick Scan chỉ hỗ trợ JPG, JPEG, PNG hoặc WebP.");
        return;
      }
      if (!beginExecution("quick_scan")) return;
      setInputPath(trimmed);
      setResult(null);
      setErrorMessage("");
      setIsProcessing(true);
      try {
        const nextResult = await invokeQuickScan({
          inputPath: trimmed,
          mode,
          detectorMode,
          orientation,
          debugDiagnostics,
        });
        setResult(nextResult);
        if (!nextResult.success) {
          setErrorMessage(nextResult.message ?? "Không thể xử lý ảnh.");
        }
      } catch (error: unknown) {
        setErrorMessage(scannerErrorMessage(error));
      } finally {
        setIsProcessing(false);
        endExecution("quick_scan");
      }
    },
    [beginExecution, debugDiagnostics, detectorMode, endExecution, mode, orientation],
  );

  useEffect(() => {
    let mounted = true;
    let unlisten: (() => void) | undefined;
    try {
      void listen<NativeDragDropPayload>("tauri://drag-drop", (event) => {
        if (!mounted || isBusy || activeWorkflow !== "quick_scan") return;
        const paths = event.payload.paths;
        const firstPath = Array.isArray(paths) && typeof paths[0] === "string" ? paths[0] : null;
        if (firstPath) void processImage(firstPath);
      })
        .then((cleanup) => {
          if (mounted) unlisten = cleanup;
          else cleanup();
        })
        .catch(() => undefined);
    } catch {
      // Browser-based previews/tests do not expose Tauri drag-drop internals.
    }
    return () => {
      mounted = false;
      unlisten?.();
    };
  }, [activeWorkflow, isBusy, processImage]);

  const handlePickImage = async () => {
    if (isBusy) return;
    const selected = await open({
      directory: false,
      multiple: false,
      filters: [{ name: "Images", extensions: ["jpg", "jpeg", "png", "webp"] }],
    });
    if (typeof selected === "string") void processImage(selected);
  };

  const handleSavePdf = async () => {
    if (!result?.success || !result.tempPdfPath || result.isSaved || isBusy) return;
    const selected = await save({
      defaultPath: `${sourceName(result.inputPath).replace(/\.[^.]+$/, "")}.pdf`,
      filters: [{ name: "PDF", extensions: ["pdf"] }],
    });
    if (typeof selected !== "string") return;
    setErrorMessage("");
    if (!beginExecution("quick_scan")) return;
    setIsProcessing(true);
    try {
      await saveQuickScanPdf(result.tempPdfPath, selected);
      setResult({ ...result, savedPdfPath: selected, isSaved: true, tempPdfPath: null });
    } catch (error: unknown) {
      setErrorMessage(scannerErrorMessage(error));
    } finally {
      setIsProcessing(false);
      endExecution("quick_scan");
    }
  };

  const handleOpenPdf = () => {
    if (!result?.isSaved || !result.savedPdfPath) return;
    void openPath(result.savedPdfPath).catch(() => {
      setErrorMessage("Không thể mở file PDF đã lưu.");
    });
  };

  const handleOpenOutputFolder = async () => {
    if (!result?.isSaved || !result.savedPdfPath || isBusy) return;
    setErrorMessage("");
    if (!beginExecution("quick_scan")) return;
    setIsProcessing(true);
    try {
      await openOutputFolder(parentDirectory(result.savedPdfPath));
    } catch (error: unknown) {
      setErrorMessage(scannerErrorMessage(error));
    } finally {
      setIsProcessing(false);
      endExecution("quick_scan");
    }
  };

  const preview = result?.detectionPreview;
  const originalPreview = preview?.previewImageDataUrl;
  const finalCorners = preview?.finalCorners;

  return (
    <main className="mx-auto max-w-[1440px] space-y-6 p-4 sm:p-6 lg:p-8">
      <section className="rounded-2xl border border-border/80 bg-card p-5 shadow-sm">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h1 className="flex items-center gap-2 text-xl font-semibold">
              <ScanLine className="h-5 w-5 text-primary" />
              Quét nhanh một ảnh
            </h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Xử lý, xem trước và lưu một tài liệu mà không cần tạo manifest.
            </p>
          </div>
          <button
            type="button"
            onClick={() => void handlePickImage()}
            disabled={isBusy}
            className="inline-flex min-h-11 items-center justify-center gap-2 rounded-lg bg-primary px-4 text-sm font-semibold text-primary-foreground disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Upload className="h-4 w-4" />
            Chọn file ảnh
          </button>
        </div>

        <div className="mt-5 rounded-xl border-2 border-dashed border-primary/30 bg-primary/5 p-8 text-center text-sm text-muted-foreground">
          <p>Thả một file ảnh vào cửa sổ ứng dụng để bắt đầu Quick Scan.</p>
          {inputPath && <p className="mt-2 break-all font-mono text-xs text-foreground">{inputPath}</p>}
          {isProcessing && <p className="mt-2 text-primary">Đang xử lý ảnh...</p>}
        </div>
      </section>

      <section className="space-y-6">
        <div className="space-y-4">
          <div className="grid min-w-0 gap-6 lg:grid-cols-2">
            <div className="min-w-0 rounded-2xl border border-border/80 bg-card p-4">
              <h2 className="mb-3 text-sm font-semibold">Ảnh gốc và góc nhận diện</h2>
              {originalPreview ? (
                <div className="relative overflow-hidden rounded-xl bg-secondary/40">
                  <img
                    src={originalPreview}
                    alt="Ảnh gốc Quick Scan"
                    width={preview?.sourceWidth}
                    height={preview?.sourceHeight}
                    className="block h-auto w-full"
                  />
                  {finalCorners && preview && (
                    <svg
                      viewBox={`0 0 ${preview.sourceWidth} ${preview.sourceHeight}`}
                      className="pointer-events-none absolute inset-0 h-full w-full"
                      preserveAspectRatio="none"
                      aria-label="Khung bốn góc tài liệu"
                    >
                      <polygon
                        points={polygonPoints(finalCorners, preview.sourceWidth, preview.sourceHeight)}
                        fill="none"
                        stroke={preview.fallbackUsed ? "#f59e0b" : "#22c55e"}
                        strokeWidth={Math.max(preview.sourceWidth, preview.sourceHeight) * 0.006}
                      />
                    </svg>
                  )}
                </div>
              ) : (
                <div className="flex min-h-48 items-center justify-center rounded-xl bg-secondary/40 p-4 text-center text-xs text-muted-foreground">
                  Preview ảnh gốc sẽ hiển thị sau khi xử lý.
                </div>
              )}
            </div>

            <div className="min-w-0 rounded-2xl border border-border/80 bg-card p-4">
              <h2 className="mb-3 text-sm font-semibold">Ảnh đã nắn phẳng và làm sạch</h2>
              {result?.processedPreviewDataUrl ? (
                <img
                  src={result.processedPreviewDataUrl}
                  alt="Ảnh đã xử lý Quick Scan"
                  className="block h-auto w-full rounded-xl bg-secondary/40"
                  loading="lazy"
                />
              ) : (
                <div className="flex min-h-48 items-center justify-center rounded-xl bg-secondary/40 p-4 text-center text-xs text-muted-foreground">
                  Preview ảnh xử lý sẽ hiển thị sau khi hoàn tất.
                </div>
              )}
            </div>
          </div>

          {errorMessage && (
            <div role="alert" className="rounded-xl border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
              {errorMessage}
            </div>
          )}
          {result?.warning && (
            <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-3 text-sm text-amber-800 dark:text-amber-300">
              {result.warning}
            </div>
          )}
        </div>

        <section className="space-y-4" aria-label="Thiết lập xử lý ảnh">
          <ScanModeSelector mode={mode} onSelectMode={setMode} disabled={isBusy} />
          <DetectorModeSelector mode={detectorMode} onSelectMode={setDetectorMode} disabled={isBusy} />
        </section>

        <section className="grid items-start gap-4 md:grid-cols-2 xl:grid-cols-3" aria-label="Tùy chọn và thao tác">
          <label className="block rounded-xl border border-border/80 bg-card p-4 text-sm">
            <span className="font-semibold">Chiều tài liệu</span>
            <select
              value={orientation}
              onChange={(event) => setOrientation(event.target.value as DocumentOrientation)}
              disabled={isBusy}
              className="mt-2 min-h-11 w-full rounded-lg border border-border bg-background px-3 text-sm"
            >
              <option value="auto">Tự động (khuyến nghị)</option>
              <option value="landscape">Khổ ngang</option>
              <option value="portrait">Khổ dọc</option>
            </select>
          </label>
          <label className="flex items-center gap-2 rounded-xl border border-border/80 bg-card p-4 text-sm">
            <input
              type="checkbox"
              checked={debugDiagnostics}
              onChange={(event) => setDebugDiagnostics(event.target.checked)}
              disabled={isBusy}
              className="h-4 w-4 accent-primary"
            />
            Bật diagnostics
          </label>
          <div className="rounded-xl border border-border/80 bg-card p-4 md:col-span-2 xl:col-span-2">
            <div className="grid gap-2 sm:grid-cols-4 lg:grid-cols-1">
              <button
                type="button"
                onClick={() => void handleSavePdf()}
                disabled={!result?.success || !result.tempPdfPath || result.isSaved || isBusy}
                className="inline-flex min-h-11 items-center justify-center gap-2 rounded-lg bg-primary px-3 text-sm font-semibold text-primary-foreground disabled:cursor-not-allowed disabled:opacity-50"
              >
                <FileDown className="h-4 w-4" />
                Lưu PDF
              </button>
              <button
                type="button"
                onClick={handleOpenPdf}
                disabled={!result?.isSaved || isBusy}
                className="inline-flex min-h-11 items-center justify-center gap-2 rounded-lg border border-border px-3 text-sm font-semibold disabled:cursor-not-allowed disabled:opacity-50"
              >
                <ExternalLink className="h-4 w-4" />
                Mở PDF
              </button>
              <button
                type="button"
                onClick={() => void handleOpenOutputFolder()}
                disabled={!result?.isSaved || !result.savedPdfPath || isBusy}
                className="inline-flex min-h-11 items-center justify-center gap-2 rounded-lg border border-border px-3 text-sm font-semibold disabled:cursor-not-allowed disabled:opacity-50"
              >
                <FolderOpen className="h-4 w-4" />
                Mở thư mục
              </button>
              <button
                type="button"
                onClick={() => inputPath && void processImage(inputPath)}
                disabled={!inputPath || isBusy}
                className="inline-flex min-h-11 items-center justify-center gap-2 rounded-lg border border-border px-3 text-sm font-semibold disabled:cursor-not-allowed disabled:opacity-50"
              >
                <RefreshCw className="h-4 w-4" />
                Quét lại
              </button>
            </div>
            {result?.isSaved && result.savedPdfPath && (
              <p className="mt-3 break-all text-xs text-muted-foreground">Đã lưu: {result.savedPdfPath}</p>
            )}
          </div>
        </section>
      </section>
    </main>
  );
}
