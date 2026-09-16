import { AppHeader } from "@/components/scanner/AppHeader";
import { BatchOptionsCard } from "@/components/scanner/BatchOptionsCard";
import { BatchProgressCard } from "@/components/scanner/BatchProgressCard";
import { DetectorModeSelector } from "@/components/scanner/DetectorModeSelector";
import { DetectorOptionsCard } from "@/components/scanner/DetectorOptionsCard";
import { DetectionPreviewPanel } from "@/components/scanner/DetectionPreviewPanel";
import { FileResultList, type FileResultItem } from "@/components/scanner/FileResultList";
import { FolderSelectorCard } from "@/components/scanner/FolderSelectorCard";
import { PageOrderReviewPanel } from "@/components/scanner/PageOrderReviewPanel";
import { ScanModeSelector, type ScanFilterMode } from "@/components/scanner/ScanModeSelector";
import { ScanPlanSummaryCard, type ScanPlanStats } from "@/components/scanner/ScanPlanSummaryCard";
import { WorkerSettingCard } from "@/components/scanner/WorkerSettingCard";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  initialScanExecutionState,
  scanExecutionReducer,
} from "@/lib/scanExecutionReducer";
import {
  planScan,
  scannerDiagnosticMessage,
  scannerErrorMessage,
  startScan,
  type ScannerDiagnostic,
} from "@/lib/scannerBridge";
import { useExecutionCoordinator } from "@/components/scanner/executionCoordinator";
import {
  loadUserPreferences,
  saveUserPreferences,
  createPlanFingerprint,
} from "@/lib/userPreferences";
import type {
  BatchPeriod,
  ProductDetectorMode,
  ExportMode,
  ReviewGroup,
  ScanPlanEvent,
  ScannerEvent,
} from "@/types/scanner";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { openPath, revealItemInDir } from "@tauri-apps/plugin-opener";
import { CheckCircle2, FileText, FolderOpen, Info, Scan, Sparkles } from "lucide-react";
import { useEffect, useReducer, useRef, useState } from "react";

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
    documentGroups: plan.documentGroups ?? 0,
    expectedArtifacts: plan.expectedArtifacts ?? 0,
    completeGroups: plan.completeGroups ?? 0,
    incompleteGroups: plan.incompleteGroups ?? 0,
    ambiguousGroups: plan.ambiguousGroups ?? 0,
    pagesNeedingReview: plan.pagesNeedingReview ?? 0,
    needsReprocess: plan.needsReprocess ?? 0,
  };
}

function defaultBatchPeriod(): BatchPeriod {
  const now = new Date();
  return { year: now.getFullYear(), month: now.getMonth() + 1 };
}

interface ScannerSettings {
  mode: ScanFilterMode;
  detectorMode: ProductDetectorMode;
  debugDiagnostics: boolean;
  reprocess: boolean;
  workers: number | null;
  period: BatchPeriod;
  exportMode: ExportMode;
}

export function AttendanceBatchView() {
  const [inputPath, setInputPath] = useState<string>("");
  const [outputPath, setOutputPath] = useState<string>("");
  const [settings, setSettings] = useState<ScannerSettings>(() => {
    const preferences = loadUserPreferences();
    return {
      mode: preferences.mode,
      detectorMode: preferences.detectorMode,
      debugDiagnostics: preferences.debugDiagnostics,
      reprocess: false,
      workers: null,
      period: defaultBatchPeriod(),
      exportMode: preferences.exportMode,
    };
  });
  const [activeTab, setActiveTab] = useState<string>("config");
  const [reviewGroups, setReviewGroups] = useState<ReviewGroup[]>([]);
  const [resolvedReviewGroups, setResolvedReviewGroups] = useState<
    Record<string, "resolved" | "skipped">
  >({});
  const [manualOrderOverrides, setManualOrderOverrides] = useState<
    Record<string, string[]>
  >({});
  const [skippedReviewGroups, setSkippedReviewGroups] = useState<string[]>([]);
  const [selectedPreview, setSelectedPreview] = useState<FileResultItem | null>(null);

  // Synchronous refs for review overrides and latest settings
  const resolvedReviewGroupsRef = useRef<Record<string, "resolved" | "skipped">>({});
  const manualOrderOverridesRef = useRef<Record<string, string[]>>({});
  const skippedReviewGroupsRef = useRef<string[]>([]);
  const latestSettingsRef = useRef<ScannerSettings>(settings);
  latestSettingsRef.current = settings;

  // Plan state and fingerprint lifecycle
  const [currentPlan, setCurrentPlan] = useState<ScanPlanEvent | null>(null);
  const latestPlanRef = useRef<ScanPlanEvent | null>(null);
  const [activePlanFingerprint, setActivePlanFingerprint] = useState<string | null>(null);
  const scanStartInFlight = useRef<boolean>(false);

  // Incremental scan plan state loaded from the Tauri scanner bridge.
  const [planStats, setPlanStats] = useState<ScanPlanStats>({
    totalEmployees: 0,
    totalImages: 0,
    newFiles: 0,
    modifiedFiles: 0,
    unchangedFiles: 0,
    rebuildFiles: 0,
    unsupportedFiles: 0,
    documentGroups: 0,
    expectedArtifacts: 0,
    completeGroups: 0,
    incompleteGroups: 0,
    ambiguousGroups: 0,
    pagesNeedingReview: 0,
    needsReprocess: 0,
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
  const planDebounceTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const scanMode = settings.mode;
  const isScanning = execution.phase === "running";
  const {
    isAnyExecuting,
    beginExecution,
    endExecution,
    registerDiagnosticHandler,
    registerEventHandler,
  } = useExecutionCoordinator();
  const { currentFile, currentEmployee, processed: processedCount, results } = execution;
  const successCount = execution.success;
  const warningCount = execution.warning;
  const failedCount = execution.failed;

  const currentFingerprint = inputPath.trim()
    ? createPlanFingerprint(inputPath, outputPath || `${inputPath.trim()}_pdf`, settings)
    : null;
  const currentFingerprintRef = useRef<string | null>(currentFingerprint);
  currentFingerprintRef.current = currentFingerprint;

  const isPlanStale = Boolean(
    isPlanReady &&
      currentPlan &&
      (!activePlanFingerprint || currentFingerprint !== activePlanFingerprint),
  );

  useEffect(() => {
    saveUserPreferences({
      mode: settings.mode,
      detectorMode: settings.detectorMode,
      exportMode: settings.exportMode,
      debugDiagnostics: settings.debugDiagnostics,
    });
  }, [settings.mode, settings.detectorMode, settings.exportMode, settings.debugDiagnostics]);

  useEffect(() => {
    return () => {
      if (planDebounceTimer.current !== undefined) {
        clearTimeout(planDebounceTimer.current);
      }
      planRequestId.current += 1;
    };
  }, []);

  const requestPlan = async (
    path: string,
    nextOutputPath: string,
    requestSettings: ScannerSettings = settings,
  ): Promise<ScanPlanEvent | null> => {
    const requestId = ++planRequestId.current;
    const requestFingerprint = createPlanFingerprint(path, nextOutputPath, requestSettings);
    setIsPlanning(true);
    setErrorMessage("");
    setHasPlanError(false);

    try {
      const plan = await planScan({
        inputRoot: path,
        outputRoot: nextOutputPath,
        mode: requestSettings.mode,
        detectorMode: requestSettings.detectorMode,
        debugDiagnostics: requestSettings.debugDiagnostics,
        reprocess: requestSettings.reprocess,
        period: requestSettings.period,
        exportMode: requestSettings.exportMode,
      });

      // Contract 1: Invalidate if requestId or fingerprint changed
      if (
        requestId !== planRequestId.current ||
        requestFingerprint !== currentFingerprintRef.current
      ) {
        return null;
      }

      setCurrentPlan(plan);
      latestPlanRef.current = plan;
      setActivePlanFingerprint(requestFingerprint);
      setPlanStats(toPlanStats(plan));
      setInputPath(plan.inputRoot);
      setOutputPath(plan.outputRoot);
      setReviewGroups(plan.reviewGroups ?? []);
      setResolvedReviewGroups({});
      resolvedReviewGroupsRef.current = {};
      setManualOrderOverrides({});
      manualOrderOverridesRef.current = {};
      setSkippedReviewGroups([]);
      skippedReviewGroupsRef.current = [];
      setIsPlanReady(true);
      return plan;
    } catch (error: unknown) {
      if (
        requestId === planRequestId.current &&
        requestFingerprint === currentFingerprintRef.current
      ) {
        setActivePlanFingerprint(null);
        setCurrentPlan(null);
        latestPlanRef.current = null;
        setPlanStats({
          totalEmployees: 0,
          totalImages: 0,
          newFiles: 0,
          modifiedFiles: 0,
          unchangedFiles: 0,
          rebuildFiles: 0,
          unsupportedFiles: 0,
          needsReprocess: 0,
        });
        setErrorMessage(scannerErrorMessage(error));
        setReviewGroups([]);
        setResolvedReviewGroups({});
        resolvedReviewGroupsRef.current = {};
        setManualOrderOverrides({});
        manualOrderOverridesRef.current = {};
        setSkippedReviewGroups([]);
        skippedReviewGroupsRef.current = [];
        setHasPlanError(true);
        setIsPlanReady(false);
      }
      return null;
    } finally {
      if (
        requestId === planRequestId.current &&
        requestFingerprint === currentFingerprintRef.current
      ) {
        setIsPlanning(false);
      }
    }
  };

  const schedulePlan = (
    path: string,
    nextOutputPath: string,
    requestSettings: ScannerSettings = latestSettingsRef.current,
  ) => {
    if (planDebounceTimer.current !== undefined) {
      clearTimeout(planDebounceTimer.current);
    }
    planDebounceTimer.current = setTimeout(() => {
      planDebounceTimer.current = undefined;
      if (!beginExecution("attendance_batch")) return;
      void requestPlan(path, nextOutputPath, requestSettings).finally(() => {
        endExecution("attendance_batch");
      });
    }, 250);
  };

  // Cập nhật thư mục nhập từ người dùng và tính toán kế hoạch quét
  const handleInputChange = (path: string) => {
    setInputPath(path);
    dispatchExecution({ type: "reset" });
    setActivePlanFingerprint(null);
    setCurrentPlan(null);
    latestPlanRef.current = null;
    setReviewGroups([]);
    setResolvedReviewGroups({});
    resolvedReviewGroupsRef.current = {};
    setManualOrderOverrides({});
    manualOrderOverridesRef.current = {};
    setSkippedReviewGroups([]);
    skippedReviewGroupsRef.current = [];
    setSelectedPreview(null);
    planRequestId.current += 1;
    setIsPlanning(false);
    setIsPlanReady(false);
    const trimmed = path.trim();
    const nextOutputPath = trimmed ? `${trimmed}_pdf` : "";
    setOutputPath(nextOutputPath);

    if (trimmed) {
      schedulePlan(trimmed, nextOutputPath, latestSettingsRef.current);
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
        needsReprocess: 0,
        documentGroups: 0,
        expectedArtifacts: 0,
        completeGroups: 0,
        incompleteGroups: 0,
        ambiguousGroups: 0,
        pagesNeedingReview: 0,
      });
    }
  };

  const handleSettingChange = (updater: (prev: ScannerSettings) => ScannerSettings) => {
    planRequestId.current += 1;
    // Immediately release loading state and clear stale review groups.
    setIsPlanning(false);
    setReviewGroups([]);
    setResolvedReviewGroups({});
    resolvedReviewGroupsRef.current = {};
    setManualOrderOverrides({});
    manualOrderOverridesRef.current = {};
    setSkippedReviewGroups([]);
    skippedReviewGroupsRef.current = [];

    const nextSettings = updater(latestSettingsRef.current);
    latestSettingsRef.current = nextSettings;
    setSettings(nextSettings);

    // If the user typed a folder path within the 250ms debounce window,
    // reschedule using the latest settings without putting side effects inside
    // the functional state updater.
    if (planDebounceTimer.current !== undefined) {
      clearTimeout(planDebounceTimer.current);
      planDebounceTimer.current = undefined;
      if (inputPath.trim()) {
        schedulePlan(inputPath.trim(), outputPath || `${inputPath.trim()}_pdf`, nextSettings);
      }
    }
  };

  const handlePeriodChange = (period: BatchPeriod) => {
    handleSettingChange((prev) => ({ ...prev, period }));
  };

  const handleExportModeChange = (exportMode: ExportMode) => {
    handleSettingChange((prev) => ({ ...prev, exportMode }));
  };

  const handleScanModeChange = (mode: ScanFilterMode) => {
    handleSettingChange((prev) => ({ ...prev, mode }));
  };

  const handleDetectorModeChange = (detectorMode: ProductDetectorMode) => {
    handleSettingChange((prev) => ({ ...prev, detectorMode }));
  };

  const handleReprocessChange = (reprocess: boolean) => {
    handleSettingChange((prev) => ({ ...prev, reprocess }));
  };

  const handleDebugDiagnosticsChange = (debugDiagnostics: boolean) => {
    // Diagnostics changes affect execution logging, not plan membership or
    // grouped review ordering. Keep the current plan and review state intact.
    const nextSettings = { ...latestSettingsRef.current, debugDiagnostics };
    latestSettingsRef.current = nextSettings;
    setSettings(nextSettings);
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
    setActivePlanFingerprint(null);
    setCurrentPlan(null);
    latestPlanRef.current = null;
    setReviewGroups([]);
    setResolvedReviewGroups({});
    setManualOrderOverrides({});
    setSkippedReviewGroups([]);
    planRequestId.current += 1;
    setIsPlanning(false);
    setIsPlanReady(false);
    setHasPlanError(false);
    setOutputPath(nextOutputPath);
    if (input) {
      schedulePlan(input, nextOutputPath, settings);
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
    if (!inputPath.trim()) return;
    if (planDebounceTimer.current !== undefined) {
      clearTimeout(planDebounceTimer.current);
      planDebounceTimer.current = undefined;
    }
    if (!beginExecution("attendance_batch")) return;
    void requestPlan(inputPath.trim(), outputPath || `${inputPath.trim()}_pdf`, settings).finally(
      () => {
        endExecution("attendance_batch");
      },
    );
  };

  const handleScannerEvent = (event: ScannerEvent) => {
    dispatchExecution({ type: "scanner_event", event });
    if (event.type === "scan_plan") {
      setPlanStats(toPlanStats(event));
      setReviewGroups(event.reviewGroups ?? []);
    }
  };

  const scannerEventHandlerRef = useRef<(event: ScannerEvent) => void>(() => undefined);
  scannerEventHandlerRef.current = handleScannerEvent;
  const diagnosticHandlerRef = useRef<(diagnostic: ScannerDiagnostic) => void>(() => undefined);
  diagnosticHandlerRef.current = (diagnostic) => {
    setErrorMessage(scannerDiagnosticMessage(diagnostic));
  };

  useEffect(() => {
    const unregisterEvents = registerEventHandler("attendance_batch", (event) => {
      scannerEventHandlerRef.current(event);
    });
    const unregisterDiagnostics = registerDiagnosticHandler("attendance_batch", (diagnostic) => {
      diagnosticHandlerRef.current(diagnostic);
    });
    return () => {
      unregisterEvents();
      unregisterDiagnostics();
    };
  }, [registerDiagnosticHandler, registerEventHandler]);

  const handleResolveReviewGroup = (groupId: string, orderedPaths: string[]) => {
    setManualOrderOverrides((current) => {
      const updated = { ...current, [groupId]: orderedPaths };
      manualOrderOverridesRef.current = updated;
      return updated;
    });
    setResolvedReviewGroups((current) => {
      const updated: Record<string, "resolved" | "skipped"> = {
        ...current,
        [groupId]: "resolved",
      };
      resolvedReviewGroupsRef.current = updated;
      return updated;
    });
  };

  const handleSkipReviewGroup = (groupId: string) => {
    setResolvedReviewGroups((current) => {
      const updated: Record<string, "resolved" | "skipped"> = {
        ...current,
        [groupId]: "skipped",
      };
      resolvedReviewGroupsRef.current = updated;
      return updated;
    });
    setSkippedReviewGroups((current) => {
      const updated = current.includes(groupId) ? current : [...current, groupId];
      skippedReviewGroupsRef.current = updated;
      return updated;
    });
  };

  const handlePreviewSource = (relativePath: string) => {
    const root = inputPath.replace(/[\\/]+$/, "");
    const sourcePath = `${root}\\${relativePath.replace(/\//g, "\\")}`;
    void openPath(sourcePath).catch((error: unknown) => {
      setErrorMessage(scannerErrorMessage(error));
    });
  };

  const handleSelectResultPreview = (item: FileResultItem) => {
    setSelectedPreview(item);
    setActiveTab("results");
  };

  const handleStartScan = () => {
    if (
      !inputPath.trim() ||
      isScanning ||
      isPlanning ||
      isAnyExecuting ||
      scanStartInFlight.current
    ) {
      return;
    }

    if (!beginExecution("attendance_batch")) return;
    scanStartInFlight.current = true;
    setErrorMessage("");

    void (async () => {
      try {
        let effectivePlan = latestPlanRef.current ?? currentPlan;
        const targetInput = inputPath.trim();
        const targetOutput = outputPath || `${targetInput}_pdf`;
        const wasPlanStale = Boolean(!effectivePlan || isPlanStale || !isPlanReady);

        // If plan is stale, missing, or not ready, refresh plan first
        if (wasPlanStale) {
          if (planDebounceTimer.current !== undefined) {
            clearTimeout(planDebounceTimer.current);
            planDebounceTimer.current = undefined;
          }
          const freshPlan = await requestPlan(targetInput, targetOutput, settings);
          if (!freshPlan) {
            // Plan request failed or was cancelled due to new input/settings.
            // Do NOT reset progress or start scan!
            return;
          }
          effectivePlan = freshPlan;
        }

        if (!effectivePlan) {
          return;
        }

        // Grouped review gate: must resolve or skip all review groups
        if (settings.exportMode === "GROUPED") {
          const groups = effectivePlan.reviewGroups ?? [];
          if (groups.length > 0) {
            // Finding P1 #2 Fix:
            // If plan was just refreshed from stale, ANY review groups in the new plan
            // REQUIRE user review! Old review approvals from previous settings cannot apply!
            if (wasPlanStale) {
              return;
            }
            // If plan was already up-to-date, verify every group is resolved/skipped in ref
            const currentResolved = resolvedReviewGroupsRef.current;
            const hasUnresolvedReview = groups.some((group) => {
              const groupKey = `${group.key.employeeRelativeDir}:${group.key.year}-${String(group.key.month).padStart(2, "0")}`;
              return !currentResolved[groupKey];
            });
            if (hasUnresolvedReview) {
              // Groups need user review before scanning. Do NOT reset progress!
              return;
            }
          }
        }

        // Fresh plan is valid and approved. Now and only now dispatch scan_started
        const stats = toPlanStats(effectivePlan);
        const totalToProcess =
          stats.newFiles + stats.modifiedFiles + (stats.rebuildFiles ?? 0);
        dispatchExecution({
          type: "scan_started",
          totalToProcess,
        });

        try {
          await startScan({
              inputRoot: effectivePlan.inputRoot,
              outputRoot: effectivePlan.outputRoot,
              mode: scanMode,
              detectorMode: settings.detectorMode,
              debugDiagnostics: settings.debugDiagnostics,
              reprocess: settings.reprocess,
              workers: settings.workers,
              period: settings.period,
              exportMode: settings.exportMode,
              manualOrder: manualOrderOverridesRef.current ?? manualOrderOverrides,
              skipGroups: skippedReviewGroupsRef.current ?? skippedReviewGroups,
            });
        } catch (error: unknown) {
          const message = scannerErrorMessage(error);
          dispatchExecution({ type: "scan_error", message });
          setErrorMessage(message);
        }
      } finally {
        scanStartInFlight.current = false;
        endExecution("attendance_batch");
      }
    })();
  };

  const handleOpenOutputFolder = () => {
    const target = outputPath || `${inputPath}_pdf`;
    if (!target) return;
    invoke("open_output_folder", { path: target }).catch(() => {
      openPath(target).catch(() => {
        revealItemInDir(target).catch((error: unknown) => {
          setErrorMessage(scannerErrorMessage(error));
        });
      });
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

        {/* Completion Success Notification */}
        {execution.phase === "completed" && (
          <div
            role="status"
            className="rounded-2xl border border-emerald-500/30 bg-emerald-500/10 p-4 sm:p-5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 backdrop-blur-sm shadow-sm"
          >
            <div className="flex items-start gap-3.5">
              <div className="p-2 rounded-xl bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 shrink-0 mt-0.5">
                <CheckCircle2 className="w-5 h-5" />
              </div>
              <div className="space-y-1">
                <div className="flex flex-wrap items-center gap-2">
                  <h3 className="text-sm font-semibold text-emerald-700 dark:text-emerald-400">
                    {isPlanStale
                      ? "Kết quả của lượt quét trước (cấu hình hiện tại đã thay đổi)"
                      : "Quá trình quét hoàn tất thành công!"}
                  </h3>
                  <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-700 dark:text-emerald-300 font-medium">
                    {execution.processed} file đã xử lý
                  </span>
                  {isPlanStale && (
                    <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-800 dark:text-amber-300 border border-amber-500/30">
                      Lượt quét trước
                    </span>
                  )}
                </div>
                <p className="text-xs text-foreground/80 leading-relaxed">
                  Đã tạo file PDF an toàn:{" "}
                  <strong className="text-emerald-700 dark:text-emerald-400 font-medium">
                    {execution.success} thành công
                  </strong>
                  {execution.warning > 0 && (
                    <span className="text-amber-600 dark:text-amber-400 font-medium">
                      {" • "}{execution.warning} cảnh báo
                    </span>
                  )}
                  {execution.failed > 0 && (
                    <span className="text-destructive font-medium">
                      {" • "}{execution.failed} lỗi
                    </span>
                  )}
                  {execution.skipped > 0 && (
                    <span className="text-muted-foreground">
                      {" • "}{execution.skipped} bỏ qua
                    </span>
                  )}
                  {execution.fallbackCount > 0 && (
                    <span className="text-orange-600 dark:text-orange-400 font-medium">
                      {" • "}{execution.fallbackCount} fallback
                    </span>
                  )}
                  . Tất cả file PDF đã được ghi vào thư mục xuất.
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2.5 w-full sm:w-auto shrink-0 self-end sm:self-center">
              <button
                type="button"
                onClick={() => setActiveTab("results")}
                className="flex-1 sm:flex-initial inline-flex items-center justify-center px-3.5 py-2 rounded-xl text-xs font-medium border border-border bg-background hover:bg-muted text-foreground transition-colors shadow-sm"
              >
                <FileText className="w-3.5 h-3.5 mr-1.5 text-primary" />
                Xem kết quả ({results.length})
              </button>
              <button
                type="button"
                onClick={handleOpenOutputFolder}
                className="flex-1 sm:flex-initial inline-flex items-center justify-center px-3.5 py-2 rounded-xl text-xs font-medium bg-emerald-600 hover:bg-emerald-700 text-white shadow-sm transition-colors"
              >
                <FolderOpen className="w-3.5 h-3.5 mr-1.5" />
                Mở thư mục PDF
              </button>
            </div>
          </div>
        )}

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
                   disabled={isScanning || isAnyExecuting}
                />

                <BatchOptionsCard
                  period={settings.period}
                  exportMode={settings.exportMode}
                  onPeriodChange={handlePeriodChange}
                  onExportModeChange={handleExportModeChange}
                  disabled={isScanning}
                />

                {settings.exportMode === "GROUPED" && !isPlanStale && reviewGroups.length > 0 && (
                  <PageOrderReviewPanel
                    groups={reviewGroups}
                    resolvedGroups={resolvedReviewGroups}
                    onResolve={handleResolveReviewGroup}
                    onSkip={handleSkipReviewGroup}
                    onPreview={handlePreviewSource}
                  />
                )}

                <ScanModeSelector
                  mode={scanMode}
                  onSelectMode={handleScanModeChange}
                  disabled={isScanning}
                />

                <DetectorModeSelector
                  mode={settings.detectorMode}
                  onSelectMode={handleDetectorModeChange}
                  disabled={isScanning}
                />

                <DetectorOptionsCard
                  debugDiagnostics={settings.debugDiagnostics}
                  reprocess={settings.reprocess}
                  needsReprocess={planStats.needsReprocess ?? 0}
                  onDebugDiagnosticsChange={handleDebugDiagnosticsChange}
                  onReprocessChange={handleReprocessChange}
                  disabled={isScanning}
                />

                <WorkerSettingCard
                  workers={settings.workers}
                  onWorkersChange={(workers) =>
                    setSettings((previous) => ({ ...previous, workers }))
                  }
                  disabled={isScanning}
                />

                
              </div>

              {/* Cột phải (1 cột trên lg/desktop): Thống kê Scan Plan & CTA */}
              <div className="lg:col-span-1 min-w-0">
                <ScanPlanSummaryCard
                  stats={planStats}
                  isPlanning={isPlanning}
                  isScanning={isScanning}
                  isExecutionBusy={isAnyExecuting}
                  isPlanStale={isPlanStale}
                  period={settings.period}
                  exportMode={settings.exportMode}
                  canScan={
                    Boolean(inputPath.trim()) &&
                    isPlanReady &&
                    !hasPlanError &&
                    !isAnyExecuting &&
                    (isPlanStale ||
                      settings.exportMode !== "GROUPED" ||
                      (reviewGroups.length > 0 &&
                        reviewGroups.every((group) => {
                          const key = `${group.key.employeeRelativeDir}:${group.key.year}-${String(group.key.month).padStart(2, "0")}`;
                          return Boolean(resolvedReviewGroups[key]);
                        })) ||
                      reviewGroups.length === 0)
                  }
                  onRefreshPlan={handleRefreshPlan}
                  onStartScan={handleStartScan}
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
                    occlusionRiskCount={execution.occlusionRiskCount}
                    failedCount={failedCount}
                    skippedCount={execution.skipped}
                    isScanning={isScanning}
                    isComplete={execution.phase === "completed"}
                    outputReady={
                      Boolean(outputPath && isPlanReady && !isPlanStale && !isPlanning && !hasPlanError)
                    }
                    onOpenOutputFolder={handleOpenOutputFolder}
                  />
                )}
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

            {isPlanStale && results.length > 0 && (
              <p
                role="status"
                className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-3 text-xs font-medium text-amber-800 dark:text-amber-300"
              >
                Kết quả của lượt quét trước. Các thay đổi thiết lập sẽ được áp dụng khi bấm Quét lại.
              </p>
            )}

            {selectedPreview && (
              <DetectionPreviewPanel
                item={selectedPreview}
                onClose={() => setSelectedPreview(null)}
              />
            )}
            <FileResultList
              items={results}
              onPreview={handlePreviewSource}
              onPreviewItem={handleSelectResultPreview}
            />
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

export default AttendanceBatchView;

// --- Hybrid Responsive Summary ---
// mobile  (default / sm):  layout 1 cột liền mạch, tabs full width, các card xếp chồng tự nhiên
// tablet  (md / lg):       layout 2 cột linh hoạt (65% cấu hình bên trái / 35% tóm tắt bên phải)
// desktop (xl / 2xl):      tối đa max-w-7xl căn giữa, bảng dữ liệu kết quả chi tiết, animation mượt mà
// Interaction:             touch target >= 44px trên tất cả các nút, focus ring rõ ràng, hover states đầy đủ
