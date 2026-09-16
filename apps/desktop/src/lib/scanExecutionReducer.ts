import type { FileResultItem } from "@/components/scanner/FileResultList";
import type { ScannerEvent } from "@/types/scanner";

export const MAX_RECENT_ACTIVITY = 100;

export type ScanExecutionPhase = "idle" | "running" | "completed" | "error";

export interface ScanExecutionState {
  phase: ScanExecutionPhase;
  totalToProcess: number;
  processed: number;
  currentFile: string;
  currentEmployee: string;
  success: number;
  warning: number;
  failed: number;
  skipped: number;
  fallbackCount: number;
  occlusionRiskCount: number;
  results: FileResultItem[];
  errorMessage: string;
  seenEventKeys: Set<string>;
}

export type ScanExecutionAction =
  | { type: "scan_started"; totalToProcess: number }
  | { type: "scanner_event"; event: ScannerEvent }
  | { type: "scan_error"; message: string }
  | { type: "reset" };

export const initialScanExecutionState: ScanExecutionState = {
  phase: "idle",
  totalToProcess: 0,
  processed: 0,
  currentFile: "",
  currentEmployee: "",
  success: 0,
  warning: 0,
  failed: 0,
  skipped: 0,
  fallbackCount: 0,
  occlusionRiskCount: 0,
  results: [],
  errorMessage: "",
  seenEventKeys: new Set<string>(),
};

function eventKey(event: ScannerEvent): string {
  switch (event.type) {
    case "scan_plan":
      return "scan_plan";
    case "file_started":
      return `started:${event.relativePath}:${event.index}`;
    case "file_completed":
    case "file_failed":
      return `terminal:${event.relativePath}`;
    case "scan_completed":
      return "scan_completed";
    case "quick_scan_completed":
      return `quick:${event.inputPath}:${event.timestamp}`;
    case "flat_scan_plan":
      return `flat_plan:${event.inputRoot}:${event.timestamp}`;
    case "flat_scan_progress":
      return `flat_progress:${event.relativePath}:${event.completedSources}:${event.timestamp}`;
    case "flat_file_completed":
      return `flat_file:${event.relativePath}:${event.timestamp}`;
    case "flat_scan_completed":
      return `flat_completed:${event.inputRoot}:${event.timestamp}`;
  }
}

function withSeenEvent(state: ScanExecutionState, key: string): ScanExecutionState | null {
  if (state.seenEventKeys.has(key)) {
    return null;
  }
  const seenEventKeys = new Set(state.seenEventKeys);
  seenEventKeys.add(key);
  return { ...state, seenEventKeys };
}

function sourceFileName(relativePath: string): string {
  return relativePath.split(/[\\/]/).pop() ?? relativePath;
}

function appendRecentResult(
  results: FileResultItem[],
  result: FileResultItem,
): FileResultItem[] {
  return [result, ...results].slice(0, MAX_RECENT_ACTIVITY);
}

function hasWarningCode(warning: string | null | undefined, code: string): boolean {
  return warning?.split(",").some((item) => item.trim() === code) ?? false;
}

export function scanExecutionReducer(
  state: ScanExecutionState,
  action: ScanExecutionAction,
): ScanExecutionState {
  switch (action.type) {
    case "reset":
      return {
        ...initialScanExecutionState,
        seenEventKeys: new Set<string>(),
      };
    case "scan_started":
      return {
        phase: "running",
        totalToProcess: action.totalToProcess,
        processed: 0,
        currentFile: "",
        currentEmployee: "",
        success: 0,
        warning: 0,
        failed: 0,
        skipped: 0,
        fallbackCount: 0,
        occlusionRiskCount: 0,
        results: [],
        errorMessage: "",
        seenEventKeys: new Set<string>(),
      };
    case "scan_error":
      return {
        ...state,
        phase: "error",
        currentFile: "",
        currentEmployee: "",
        errorMessage: action.message,
      };
    case "scanner_event": {
      const event = action.event;
      if (
        event.type === "quick_scan_completed" ||
        event.type === "flat_scan_plan" ||
        event.type === "flat_scan_progress" ||
        event.type === "flat_file_completed" ||
        event.type === "flat_scan_completed"
      ) {
        // FolderScanView owns the flat workflow state because its manifest,
        // artifact and source-progress fields differ from attendance results.
        // AppShell still routes these events centrally; this reducer is scoped
        // to the attendance workflow and must not consume foreign events.
        return state;
      }
      const nextState = withSeenEvent(state, eventKey(event));
      if (nextState === null) {
        return state;
      }

      switch (event.type) {
        case "scan_plan":
          return {
            ...nextState,
            totalToProcess: event.filesToProcess,
            skipped: event.unchanged,
          };
        case "file_started":
          return {
            ...nextState,
            phase: "running",
            currentFile: event.relativePath,
            currentEmployee: event.employeeName,
          };
        case "file_completed": {
          const isWarning = Boolean(event.warning);
          const result: FileResultItem = {
            id: `${event.relativePath}:${event.timestamp}`,
            relativePath: event.relativePath,
            employeeName: event.employeeName,
            sourceFile: sourceFileName(event.relativePath),
            targetPdf: event.outputRelativePath,
            status: isWarning ? "warning" : "success",
            documentDetected: event.documentDetected,
            message: event.warning ?? undefined,
            detectionReason: event.detectionReason ?? undefined,
            detectionReasonCodes: event.detectionReasonCodes ?? [],
            detectionPreview: event.detectionPreview ?? undefined,
            timestamp: event.timestamp,
          };
          return {
            ...nextState,
            processed: nextState.processed + 1,
            success: nextState.success + (isWarning ? 0 : 1),
            warning: nextState.warning + (isWarning ? 1 : 0),
            fallbackCount: nextState.fallbackCount + (event.documentDetected ? 0 : 1),
            occlusionRiskCount:
              nextState.occlusionRiskCount +
              (hasWarningCode(event.warning, "OCCLUSION_RISK") ? 1 : 0),
            results: appendRecentResult(nextState.results, result),
          };
        }
        case "file_failed": {
          const result: FileResultItem = {
            id: `${event.relativePath}:${event.timestamp}`,
            relativePath: event.relativePath,
            employeeName: event.employeeName,
            sourceFile: sourceFileName(event.relativePath),
            targetPdf: "",
            status: "failed",
            documentDetected: false,
            message: `${event.errorCode}: ${event.message}`,
            detectionReason: event.detectionReason ?? undefined,
            detectionReasonCodes: event.detectionReasonCodes ?? [],
            timestamp: event.timestamp,
          };
          return {
            ...nextState,
            processed: nextState.processed + 1,
            failed: nextState.failed + 1,
            results: appendRecentResult(nextState.results, result),
          };
        }
        case "scan_completed":
          return {
            ...nextState,
            phase: "completed",
            processed: Math.max(nextState.processed, event.totalProcessed),
            success: event.success,
            warning: event.warning,
            failed: event.failed,
            skipped: event.skipped,
            occlusionRiskCount: event.occlusionRiskCount ?? nextState.occlusionRiskCount,
            currentFile: "",
            currentEmployee: "",
          };
      }
    }
  }
}
