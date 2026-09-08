import type { FileResultItem } from "@/components/scanner/FileResultList";
import type { ScannerEvent } from "@/types/scanner";

export const MAX_RECENT_ACTIVITY = 100;

export type ScanExecutionPhase = "idle" | "running" | "completed" | "error";

export interface ScanExecutionState {
  phase: ScanExecutionPhase;
  totalToProcess: number;
  processed: number;
  currentFile: string;
  success: number;
  warning: number;
  failed: number;
  skipped: number;
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
  success: 0,
  warning: 0,
  failed: 0,
  skipped: 0,
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
        success: 0,
        warning: 0,
        failed: 0,
        skipped: 0,
        results: [],
        errorMessage: "",
        seenEventKeys: new Set<string>(),
      };
    case "scan_error":
      return {
        ...state,
        phase: "error",
        currentFile: "",
        errorMessage: action.message,
      };
    case "scanner_event": {
      const event = action.event;
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
          };
        case "file_completed": {
          const isWarning = Boolean(event.warning);
          const result: FileResultItem = {
            id: `${event.relativePath}:${event.timestamp}`,
            employeeName: event.employeeName,
            sourceFile: sourceFileName(event.relativePath),
            targetPdf: event.outputRelativePath,
            status: isWarning ? "warning" : "success",
            documentDetected: event.documentDetected,
            message: event.warning ?? undefined,
            timestamp: event.timestamp,
          };
          return {
            ...nextState,
            processed: nextState.processed + 1,
            success: nextState.success + (isWarning ? 0 : 1),
            warning: nextState.warning + (isWarning ? 1 : 0),
            results: appendRecentResult(nextState.results, result),
          };
        }
        case "file_failed": {
          const result: FileResultItem = {
            id: `${event.relativePath}:${event.timestamp}`,
            employeeName: event.employeeName,
            sourceFile: sourceFileName(event.relativePath),
            targetPdf: "",
            status: "failed",
            documentDetected: false,
            message: `${event.errorCode}: ${event.message}`,
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
            currentFile: "",
          };
      }
    }
  }
}
