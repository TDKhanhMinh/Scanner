/**
 * Canonical contracts and type definitions for Attendance Scanner protocol v1.
 * Matches Python models in `attendance_scanner.contracts` and `attendance_scanner.events`.
 */

export const PROTOCOL_VERSION = 1;

export type ScanMode = "gray" | "bw" | "color";

export type FileClassification =
  | "new"
  | "modified"
  | "unchanged"
  | "rebuild"
  | "collision"
  | "unsupported";

export type FileProcessingStatus =
  | "pending"
  | "processing"
  | "success"
  | "warning"
  | "failed"
  | "skipped";

export const VALID_SCANNER_ERROR_CODES = new Set<string>([
  "INVALID_INPUT_ROOT",
  "OUTPUT_NOT_WRITABLE",
  "IMAGE_DECODE_FAILED",
  "PDF_WRITE_FAILED",
  "STATE_READ_FAILED",
  "STATE_WRITE_FAILED",
  "OUTPUT_COLLISION",
  "UNEXPECTED_ERROR",
]);

export type ScannerErrorCode =
  | "INVALID_INPUT_ROOT"
  | "OUTPUT_NOT_WRITABLE"
  | "IMAGE_DECODE_FAILED"
  | "PDF_WRITE_FAILED"
  | "STATE_READ_FAILED"
  | "STATE_WRITE_FAILED"
  | "OUTPUT_COLLISION"
  | "UNEXPECTED_ERROR";

export const VALID_SCANNER_WARNING_CODES = new Set<string>([
  "DOCUMENT_NOT_DETECTED",
  "IMAGE_DOWNSCALED",
  "WARP_FALLBACK",
]);

export type ScannerWarningCode =
  | "DOCUMENT_NOT_DETECTED"
  | "IMAGE_DOWNSCALED"
  | "WARP_FALLBACK";

export interface DiscoveredFile {
  employeeName: string;
  fileName: string;
  relativePath: string;
  absolutePath: string;
  size: number;
  mtimeNs: number;
  sha256?: string | null;
  classification: FileClassification;
  targetRelativePdf: string;
}

export interface ScanPlan {
  inputRoot: string;
  outputRoot: string;
  employees: number;
  totalImages: number;
  new: number;
  modified: number;
  unchanged: number;
  filesToProcess: number;
  collisions: string[];
  // Backward-compatible aliases
  totalEmployees?: number;
  newCount?: number;
  modifiedCount?: number;
  unchangedCount?: number;
}

export interface ScanBatchRequest {
  inputRoot: string;
  outputRoot?: string | null;
  mode: ScanMode;
  workers: number;
}

export interface FileResult {
  relativePath: string;
  employeeName: string;
  targetRelativePdf: string;
  status: FileProcessingStatus;
  documentDetected: boolean;
  warning?: string | null;
  errorCode?: ScannerErrorCode | null;
  errorMessage?: string | null;
  durationMs?: number | null;
  processedAt: string;
}

export interface BatchSummary {
  totalImages: number;
  success: number;
  failed: number;
  warning: number;
  skipped: number;
  durationMs?: number | null;
}

// ---------------------------------------------------------------------------
// JSONL Protocol Events
// ---------------------------------------------------------------------------

export interface BaseEvent {
  protocolVersion: number;
  timestamp: string;
  [extraKey: string]: unknown; // Forward compatibility for extension fields
}

export interface ScanPlanEvent extends BaseEvent {
  type: "scan_plan";
  inputRoot: string;
  outputRoot: string;
  employees: number;
  totalImages: number;
  new: number;
  modified: number;
  unchanged: number;
  filesToProcess: number;
  collisions: string[];
  // Backward-compatible aliases
  totalEmployees?: number;
  newCount?: number;
  modifiedCount?: number;
  unchangedCount?: number;
}

export interface FileStartedEvent extends BaseEvent {
  type: "file_started";
  relativePath: string;
  employeeName: string;
  index: number;
  total: number;
}

export interface FileCompletedEvent extends BaseEvent {
  type: "file_completed";
  relativePath: string;
  employeeName: string;
  outputRelativePath: string;
  documentDetected: boolean;
  warning?: string | null;
  durationMs: number;
}

export interface FileFailedEvent extends BaseEvent {
  type: "file_failed";
  relativePath: string;
  employeeName: string;
  errorCode: ScannerErrorCode;
  message: string;
}

export interface ScanCompletedEvent extends BaseEvent {
  type: "scan_completed";
  totalProcessed: number;
  success: number;
  failed: number;
  warning: number;
  skipped: number;
  durationMs: number;
}

export type ScannerEvent =
  | ScanPlanEvent
  | FileStartedEvent
  | FileCompletedEvent
  | FileFailedEvent
  | ScanCompletedEvent;

export type ScannerEventType = ScannerEvent["type"];

// Type guards
export function isScanPlanEvent(event: ScannerEvent): event is ScanPlanEvent {
  return event.type === "scan_plan";
}

export function isFileStartedEvent(event: ScannerEvent): event is FileStartedEvent {
  return event.type === "file_started";
}

export function isFileCompletedEvent(event: ScannerEvent): event is FileCompletedEvent {
  return event.type === "file_completed";
}

export function isFileFailedEvent(event: ScannerEvent): event is FileFailedEvent {
  return event.type === "file_failed";
}

export function isScanCompletedEvent(event: ScannerEvent): event is ScanCompletedEvent {
  return event.type === "scan_completed";
}
