import type {
  DetectorMode,
  DetectionPreview,
  DocumentOrientation,
  FlatExportMode,
  ScanMode,
  ScannerErrorCode,
} from "@/types/scanner";

export type { DocumentOrientation, FlatExportMode, WorkflowMode } from "@/types/scanner";

export interface QuickScanRequest {
  inputPath: string;
  mode: ScanMode;
  detectorMode: DetectorMode;
  orientation: DocumentOrientation;
  debugDiagnostics: boolean;
}

export interface QuickScanResult {
  success: boolean;
  inputPath: string;
  tempPdfPath: string | null;
  savedPdfPath: string | null;
  isSaved: boolean;
  documentDetected: boolean;
  durationMs: number;
  detectionPreview: DetectionPreview | null;
  processedPreviewDataUrl: string | null;
  errorCode: ScannerErrorCode | null;
  message: string | null;
  warning: string | null;
  occlusionRisk?: boolean;
}

export interface FlatScanRequest {
  inputRoot: string;
  outputRoot: string;
  exportMode: FlatExportMode;
  mode: ScanMode;
  detectorMode: DetectorMode;
  orientation: DocumentOrientation;
  workers: number;
}

export interface FlatScanRunOutcome {
  exitCode: number;
}
