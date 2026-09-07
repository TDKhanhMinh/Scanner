import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { parseScannerEvent } from "@/lib/eventParser";
import type {
  ScanMode,
  ScanPlanEvent,
  ScannerEvent,
} from "@/types/scanner";

export const SCANNER_EVENT_CHANNEL = "scanner://event";
export const SCANNER_STDERR_CHANNEL = "scanner://stderr";

export interface ScannerDiagnostic {
  stream: "stderr";
  message: string;
}

export interface ScannerBridgeError {
  kind?: string;
  message?: string;
  code?: number;
}

export interface ScanRunOutcome {
  exitCode: number;
}

export interface ScannerRequest {
  inputRoot: string;
  outputRoot?: string | null;
  mode?: ScanMode | null;
  workers?: number | null;
}

export async function planScan(request: ScannerRequest): Promise<ScanPlanEvent> {
  return invoke<ScanPlanEvent>("plan_scan", {
    inputRoot: request.inputRoot,
    outputRoot: request.outputRoot ?? null,
    mode: request.mode ?? null,
  });
}

export async function startScan(request: ScannerRequest): Promise<ScanRunOutcome> {
  return invoke<ScanRunOutcome>("start_scan", {
    inputRoot: request.inputRoot,
    outputRoot: request.outputRoot ?? null,
    mode: request.mode ?? null,
    workers: request.workers ?? null,
  });
}

export function listenScannerEvents(
  onEvent: (event: ScannerEvent) => void,
): Promise<UnlistenFn> {
  return listen<unknown>(SCANNER_EVENT_CHANNEL, (event) => {
    const parsed = parseScannerEvent(JSON.stringify(event.payload));
    if (parsed !== null) {
      onEvent(parsed);
    }
  });
}

export function listenScannerDiagnostics(
  onDiagnostic: (diagnostic: ScannerDiagnostic) => void,
): Promise<UnlistenFn> {
  return listen<ScannerDiagnostic>(SCANNER_STDERR_CHANNEL, (event) => {
    onDiagnostic(event.payload);
  });
}

export function scannerErrorMessage(error: unknown): string {
  if (typeof error === "string") {
    return error;
  }
  if (typeof error === "object" && error !== null) {
    const bridgeError = error as ScannerBridgeError;
    if (bridgeError.message) {
      return bridgeError.message;
    }
  }
  return "Không thể kết nối tới scanner sidecar.";
}
