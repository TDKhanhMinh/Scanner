import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { parseScannerEvent } from "@/lib/eventParser";
import type {
  BatchPeriod,
  ExportMode,
  ScanMode,
  ScanPlanEvent,
  ScannerEvent,
} from "@/types/scanner";

export const SCANNER_EVENT_CHANNEL = "scanner://event";
export const SCANNER_STDERR_CHANNEL = "scanner://stderr";

export interface ScannerDiagnostic {
  stream: "stderr";
  message: string;
  errorCode?: string;
}

export interface ScannerBridgeError {
  kind?: string;
  message?: string;
  errorCode?: string;
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
  period?: BatchPeriod | null;
  exportMode?: ExportMode | null;
  manualOrder?: Record<string, string[]> | null;
  skipGroups?: string[] | null;
}

export async function planScan(request: ScannerRequest): Promise<ScanPlanEvent> {
  return invoke<ScanPlanEvent>("plan_scan", {
    inputRoot: request.inputRoot,
    outputRoot: request.outputRoot ?? null,
    mode: request.mode ?? null,
    year: request.period?.year ?? null,
    month: request.period?.month ?? null,
    exportMode: request.exportMode ?? null,
  });
}

export async function startScan(request: ScannerRequest): Promise<ScanRunOutcome> {
  return invoke<ScanRunOutcome>("start_scan", {
    inputRoot: request.inputRoot,
    outputRoot: request.outputRoot ?? null,
    mode: request.mode ?? null,
    workers: request.workers ?? null,
    year: request.period?.year ?? null,
    month: request.period?.month ?? null,
    exportMode: request.exportMode ?? null,
    manualOrder: request.manualOrder ?? null,
    skipGroups: request.skipGroups ?? null,
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

const USER_MESSAGES: Record<string, string> = {
  INVALID_REQUEST: "Yêu cầu quét chưa hợp lệ. Hãy kiểm tra lại các tùy chọn và đường dẫn.",
  INVALID_INPUT_ROOT:
    "Không tìm thấy hoặc không thể truy cập thư mục ảnh gốc. Hãy kiểm tra đường dẫn và quyền truy cập.",
  OUTPUT_NOT_WRITABLE:
    "Không thể ghi vào thư mục xuất PDF. Hãy chọn thư mục khác hoặc kiểm tra quyền truy cập.",
  IMAGE_DECODE_FAILED:
    "Không thể đọc ảnh này. Hãy kiểm tra file có bị hỏng và thuộc định dạng được hỗ trợ.",
  PDF_WRITE_FAILED:
    "Không thể tạo file PDF. Hãy kiểm tra dung lượng và quyền ghi của thư mục xuất.",
  STATE_READ_FAILED:
    "Không thể đọc trạng thái quét trước đó. Hãy thử lại hoặc chọn lại thư mục.",
  STATE_WRITE_FAILED:
    "Không thể lưu trạng thái quét. Hãy kiểm tra quyền ghi của thư mục ứng dụng.",
  OUTPUT_COLLISION:
    "Tên file PDF bị trùng. Hãy đổi tên ảnh hoặc chọn thư mục xuất khác.",
  GROUP_EXPORT_BLOCKED:
    "Không thể tạo grouped PDF vì một source page trong group bị lỗi. Hãy sửa source và quét lại group.",
  SCANNER_ALREADY_RUNNING: "Một đợt quét khác đang chạy. Hãy chờ đợt quét hiện tại hoàn tất.",
  SIDECAR_LAUNCH_FAILED:
    "Không thể khởi động scanner sidecar. Hãy kiểm tra bản cài đặt và thử lại.",
  SIDECAR_STREAM_FAILED: "Kết nối với scanner sidecar bị gián đoạn. Hãy thử lại.",
  INVALID_SCANNER_EVENT: "Scanner sidecar trả về dữ liệu không hợp lệ. Hãy thử lại.",
  MISSING_SCAN_PLAN: "Scanner sidecar chưa tạo được kế hoạch quét. Hãy kiểm tra thư mục đầu vào.",
  SIDECAR_EXITED: "Scanner sidecar đã dừng bất thường. Hãy thử lại; nếu lỗi lặp lại, gửi mã cho hỗ trợ.",
  SCANNER_INTERNAL_ERROR: "Scanner gặp lỗi nội bộ. Hãy thử lại; nếu lỗi lặp lại, gửi mã cho hỗ trợ.",
  UNEXPECTED_ERROR:
    "Đã xảy ra lỗi không xác định khi quét. Vui lòng thử lại; nếu lỗi lặp lại, gửi mã cho bộ phận hỗ trợ.",
};

const BRIDGE_ERROR_CODES: Record<string, string> = {
  alreadyRunning: "SCANNER_ALREADY_RUNNING",
  invalidRequest: "INVALID_REQUEST",
  launchFailed: "SIDECAR_LAUNCH_FAILED",
  streamFailed: "SIDECAR_STREAM_FAILED",
  invalidEvent: "INVALID_SCANNER_EVENT",
  missingPlan: "MISSING_SCAN_PLAN",
  sidecarExited: "SIDECAR_EXITED",
  internal: "SCANNER_INTERNAL_ERROR",
};

function hasOwnMessage(code: string): boolean {
  return Object.prototype.hasOwnProperty.call(USER_MESSAGES, code);
}

function bridgeErrorCode(kind: string): string | undefined {
  return Object.prototype.hasOwnProperty.call(BRIDGE_ERROR_CODES, kind)
    ? BRIDGE_ERROR_CODES[kind]
    : undefined;
}

function formatScannerError(code: string): string {
  const normalizedCode = hasOwnMessage(code) ? code : "UNEXPECTED_ERROR";
  return `${normalizedCode}: ${USER_MESSAGES[normalizedCode]}`;
}

export function scannerErrorMessage(error: unknown): string {
  if (typeof error === "object" && error !== null) {
    const bridgeError = error as ScannerBridgeError;
    if (bridgeError.errorCode) {
      return formatScannerError(bridgeError.errorCode);
    }
    if (bridgeError.kind) {
      const code = bridgeErrorCode(bridgeError.kind);
      if (code) {
        return formatScannerError(code);
      }
    }
  }
  return formatScannerError("UNEXPECTED_ERROR");
}

export function scannerDiagnosticMessage(diagnostic: unknown): string {
  if (
    typeof diagnostic === "object" &&
    diagnostic !== null &&
    "errorCode" in diagnostic &&
    typeof diagnostic.errorCode === "string" &&
    hasOwnMessage(diagnostic.errorCode)
  ) {
    return formatScannerError(diagnostic.errorCode);
  }
  return formatScannerError("UNEXPECTED_ERROR");
}
