import {
  AlertTriangle,
  CheckCircle2,
  Eye,
  FileOutput,
  FileText,
  XCircle,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";

export interface FileResultItem {
  id: string;
  relativePath: string;
  employeeName: string;
  sourceFile: string;
  targetPdf: string;
  status: "success" | "warning" | "failed";
  documentDetected: boolean;
  message?: string;
  detectionReason?: string;
  detectionReasonCodes?: string[];
  timestamp?: string;
}

export interface FileResultListProps {
  items: FileResultItem[];
  onPreview?: (relativePath: string) => void;
}

const WARNING_MESSAGE_MAP: Record<string, string> = {
  DOCUMENT_CLIPPED: "Ảnh chụp sát biên/thiếu góc — đã giữ nguyên ảnh gốc, chưa cắt gọt",
  DOCUMENT_NOT_DETECTED: "Không tìm thấy biên giấy rõ ràng — dùng ảnh gốc",
  WARP_FALLBACK: "Không thể nắn phẳng 4 góc — dùng ảnh gốc",
  IMAGE_DOWNSCALED: "Ảnh kích thước lớn — đã hạ tỷ lệ",
};

const DETECTION_REASON_MESSAGE_MAP: Record<string, string> = {
  SEGMENTATION_LOW_CONFIDENCE: "AI không đủ tự tin về vùng tờ giấy",
  MASK_INVALID: "Mask nhận diện không hợp lệ",
  MASK_AMBIGUOUS_COMPONENTS: "Ảnh có nhiều vùng giấy gây mơ hồ",
  QUAD_FIT_FAILED: "Không khớp được tứ giác 4 góc",
  CV_NO_CANDIDATE: "OpenCV không tìm thấy ứng viên phù hợp",
  HYBRID_AMBIGUOUS: "Các tín hiệu nhận diện đang mâu thuẫn",
  REFINEMENT_REJECTED: "Tinh chỉnh góc bị từ chối do chất lượng thấp",
  PERSPECTIVE_INVALID: "Phối cảnh 4 góc không hợp lệ",
  FALLBACK_FULL_IMAGE: "Đã giữ nguyên ảnh gốc để tránh cắt nhầm",
};

export function formatWarningMessage(message?: string): string {
  if (!message) return "";
  const codes = message.split(",").map((s) => s.trim());
  const formatted = codes.map((c) => WARNING_MESSAGE_MAP[c] ?? c);
  return formatted.join("; ");
}

export function formatDetectionReasonMessage(
  reason?: string,
  reasonCodes: string[] = [],
): string {
  const codes = [reason, ...reasonCodes].filter(
    (code, index, values): code is string => Boolean(code) && values.indexOf(code) === index,
  );
  return codes.map((code) => DETECTION_REASON_MESSAGE_MAP[code] ?? code).join("; ");
}

function StatusBadge({ status }: Pick<FileResultItem, "status">) {
  if (status === "success") {
    return (
      <Badge variant="success" className="shrink-0 text-[11px]">
        <CheckCircle2 className="mr-1 h-3 w-3" />
        Thành công
      </Badge>
    );
  }
  if (status === "warning") {
    return (
      <Badge variant="warning" className="shrink-0 text-[11px]">
        <AlertTriangle className="mr-1 h-3 w-3" />
        Cảnh báo
      </Badge>
    );
  }
  return (
    <Badge variant="destructive" className="shrink-0 text-[11px]">
      <XCircle className="mr-1 h-3 w-3" />
      Thất bại
    </Badge>
  );
}

export function FileResultList({ items, onPreview }: FileResultListProps) {
  if (items.length === 0) {
    return (
      <div className="rounded-2xl border border-dashed border-border/80 bg-card/20 p-8 text-center">
        <FileText className="mx-auto mb-2 h-8 w-8 text-muted-foreground opacity-50" />
        <p className="text-sm font-medium text-muted-foreground">
          Chưa có kết quả quét nào.
        </p>
        <p className="mt-1 text-xs text-muted-foreground/70">
          Các file ảnh sau khi được quét sẽ xuất hiện tại đây theo thời gian thực.
        </p>
      </div>
    );
  }

  return (
    <div
      role="region"
      aria-label="Hoạt động quét gần đây"
      aria-live="polite"
      className="space-y-3"
    >
      {items.map((item, index) => (
        <article
          key={item.id || index}
          className="recent-activity-item rounded-xl border border-border/70 bg-card/60 p-4 shadow-sm backdrop-blur-sm"
        >
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <span className="block text-[11px] font-semibold uppercase tracking-wider text-primary">
                {item.employeeName}
              </span>
              <p className="mt-0.5 truncate font-mono text-sm font-medium text-foreground">
                {item.sourceFile}
              </p>
              <p className="mt-1 truncate font-mono text-xs text-muted-foreground" title={item.relativePath}>
                {item.relativePath}
              </p>
            </div>
            <div className="flex items-center gap-2">
              {onPreview && (
                <button
                  type="button"
                  onClick={() => onPreview(item.relativePath)}
                  className="flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring sm:h-9 sm:w-9"
                  title={`Xem trước ${item.sourceFile}`}
                  aria-label={`Xem trước ${item.sourceFile}`}
                >
                  <Eye className="h-4 w-4" />
                </button>
              )}
              <StatusBadge status={item.status} />
            </div>
          </div>

          <div className="mt-3 grid grid-cols-1 gap-2 text-xs md:grid-cols-2 md:gap-4">
            <div className="flex min-w-0 items-center gap-1.5 font-mono text-muted-foreground">
              <FileOutput className="h-3.5 w-3.5 shrink-0 text-primary" />
              <span className="truncate" title={item.targetPdf || "Chưa có output do file thất bại"}>
                {item.targetPdf || "Chưa tạo output"}
              </span>
            </div>
            <div className="min-w-0 text-muted-foreground">
              {item.message ? (
                <div className="flex flex-col gap-0.5">
                  <span className="text-amber-400 font-medium">
                    {formatWarningMessage(item.message)}
                  </span>
                  {item.detectionReason || (item.detectionReasonCodes?.length ?? 0) > 0 ? (
                    <span className="text-[11px] text-muted-foreground">
                      {formatDetectionReasonMessage(item.detectionReason, item.detectionReasonCodes)}
                    </span>
                  ) : null}
                  {item.message.includes("DOCUMENT_CLIPPED") && (
                    <span className="text-[11px] text-muted-foreground">
                      Gợi ý: Chụp lại để lấy trọn 4 mép giấy hoặc nắn góc thủ công
                    </span>
                  )}
                </div>
              ) : item.documentDetected ? (
                <span className="text-emerald-400">Nắn thẳng 4 góc thành công</span>
              ) : (
                "Chuẩn"
              )}
            </div>
          </div>
        </article>
      ))}
    </div>
  );
}
