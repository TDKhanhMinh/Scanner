import {
  AlertTriangle,
  CheckCircle2,
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
  timestamp?: string;
}

export interface FileResultListProps {
  items: FileResultItem[];
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

export function FileResultList({ items }: FileResultListProps) {
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
            <StatusBadge status={item.status} />
          </div>

          <div className="mt-3 grid grid-cols-1 gap-2 text-xs md:grid-cols-2 md:gap-4">
            <div className="flex min-w-0 items-center gap-1.5 font-mono text-muted-foreground">
              <FileOutput className="h-3.5 w-3.5 shrink-0 text-primary" />
              <span className="truncate" title={item.targetPdf || "Chưa có output do file thất bại"}>
                {item.targetPdf || "Chưa tạo output"}
              </span>
            </div>
            <div className="text-muted-foreground">
              {item.message ? (
                <span className="text-amber-400">{item.message}</span>
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
