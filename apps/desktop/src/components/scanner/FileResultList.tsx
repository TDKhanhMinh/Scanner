import { FileText, CheckCircle2, AlertTriangle, XCircle, FileOutput } from "lucide-react";
import { Badge } from "@/components/ui/badge";

export interface FileResultItem {
  id: string;
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

export function FileResultList({ items }: FileResultListProps) {
  if (items.length === 0) {
    return (
      <div className="rounded-2xl border border-dashed border-border/80 p-8 text-center bg-card/20">
        <FileText className="w-8 h-8 text-muted-foreground mx-auto mb-2 opacity-50" />
        <p className="text-sm font-medium text-muted-foreground">
          Chưa có kết quả quét nào.
        </p>
        <p className="text-xs text-muted-foreground/70 mt-1">
          Các file ảnh sau khi được quét sẽ xuất hiện tại đây theo thời gian thực.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Mobile Card List View (< md) */}
      <div className="space-y-3 md:hidden">
        {items.map((item, index) => (
          <div
            key={item.id || index}
            className="rounded-xl border border-border/70 bg-card/60 p-4 space-y-2.5 shadow-sm backdrop-blur-sm"
          >
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0">
                <span className="text-[11px] font-semibold text-primary uppercase tracking-wider block">
                  {item.employeeName}
                </span>
                <p className="text-sm font-medium text-foreground truncate font-mono mt-0.5">
                  {item.sourceFile}
                </p>
              </div>

              {item.status === "success" && (
                <Badge variant="success" className="shrink-0 text-[11px]">
                  <CheckCircle2 className="w-3 h-3 mr-1" />
                  Thành công
                </Badge>
              )}
              {item.status === "warning" && (
                <Badge variant="warning" className="shrink-0 text-[11px]">
                  <AlertTriangle className="w-3 h-3 mr-1" />
                  Cảnh báo
                </Badge>
              )}
              {item.status === "failed" && (
                <Badge variant="destructive" className="shrink-0 text-[11px]">
                  <XCircle className="w-3 h-3 mr-1" />
                  Lỗi
                </Badge>
              )}
            </div>

            <div className="flex items-center gap-1.5 text-xs text-muted-foreground font-mono truncate">
              <FileOutput className="w-3.5 h-3.5 text-primary shrink-0" />
              <span className="truncate">{item.targetPdf}</span>
            </div>

            {item.message && (
              <p className="text-xs text-amber-300/90 bg-amber-500/10 rounded-lg p-2 leading-relaxed">
                {item.message}
              </p>
            )}
          </div>
        ))}
      </div>

      {/* Tablet & Desktop Table View (>= md) */}
      <div className="hidden md:block w-full overflow-x-auto rounded-xl border border-border/80 bg-card/50 shadow-sm">
        <table className="min-w-[640px] w-full text-sm text-left">
          <thead className="bg-muted/50 border-b border-border text-xs text-muted-foreground uppercase tracking-wider font-semibold">
            <tr>
              <th scope="col" className="py-3 px-4 w-12 text-center">#</th>
              <th scope="col" className="py-3 px-4">Nhân viên</th>
              <th scope="col" className="py-3 px-4">File ảnh nguồn</th>
              <th scope="col" className="py-3 px-4">Trạng thái</th>
              <th scope="col" className="py-3 px-4">PDF đầu ra</th>
              <th scope="col" className="py-3 px-4">Chẩn đoán / Ghi chú</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/50 text-foreground">
            {items.map((item, index) => (
              <tr key={item.id || index} className="hover:bg-muted/30 transition-colors">
                <td className="py-3 px-4 text-center text-xs text-muted-foreground font-mono">
                  {index + 1}
                </td>
                <td className="py-3 px-4 font-medium text-foreground">
                  {item.employeeName}
                </td>
                <td className="py-3 px-4 font-mono text-xs text-muted-foreground">
                  {item.sourceFile}
                </td>
                <td className="py-3 px-4">
                  {item.status === "success" && (
                    <Badge variant="success" className="text-xs">
                      <CheckCircle2 className="w-3 h-3 mr-1" />
                      Thành công
                    </Badge>
                  )}
                  {item.status === "warning" && (
                    <Badge variant="warning" className="text-xs">
                      <AlertTriangle className="w-3 h-3 mr-1" />
                      Fallback
                    </Badge>
                  )}
                  {item.status === "failed" && (
                    <Badge variant="destructive" className="text-xs">
                      <XCircle className="w-3 h-3 mr-1" />
                      Thất bại
                    </Badge>
                  )}
                </td>
                <td className="py-3 px-4 font-mono text-xs text-primary truncate max-w-[200px]" title={item.targetPdf}>
                  {item.targetPdf}
                </td>
                <td className="py-3 px-4 text-xs text-muted-foreground">
                  {item.message ? (
                    <span className="text-amber-400">{item.message}</span>
                  ) : item.documentDetected ? (
                    <span className="text-emerald-400">Nắn thẳng 4 góc thành công</span>
                  ) : (
                    "Chuẩn"
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// --- Hybrid Responsive Summary ---
// mobile  (default / sm):  chuyển sang Card list (mỗi file 1 thẻ riêng biệt, tránh scroll ngang khó thao tác)
// tablet  (md / lg):       hiển thị bảng đầy đủ với container bọc overflow-x-auto an toàn
// desktop (xl / 2xl):      bảng cố định tiêu đề, hover highlight từng hàng, hiển thị chẩn đoán chi tiết
// Interaction:             touch target an toàn trên mobile, hover effect mượt mà trên desktop
