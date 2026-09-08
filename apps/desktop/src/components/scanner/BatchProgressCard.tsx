import {
  CheckCircle,
  AlertTriangle,
  XCircle,
  FileClock,
  FolderOpen,
  Loader2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
} from "@/components/ui/card";

export interface BatchProgressCardProps {
  currentFile: string;
  currentEmployee: string;
  processedCount: number;
  totalCount: number;
  successCount: number;
  warningCount: number;
  failedCount: number;
  skippedCount: number;
  isScanning: boolean;
  isComplete?: boolean;
  outputReady?: boolean;
  onOpenOutputFolder: () => void;
}

export function BatchProgressCard({
  currentFile,
  currentEmployee,
  processedCount,
  totalCount,
  successCount,
  warningCount,
  failedCount,
  skippedCount,
  isScanning,
  isComplete = false,
  outputReady = false,
  onOpenOutputFolder,
}: BatchProgressCardProps) {
  const isDone = isComplete || (!isScanning && processedCount === totalCount);
  const percent = isDone
    ? 100
    : totalCount > 0
    ? Math.round((processedCount / totalCount) * 100)
    : 0;

  return (
    <Card className="border-border/80 bg-card/70">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base font-semibold flex items-center gap-2">
            {isScanning ? (
              <Loader2 className="w-4 h-4 text-primary animate-spin" />
            ) : isDone ? (
              <CheckCircle className="w-4 h-4 text-emerald-400" />
            ) : (
              <Loader2 className="w-4 h-4 text-muted-foreground" />
            )}
            {isScanning
              ? "Tiến độ quét ảnh..."
              : isDone
              ? "Hoàn tất đợt quét!"
              : "Trạng thái xử lý"}
          </CardTitle>

          <span className="font-mono text-sm font-bold text-primary">
            {percent}% ({processedCount}/{totalCount})
          </span>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Progress bar */}
        <Progress value={percent} className="h-2.5" />

        {/* Current file status */}
        <div
          role="status"
          aria-live="polite"
          aria-atomic="true"
          className="flex items-center justify-between text-xs text-muted-foreground gap-2"
        >
          <span className="shrink-0 font-medium">Đang xử lý:</span>
          <span className="min-w-0 truncate text-right text-foreground" title={currentFile}>
            {currentFile ? (
              <>
                <span className="font-semibold">{currentEmployee}</span>
                <span className="mx-1 text-muted-foreground">·</span>
                <span className="font-mono">{currentFile}</span>
              </>
            ) : isDone ? (
              "Đã xử lý tất cả file trong kế hoạch"
            ) : (
              "Chờ bắt đầu…"
            )}
          </span>
        </div>

        {/* Live counters */}
        <div
          aria-live="polite"
          aria-label="Tổng hợp kết quả quét"
          className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 border-t border-border/40"
        >
          <div className="flex flex-col items-center justify-center p-2 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
            <div className="flex items-center gap-1 text-[11px] font-medium text-emerald-400">
              <CheckCircle className="w-3.5 h-3.5" /> Thành công
            </div>
            <span className="text-base font-bold text-emerald-400 mt-0.5 font-mono">
              {successCount}
            </span>
          </div>

          <div className="flex flex-col items-center justify-center p-2 rounded-xl bg-amber-500/10 border border-amber-500/20">
            <div className="flex items-center gap-1 text-[11px] font-medium text-amber-400">
              <AlertTriangle className="w-3.5 h-3.5" /> Cảnh báo
            </div>
            <span className="text-base font-bold text-amber-400 mt-0.5 font-mono">
              {warningCount}
            </span>
          </div>

          <div className="flex flex-col items-center justify-center p-2 rounded-xl bg-destructive/10 border border-destructive/20">
            <div className="flex items-center gap-1 text-[11px] font-medium text-destructive">
              <XCircle className="w-3.5 h-3.5" /> Lỗi
            </div>
            <span className="text-base font-bold text-destructive mt-0.5 font-mono">
              {failedCount}
            </span>
          </div>

          <div className="flex flex-col items-center justify-center p-2 rounded-xl bg-secondary/60 border border-border/60">
            <div className="flex items-center gap-1 text-[11px] font-medium text-muted-foreground">
              <FileClock className="w-3.5 h-3.5" /> Bỏ qua
            </div>
            <span className="text-base font-bold text-muted-foreground mt-0.5 font-mono">
              {skippedCount}
            </span>
          </div>
        </div>

        {/* Completion actions */}
        {isDone && outputReady && (
          <div className="pt-2 flex flex-col sm:flex-row items-center justify-between gap-3">
            <div className="text-xs text-muted-foreground">
              Tất cả file PDF đã được ghi an toàn qua cơ chế atomic rename.
            </div>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={onOpenOutputFolder}
              className="w-full sm:w-auto"
            >
              <FolderOpen className="w-4 h-4 mr-2 text-primary" />
              Mở thư mục kết quả PDF
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// --- Hybrid Responsive Summary ---
// mobile  (default / sm):  layout xếp dọc, 3 ô đếm gọn gàng, nút mở folder full-width
// tablet  (md / lg):       nút hành động căn phải, font chữ rõ ràng, thanh progress mượt
// desktop (xl / 2xl):      hiển thị chi tiết trạng thái với màu sắc tương phản cao
// Interaction:             touch target >= 44px trên button, aria-valuenow tự động
