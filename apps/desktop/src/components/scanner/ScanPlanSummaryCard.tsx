import {
  Layers,
  FileCheck2,
  AlertCircle,
  FileClock,
  Scan,
  Users,
  Loader2,
  RefreshCw,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
} from "@/components/ui/card";
import type { BatchPeriod, ExportMode } from "@/types/scanner";

export interface ScanPlanStats {
  totalEmployees: number;
  totalImages: number;
  newFiles: number;
  modifiedFiles: number;
  unchangedFiles: number;
  rebuildFiles?: number;
  unsupportedFiles?: number;
  collisions?: string[];
  documentGroups?: number;
  expectedArtifacts?: number;
  completeGroups?: number;
  incompleteGroups?: number;
  ambiguousGroups?: number;
  pagesNeedingReview?: number;
}

export interface ScanPlanSummaryCardProps {
  stats: ScanPlanStats;
  isPlanning?: boolean;
  isScanning?: boolean;
  canScan?: boolean;
  period?: BatchPeriod;
  exportMode?: ExportMode;
  onRefreshPlan: () => void;
  onStartScan: () => void;
}

export function ScanPlanSummaryCard({
  stats,
  isPlanning = false,
  isScanning = false,
  canScan = false,
  period,
  exportMode = "PER_IMAGE",
  onRefreshPlan,
  onStartScan,
}: ScanPlanSummaryCardProps) {
  const rebuildFiles = stats.rebuildFiles ?? 0;
  const unsupportedFiles = stats.unsupportedFiles ?? 0;
  const filesToProcess = stats.newFiles + stats.modifiedFiles + rebuildFiles;
  const documentGroups = stats.documentGroups ?? 0;
  const expectedArtifacts = stats.expectedArtifacts ?? 0;

  return (
    <Card className="border-border/80 flex flex-col justify-between h-full">
      <div>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2 text-base font-semibold">
              <Layers className="w-4 h-4 text-primary" />
              Kế hoạch quét (Scan Plan)
            </CardTitle>
            <Button
              variant="ghost"
              size="sm"
              onClick={onRefreshPlan}
              disabled={isPlanning || isScanning}
              className="h-8 px-2 text-xs text-muted-foreground hover:text-foreground"
              title="Làm mới phân tích thư mục"
            >
              <RefreshCw className={`w-3.5 h-3.5 mr-1 ${isPlanning ? "animate-spin" : ""}`} />
              Làm mới
            </Button>
          </div>
          <CardDescription className="text-xs">
            {isPlanning
              ? "Đang phân tích thư mục và manifest..."
              : period
              ? `Kỳ ${period.year}-${String(period.month).padStart(2, "0")} • ${
                  exportMode === "GROUPED" ? "Ghép theo nhóm" : "Mỗi ảnh một PDF"
                }`
              : "Hệ thống tự nhận diện file mới và file đã sửa đổi"}
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-3 pt-1">
          {/* Employee Count */}
          <div className="flex items-center justify-between py-2 border-b border-border/40 text-xs">
            <span className="text-muted-foreground flex items-center gap-1.5">
              <Users className="w-3.5 h-3.5" /> Nhân viên:
            </span>
            <span className="font-semibold text-foreground text-sm">
              {stats.totalEmployees}
            </span>
          </div>

          {/* Total Images */}
          <div className="flex items-center justify-between py-2 border-b border-border/40 text-xs">
            <span className="text-muted-foreground flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5" /> Tổng số ảnh:
            </span>
            <span className="font-semibold text-foreground text-sm">
              {stats.totalImages}
            </span>
          </div>

          {/* New files */}
          <div className="flex items-center justify-between py-2 border-b border-border/40 text-xs">
            <span className="text-emerald-400 flex items-center gap-1.5 font-medium">
              <FileCheck2 className="w-3.5 h-3.5" /> File mới (New):
            </span>
            <Badge variant="success" className="font-mono text-xs">
              +{stats.newFiles}
            </Badge>
          </div>

          {/* Modified files */}
          <div className="flex items-center justify-between py-2 border-b border-border/40 text-xs">
            <span className="text-amber-400 flex items-center gap-1.5 font-medium">
              <AlertCircle className="w-3.5 h-3.5" /> Sửa đổi (Modified):
            </span>
            <Badge variant="warning" className="font-mono text-xs">
              {stats.modifiedFiles}
            </Badge>
          </div>

          {/* Missing output files */}
          <div className="flex items-center justify-between py-2 border-b border-border/40 text-xs">
            <span className="text-sky-400 flex items-center gap-1.5 font-medium">
              <RefreshCw className="w-3.5 h-3.5" /> Dựng lại (Rebuild):
            </span>
            <Badge variant="secondary" className="font-mono text-xs">
              {rebuildFiles}
            </Badge>
          </div>

          {/* Unchanged files */}
          <div className="flex items-center justify-between py-2 text-xs">
            <span className="text-muted-foreground flex items-center gap-1.5">
              <FileClock className="w-3.5 h-3.5" /> Bỏ qua (Unchanged):
            </span>
            <span className="font-mono text-xs text-muted-foreground font-semibold">
              {stats.unchangedFiles}
            </span>
          </div>

          {(exportMode === "GROUPED" || documentGroups > 0) && (
            <>
              <div className="flex items-center justify-between border-b border-border/40 py-2 text-xs">
                <span className="text-muted-foreground flex items-center gap-1.5">
                  <Layers className="h-3.5 w-3.5" /> Nhóm tài liệu:
                </span>
                <span className="font-mono text-sm font-semibold text-foreground">
                  {documentGroups}
                </span>
              </div>
              <div className="flex items-center justify-between border-b border-border/40 py-2 text-xs">
                <span className="text-muted-foreground flex items-center gap-1.5">
                  <FileCheck2 className="h-3.5 w-3.5" /> Artifact dự kiến:
                </span>
                <span className="font-mono text-sm font-semibold text-foreground">
                  {expectedArtifacts}
                </span>
              </div>
              <div className="grid grid-cols-3 gap-1.5 border-b border-border/40 py-2 text-[11px]">
                <span className="rounded-md bg-emerald-500/10 px-1.5 py-1 text-center text-emerald-400">
                  Đủ: {stats.completeGroups ?? 0}
                </span>
                <span className="rounded-md bg-amber-500/10 px-1.5 py-1 text-center text-amber-300">
                  Thiếu: {stats.incompleteGroups ?? 0}
                </span>
                <span className="rounded-md bg-destructive/10 px-1.5 py-1 text-center text-destructive">
                  Mơ hồ: {stats.ambiguousGroups ?? 0}
                </span>
              </div>
              {(stats.pagesNeedingReview ?? 0) > 0 && (
                <p className="rounded-lg border border-amber-500/20 bg-amber-500/10 px-2.5 py-2 text-xs text-amber-300">
                  Cần review {stats.pagesNeedingReview} page trước khi ghép PDF.
                </p>
              )}
            </>
          )}

          {/* Target to process highlight */}
          <div className="rounded-xl bg-primary/10 border border-primary/20 p-3 mt-3 flex items-center justify-between text-xs">
            <span className="font-medium text-foreground">Cần quét đợt này:</span>
            <span className="font-bold text-primary text-base">
              {filesToProcess} file
            </span>
          </div>

          {(unsupportedFiles > 0 || (stats.collisions?.length ?? 0) > 0) && (
            <div className="rounded-xl border border-amber-500/25 bg-amber-500/10 p-3 text-xs text-amber-300">
              <div className="flex items-center gap-1.5 font-semibold">
                <AlertCircle className="h-3.5 w-3.5" />
                Cần lưu ý trước khi quét
              </div>
              {unsupportedFiles > 0 && (
                <p className="mt-1.5">
                  Bỏ qua {unsupportedFiles} file không thuộc định dạng ảnh được hỗ trợ.
                </p>
              )}
              {(stats.collisions?.length ?? 0) > 0 && (
                <p className="mt-1.5">
                  Có {stats.collisions?.length} tên PDF có nguy cơ trùng; hệ thống đã áp dụng hậu tố ổn định.
                </p>
              )}
            </div>
          )}

          {!isPlanning && canScan && stats.totalImages === 0 && (
            <p className="rounded-xl border border-dashed border-border/80 p-3 text-xs text-muted-foreground">
              Không tìm thấy ảnh hợp lệ trực tiếp trong các thư mục nhân viên.
            </p>
          )}
        </CardContent>
      </div>

      <CardFooter className="flex flex-col gap-2 pt-2">
        <Button
          type="button"
          onClick={onStartScan}
          disabled={!canScan || isPlanning || isScanning || filesToProcess === 0}
          className="w-full shadow-md shadow-primary/20 text-sm font-semibold"
          size="lg"
        >
          {isScanning ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              Đang quét...
            </>
          ) : (
            <>
              <Scan className="w-4 h-4 mr-2" />
              Quét các file mới ({filesToProcess})
            </>
          )}
        </Button>

        <p className="text-[11px] text-muted-foreground text-center leading-relaxed">
          Tự động bỏ qua {stats.unchangedFiles} file đã xử lý để tiết kiệm thời gian.
        </p>
      </CardFooter>
    </Card>
  );
}

// --- Hybrid Responsive Summary ---
// mobile  (default / sm):  card full width, các hàng thống kê rõ ràng, nút bấm lớn dễ tap
// tablet  (md / lg):       hiển thị ở cột bên phải, chiều cao khớp cột bên trái
// desktop (xl / 2xl):      sticky placement khi cuộn, badge nổi bật với độ tương phản cao
// Interaction:             touch target >= 44px, feedback trạng thái loading khi scan
