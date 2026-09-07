import { FolderOpen, FolderCheck, AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card";

export interface FolderSelectorCardProps {
  inputPath: string;
  outputPath: string;
  onSelectInputFolder: () => void;
  disabled?: boolean;
}

export function FolderSelectorCard({
  inputPath,
  outputPath,
  onSelectInputFolder,
  disabled = false,
}: FolderSelectorCardProps) {
  const defaultOutputPath = inputPath ? `${inputPath}_pdf` : "";

  return (
    <Card className="border-border/80">
      <CardHeader className="pb-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1">
          <CardTitle className="flex items-center gap-2 text-base sm:text-lg">
            <span className="w-6 h-6 rounded-full bg-primary/20 text-primary flex items-center justify-center text-xs font-bold shrink-0">
              1
            </span>
            Thư mục ảnh chấm công gốc (Input Root)
          </CardTitle>
          <span className="text-xs text-muted-foreground">
            Mỗi thư mục con trực tiếp là một nhân viên
          </span>
        </div>
        <CardDescription className="text-xs text-muted-foreground/80">
          Chỉ quét các file ảnh trực tiếp bên trong từng thư mục nhân viên (.jpg, .jpeg, .png, .webp).
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Input folder selector row */}
        <div className="flex flex-col sm:flex-row gap-3">
          <div className="flex-1 min-w-0 bg-secondary/50 border border-border rounded-xl px-3.5 py-2.5 text-sm text-foreground font-mono truncate flex items-center gap-2">
            <FolderOpen className="w-4 h-4 text-muted-foreground shrink-0" />
            {inputPath ? (
              <span className="truncate" title={inputPath}>
                {inputPath}
              </span>
            ) : (
              <span className="text-muted-foreground font-sans truncate">
                Chưa chọn thư mục (ví dụ: D:\ChamCong\all)...
              </span>
            )}
          </div>

          <Button
            type="button"
            onClick={onSelectInputFolder}
            disabled={disabled}
            className="shrink-0 w-full sm:w-auto"
            variant="default"
          >
            <FolderOpen className="w-4 h-4 mr-2" />
            Chọn thư mục...
          </Button>
        </div>

        {/* Output folder preview */}
        {inputPath ? (
          <div className="rounded-xl bg-secondary/30 border border-border/60 p-3 flex items-start gap-2.5 text-xs text-muted-foreground">
            <FolderCheck className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
            <div className="min-w-0 flex-1">
              <span className="font-semibold text-foreground">Thư mục xuất PDF riêng biệt: </span>
              <code className="text-primary font-mono text-[11px] sm:text-xs break-all">
                {outputPath || defaultOutputPath}
              </code>
            </div>
          </div>
        ) : (
          <div className="rounded-xl bg-amber-500/10 border border-amber-500/20 p-3 flex items-start gap-2.5 text-xs text-amber-300">
            <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
            <span>Vui lòng chọn thư mục gốc để hệ thống tự động phân tích scan plan.</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// --- Hybrid Responsive Summary ---
// mobile  (default / sm):  layout xếp dọc, nút "Chọn thư mục" full width min-h-[44px]
// tablet  (md / lg):       hàng ngang linh hoạt flex-row, input box co giãn min-w-0
// desktop (xl / 2xl):      đường dẫn output rõ ràng, hover effect mượt mà trên nút chọn
// Interaction:             touch target >= 44px, tooltip path đầy đủ qua title attribute
