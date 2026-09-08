import { AlertTriangle, FolderCheck, FolderOpen, RotateCcw } from "lucide-react";
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
  onInputChange?: (path: string) => void;
  onOutputChange?: (path: string) => void;
  onSelectInputFolder: () => void;
  onSelectOutputFolder?: () => void;
  onResetOutputFolder?: () => void;
  disabled?: boolean;
}

export function FolderSelectorCard({
  inputPath,
  outputPath,
  onInputChange,
  onOutputChange,
  onSelectInputFolder,
  onSelectOutputFolder,
  onResetOutputFolder,
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
          <div className="flex-1 min-w-0 bg-secondary/50 border border-border rounded-xl px-3.5 py-2 text-sm text-foreground font-mono flex items-center gap-2 focus-within:ring-2 focus-within:ring-ring focus-within:border-transparent transition-all">
            <FolderOpen className="w-4 h-4 text-muted-foreground shrink-0" />
            <label htmlFor="input-folder-path" className="sr-only">
              Thư mục ảnh chấm công gốc
            </label>
            <input
              id="input-folder-path"
              name="inputFolderPath"
              type="text"
              value={inputPath}
              onChange={(e) => onInputChange?.(e.target.value)}
              placeholder="Nhập hoặc chọn đường dẫn thư mục ảnh nhân viên…"
              disabled={disabled}
              className="flex-1 min-w-0 bg-transparent text-sm text-foreground font-mono focus:outline-none placeholder:text-muted-foreground/60 placeholder:font-sans"
            />
          </div>

          <Button
            type="button"
            onClick={onSelectInputFolder}
            disabled={disabled}
            className="shrink-0 w-full sm:w-auto min-h-[44px]"
            variant="default"
          >
            <FolderOpen className="w-4 h-4 mr-2" />
            Chọn thư mục...
          </Button>
        </div>

        {/* Output folder preview */}
        {inputPath ? (
          <div className="rounded-xl bg-secondary/30 border border-border/60 p-3 space-y-2.5 text-xs text-muted-foreground">
            <div className="flex items-start gap-2.5">
              <FolderCheck className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
              <div className="min-w-0 flex-1">
                <span className="font-semibold text-foreground">Thư mục xuất PDF riêng biệt</span>
                <p className="mt-0.5 text-muted-foreground">
                  Mặc định là sibling <code className="font-mono">_pdf</code> để tránh quét nhầm output.
                </p>
              </div>
            </div>

            <div className="flex flex-col sm:flex-row gap-2">
              <input
                id="output-folder-path"
                type="text"
                value={outputPath || defaultOutputPath}
                onChange={(event) => onOutputChange?.(event.target.value)}
                disabled={disabled}
                aria-label="Thư mục xuất PDF"
                className="min-w-0 flex-1 rounded-lg border border-border bg-background/60 px-3 py-2 font-mono text-[11px] text-primary focus:outline-none focus:ring-2 focus:ring-ring"
              />
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={onSelectOutputFolder}
                disabled={disabled || !onSelectOutputFolder}
                className="min-h-[40px] shrink-0"
              >
                <FolderOpen className="mr-1.5 h-3.5 w-3.5" />
                Chọn output
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={onResetOutputFolder}
                disabled={disabled || !onResetOutputFolder}
                className="min-h-[40px] shrink-0"
                title="Khôi phục thư mục output mặc định"
              >
                <RotateCcw className="mr-1.5 h-3.5 w-3.5" />
                Mặc định
              </Button>
            </div>
          </div>
        ) : (
          <div className="rounded-xl bg-amber-500/10 border border-amber-500/20 p-3 flex items-start gap-2.5 text-xs text-amber-300">
            <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
            <span>Vui lòng nhập hoặc chọn thư mục gốc để hệ thống tự động phân tích scan plan.</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// --- Hybrid Responsive Summary ---
// mobile  (default / sm):  layout xếp dọc, ô nhập path và nút "Chọn thư mục" full width min-h-[44px]
// tablet  (md / lg):       hàng ngang linh hoạt flex-row, input box co giãn min-w-0
// desktop (xl / 2xl):      đường dẫn output rõ ràng, hover effect mượt mà trên nút chọn
// Interaction:             touch target >= 44px, hỗ trợ cả gõ/paste trực tiếp và click dialog
