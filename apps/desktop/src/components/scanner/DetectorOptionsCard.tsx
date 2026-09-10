import { Bug, RefreshCw } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export interface DetectorOptionsCardProps {
  debugDiagnostics: boolean;
  reprocess: boolean;
  needsReprocess: number;
  onDebugDiagnosticsChange: (enabled: boolean) => void;
  onReprocessChange: (enabled: boolean) => void;
  disabled?: boolean;
}

export function DetectorOptionsCard({
  debugDiagnostics,
  reprocess,
  needsReprocess,
  onDebugDiagnosticsChange,
  onReprocessChange,
  disabled = false,
}: DetectorOptionsCardProps) {
  return (
    <Card className="border-border/80">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-base sm:text-lg">
          <Bug className="h-4 w-4 text-primary" />
          Tùy chọn nâng cao
        </CardTitle>
        <CardDescription className="text-xs text-muted-foreground/80">
          Chỉ bật khi cần kiểm tra chất lượng hoặc chạy lại output cũ.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3 text-xs">
        <label className={`flex gap-3 rounded-xl border border-border/60 bg-secondary/30 p-3 ${disabled ? "cursor-not-allowed opacity-50" : "cursor-pointer"}`}>
          <input
            type="checkbox"
            checked={debugDiagnostics}
            onChange={(event) => onDebugDiagnosticsChange(event.target.checked)}
            disabled={disabled}
            className="mt-0.5 h-4 w-4 accent-primary"
          />
          <span>
            <span className="block font-semibold text-foreground">Bật diagnostics</span>
            <span className="mt-0.5 block leading-relaxed text-muted-foreground">
              Hiển thị thêm thông tin an toàn về đường quyết định; không ghi mask thô vào manifest.
            </span>
          </span>
        </label>

        <label className={`flex gap-3 rounded-xl border border-border/60 bg-secondary/30 p-3 ${disabled ? "cursor-not-allowed opacity-50" : "cursor-pointer"}`}>
          <input
            type="checkbox"
            checked={reprocess}
            onChange={(event) => onReprocessChange(event.target.checked)}
            disabled={disabled || needsReprocess === 0}
            className="mt-0.5 h-4 w-4 accent-primary"
          />
          <span>
            <span className="flex items-center gap-1.5 font-semibold text-foreground">
              <RefreshCw className="h-3.5 w-3.5 text-primary" />
              Cho phép reprocess
            </span>
            <span className="mt-0.5 block leading-relaxed text-muted-foreground">
              {needsReprocess > 0
                ? `${needsReprocess} file cần chạy lại theo detector hiện tại.`
                : "Không có output cũ cần chạy lại."}
            </span>
          </span>
        </label>
      </CardContent>
    </Card>
  );
}
