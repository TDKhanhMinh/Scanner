import { CalendarDays, FileStack, Files } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { BatchPeriod, ExportMode } from "@/types/scanner";

export interface BatchOptionsCardProps {
  period: BatchPeriod;
  exportMode: ExportMode;
  onPeriodChange: (period: BatchPeriod) => void;
  onExportModeChange: (mode: ExportMode) => void;
  disabled?: boolean;
}

const MONTHS = [
  "Tháng 1",
  "Tháng 2",
  "Tháng 3",
  "Tháng 4",
  "Tháng 5",
  "Tháng 6",
  "Tháng 7",
  "Tháng 8",
  "Tháng 9",
  "Tháng 10",
  "Tháng 11",
  "Tháng 12",
];

const EXPORT_OPTIONS: Array<{
  mode: ExportMode;
  label: string;
  description: string;
}> = [
  {
    mode: "PER_IMAGE",
    label: "Một ảnh → một PDF",
    description: "Giữ mỗi ảnh là một file PDF riêng.",
  },
  {
    mode: "GROUPED",
    label: "Nhiều ảnh → một PDF",
    description: "Ghép theo nhân viên và tháng/năm đã chọn.",
  },
];

function formatPeriod(period: BatchPeriod): string {
  return `${period.year}-${String(period.month).padStart(2, "0")}`;
}

export function BatchOptionsCard({
  period,
  exportMode,
  onPeriodChange,
  onExportModeChange,
  disabled = false,
}: BatchOptionsCardProps) {
  return (
    <Card className="border-border/80">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-base sm:text-lg">
          <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary/20 text-xs font-bold text-primary">
            2
          </span>
          Thời gian và cách xuất
        </CardTitle>
        <CardDescription className="text-xs text-muted-foreground/80">
          Kỳ được chọn là dữ liệu nghiệp vụ của đợt quét, không tự lấy lại từ ngày hệ thống.
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-4">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <div className="space-y-1.5">
            <label htmlFor="batch-month" className="flex items-center gap-1.5 text-xs font-medium">
              <CalendarDays className="h-3.5 w-3.5 text-primary" />
              Tháng xử lý
            </label>
            <select
              id="batch-month"
              name="batchMonth"
              value={period.month}
              onChange={(event) =>
                onPeriodChange({ ...period, month: Number(event.target.value) })
              }
              disabled={disabled}
              className="min-h-[44px] w-full rounded-lg border border-border bg-background/60 px-3 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
            >
              {MONTHS.map((label, index) => (
                <option key={label} value={index + 1}>
                  {label}
                </option>
              ))}
            </select>
          </div>

          <div className="space-y-1.5">
            <label htmlFor="batch-year" className="flex items-center gap-1.5 text-xs font-medium">
              <CalendarDays className="h-3.5 w-3.5 text-primary" />
              Năm xử lý
            </label>
            <input
              id="batch-year"
              name="batchYear"
              type="number"
              min={1}
              max={9999}
              step={1}
              value={period.year}
              onChange={(event) => onPeriodChange({ ...period, year: Number(event.target.value) })}
              disabled={disabled}
              className="min-h-[44px] w-full rounded-lg border border-border bg-background/60 px-3 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
            />
          </div>
        </div>

        <fieldset className="space-y-2">
          <legend className="flex items-center gap-1.5 text-xs font-medium">
            <FileStack className="h-3.5 w-3.5 text-primary" />
            Kiểu xuất PDF
          </legend>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            {EXPORT_OPTIONS.map((option) => {
              const selected = option.mode === exportMode;
              return (
                <label
                  key={option.mode}
                  className={`flex min-h-[72px] cursor-pointer gap-3 rounded-xl border p-3 transition-colors ${
                    selected
                      ? "border-primary bg-primary/10 ring-1 ring-primary/30"
                      : "border-border/60 bg-secondary/30 hover:border-border hover:bg-secondary/60"
                  } ${disabled ? "cursor-not-allowed opacity-50" : ""}`}
                >
                  <input
                    type="radio"
                    name="exportMode"
                    value={option.mode}
                    checked={selected}
                    onChange={() => onExportModeChange(option.mode)}
                    disabled={disabled}
                    className="mt-1 h-4 w-4 accent-primary"
                  />
                  <span className="min-w-0">
                    <span className="block text-sm font-semibold text-foreground">{option.label}</span>
                    <span className="mt-0.5 block text-xs leading-relaxed text-muted-foreground">
                      {option.description}
                    </span>
                  </span>
                </label>
              );
            })}
          </div>
        </fieldset>

        <div className="flex items-start gap-2 rounded-xl border border-primary/20 bg-primary/5 p-3 text-xs text-muted-foreground">
          <Files className="mt-0.5 h-3.5 w-3.5 shrink-0 text-primary" />
          <p>
            Preview tên grouped: <span className="font-mono text-primary">{formatPeriod(period)}_&lt;Employee&gt;.pdf</span>
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
