import { Gauge } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export interface WorkerSettingCardProps {
  workers: number | null;
  onWorkersChange: (workers: number | null) => void;
  disabled?: boolean;
}

export function WorkerSettingCard({
  workers,
  onWorkersChange,
  disabled = false,
}: WorkerSettingCardProps) {
  return (
    <Card className="border-border/80">
      <CardHeader className="pb-3">
        <div className="flex items-center gap-2">
          <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary/20 text-xs font-bold text-primary">
            3
          </span>
          <CardTitle className="flex items-center gap-2 text-base sm:text-lg">
            <Gauge className="h-4 w-4 text-primary" />
            Hiệu năng quét
          </CardTitle>
        </div>
        <CardDescription className="text-xs text-muted-foreground/80">
          Có thể để tự động hoặc giới hạn số ảnh xử lý đồng thời. Thiết lập này chỉ áp dụng cho phiên hiện tại.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <label className="flex flex-col gap-2 text-sm font-medium text-foreground sm:flex-row sm:items-center sm:justify-between">
          <span>Số worker</span>
          <select
            value={workers ?? "auto"}
            onChange={(event) => {
              const value = event.target.value;
              onWorkersChange(value === "auto" ? null : Number(value));
            }}
            disabled={disabled}
            className="min-h-[44px] rounded-lg border border-border bg-background px-3 text-sm font-normal text-foreground outline-none focus:ring-2 focus:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
          >
            <option value="auto">Tự động (khuyến nghị)</option>
            <option value="1">1 worker</option>
            <option value="2">2 workers</option>
            <option value="3">3 workers</option>
            <option value="4">4 workers</option>
          </select>
        </label>
      </CardContent>
    </Card>
  );
}
