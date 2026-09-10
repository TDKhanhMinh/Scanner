import { CheckCircle2, ScanLine, ShieldCheck, Sparkles } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { ProductDetectorMode } from "@/types/scanner";

export interface DetectorModeSelectorProps {
  mode: ProductDetectorMode;
  onSelectMode: (mode: ProductDetectorMode) => void;
  disabled?: boolean;
}

const MODES: Array<{
  id: ProductDetectorMode;
  name: string;
  badge: string;
  description: string;
  icon: typeof Sparkles;
}> = [
  {
    id: "ai_enhanced",
    name: "AI Enhanced",
    badge: "Mặc định",
    description: "Segmentation kết hợp CV để tìm đủ 4 mép giấy, có fallback an toàn khi cần.",
    icon: Sparkles,
  },
  {
    id: "classic",
    name: "Classic",
    badge: "Fallback",
    description: "Nhận diện OpenCV cổ điển, phù hợp khi cần đường xử lý ổn định và đơn giản.",
    icon: ShieldCheck,
  },
];

export function DetectorModeSelector({
  mode,
  onSelectMode,
  disabled = false,
}: DetectorModeSelectorProps) {
  return (
    <Card className="border-border/80">
      <CardHeader className="pb-3">
        <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
          <CardTitle className="flex items-center gap-2 text-base sm:text-lg">
            <ScanLine className="h-5 w-5 text-primary" />
            Chế độ nhận diện tài liệu
          </CardTitle>
          <span className="text-xs text-muted-foreground">Không cần chọn tên model</span>
        </div>
        <CardDescription className="text-xs text-muted-foreground/80">
          Chọn cách hệ thống tìm 4 góc tờ giấy trước khi nắn phối cảnh.
        </CardDescription>
      </CardHeader>

      <CardContent className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        {MODES.map((option) => {
          const selected = option.id === mode;
          const Icon = option.icon;
          return (
            <button
              key={option.id}
              type="button"
              aria-pressed={selected}
              disabled={disabled}
              onClick={() => onSelectMode(option.id)}
              className={`flex min-h-[112px] flex-col justify-between rounded-xl border p-4 text-left transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${
                selected
                  ? "border-primary bg-primary/10 shadow-sm ring-1 ring-primary/30"
                  : "border-border/60 bg-secondary/30 text-muted-foreground hover:border-border hover:bg-secondary/60"
              } ${disabled ? "cursor-not-allowed opacity-50" : "cursor-pointer"}`}
            >
              <div>
                <div className="mb-1.5 flex items-center justify-between gap-2">
                  <span className="flex items-center gap-1.5 text-sm font-semibold text-foreground">
                    <Icon className={`h-4 w-4 ${selected ? "text-primary" : "text-muted-foreground"}`} />
                    {option.name}
                  </span>
                  <Badge variant="secondary" className="px-1.5 py-0 text-[10px] font-medium">
                    {option.badge}
                  </Badge>
                </div>
                <p className="text-xs leading-relaxed text-muted-foreground">{option.description}</p>
              </div>
              {selected && (
                <span className="mt-3 flex items-center gap-1 text-[11px] font-medium text-primary">
                  <CheckCircle2 className="h-3.5 w-3.5" /> Đang sử dụng
                </span>
              )}
            </button>
          );
        })}
      </CardContent>
    </Card>
  );
}
