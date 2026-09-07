import { CheckCircle2, Sparkles, FileText, Palette } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card";

export type ScanFilterMode = "gray" | "bw" | "color";

export interface ScanModeSelectorProps {
  mode: ScanFilterMode;
  onSelectMode: (mode: ScanFilterMode) => void;
  disabled?: boolean;
}

interface ModeOption {
  id: ScanFilterMode;
  name: string;
  badge?: string;
  icon: typeof FileText;
  description: string;
  highlight: string;
}

const MODES: ModeOption[] = [
  {
    id: "gray",
    name: "Grayscale",
    badge: "Mặc định",
    icon: Sparkles,
    description: "CLAHE + nhẹ denoise + làm nét. Giữ rõ nét chữ bút nhạt và kẻ bảng.",
    highlight: "Tốt nhất cho hầu hết bảng chấm công",
  },
  {
    id: "bw",
    name: "Black & White",
    icon: FileText,
    description: "Adaptive threshold nhị phân. Làm sạch nền hoàn toàn, nền trắng chữ đen.",
    highlight: "Thích hợp tài liệu chữ đậm rõ nét",
  },
  {
    id: "color",
    name: "Color Enhanced",
    icon: Palette,
    description: "Cân bằng sáng CLAHE trên kênh luminance, giữ nguyên màu chữ ký & con dấu đỏ.",
    highlight: "Bảo tồn màu mực dấu xác nhận",
  },
];

export function ScanModeSelector({
  mode,
  onSelectMode,
  disabled = false,
}: ScanModeSelectorProps) {
  return (
    <Card className="border-border/80">
      <CardHeader className="pb-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1">
          <CardTitle className="flex items-center gap-2 text-base sm:text-lg">
            <span className="w-6 h-6 rounded-full bg-primary/20 text-primary flex items-center justify-center text-xs font-bold shrink-0">
              2
            </span>
            Chế độ xử lý ảnh (Scan Mode)
          </CardTitle>
          <span className="text-xs text-muted-foreground">
            Pipeline OpenCV xử lý ảnh cục bộ
          </span>
        </div>
        <CardDescription className="text-xs text-muted-foreground/80">
          Chọn thuật toán xử lý phù hợp với chất lượng ảnh chụp bảng chấm công.
        </CardDescription>
      </CardHeader>

      <CardContent>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {MODES.map((item) => {
            const isSelected = mode === item.id;
            const Icon = item.icon;

            return (
              <button
                key={item.id}
                type="button"
                disabled={disabled}
                onClick={() => onSelectMode(item.id)}
                className={`p-4 rounded-xl border text-left transition-all relative flex flex-col justify-between min-h-[110px] cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${
                  isSelected
                    ? "border-primary bg-primary/10 shadow-sm ring-1 ring-primary/30"
                    : "border-border/60 bg-secondary/30 hover:border-border hover:bg-secondary/60 text-muted-foreground"
                } ${disabled ? "opacity-50 pointer-events-none" : ""}`}
              >
                <div>
                  <div className="flex items-center justify-between gap-2 mb-1.5">
                    <div className="flex items-center gap-1.5 font-semibold text-sm text-foreground">
                      <Icon className={`w-4 h-4 ${isSelected ? "text-primary" : "text-muted-foreground"}`} />
                      <span>{item.name}</span>
                    </div>

                    {item.badge ? (
                      <Badge variant="secondary" className="text-[10px] px-1.5 py-0 font-medium">
                        {item.badge}
                      </Badge>
                    ) : isSelected ? (
                      <CheckCircle2 className="w-4 h-4 text-primary shrink-0" />
                    ) : null}
                  </div>

                  <p className="text-xs text-muted-foreground leading-relaxed">
                    {item.description}
                  </p>
                </div>

                <div className="mt-3 pt-2 border-t border-border/40 text-[11px] font-medium text-primary/90">
                  {item.highlight}
                </div>
              </button>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}

// --- Hybrid Responsive Summary ---
// mobile  (default / sm):  grid-cols-1, các card chế độ xếp dọc, touch target thoải mái
// tablet  (md / lg):       grid-cols-3 hàng ngang 3 cột, so sánh trực quan các chế độ
// desktop (xl / 2xl):      hover variant rõ rệt, ring focus sắc nét, chiều cao thẻ đồng đều
// Interaction:             touch target >= 44px, nút tương tác phản hồi tức thời
