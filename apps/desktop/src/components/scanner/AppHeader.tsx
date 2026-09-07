import { Scan, ShieldCheck, Cpu } from "lucide-react";
import { Badge } from "@/components/ui/badge";

export interface AppHeaderProps {
  appName?: string;
  version?: string;
  sidecarStatus?: "ready" | "busy" | "offline";
}

export function AppHeader({
  appName = "Attendance Scanner Desktop",
  version = "v0.1.0 MVP",
  sidecarStatus = "ready",
}: AppHeaderProps) {
  return (
    <header className="border-b border-border/80 bg-background/80 backdrop-blur-md px-4 sm:px-6 py-3.5 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto flex items-center justify-between gap-3">
        {/* Brand & Logo */}
        <div className="flex items-center gap-3 min-w-0">
          <div className="h-10 w-10 shrink-0 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center shadow-md shadow-blue-500/20 ring-1 ring-white/20">
            <Scan className="w-5 h-5 text-white" />
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <h1 className="text-base sm:text-lg font-bold tracking-tight text-foreground truncate">
                {appName}
              </h1>
              <Badge variant="secondary" className="text-[10px] font-mono py-0 px-1.5 shrink-0">
                {version}
              </Badge>
            </div>
            <p className="text-xs text-muted-foreground truncate hidden sm:block">
              Số hóa bảng chấm công tự động • 1 ảnh → 1 PDF độc lập
            </p>
          </div>
        </div>

        {/* System Badges */}
        <div className="flex items-center gap-2 shrink-0">
          <Badge
            variant="success"
            className="flex items-center gap-1.5 text-xs py-1 px-2.5 font-medium"
          >
            <ShieldCheck className="w-3.5 h-3.5" />
            <span className="hidden md:inline">100% Local-First</span>
            <span className="md:hidden">Local</span>
          </Badge>

          <Badge
            variant={sidecarStatus === "ready" ? "secondary" : "warning"}
            className="flex items-center gap-1.5 text-xs py-1 px-2.5 font-medium"
          >
            <Cpu className="w-3.5 h-3.5 text-blue-400" />
            <span className="hidden md:inline">Python 3.11 Sidecar</span>
            <span className="md:hidden">Sidecar</span>
          </Badge>
        </div>
      </div>
    </header>
  );
}

// --- Hybrid Responsive Summary ---
// mobile  (default / sm):  logo + tên app rút gọn, huy hiệu trạng thái thu gọn text (Local, Sidecar)
// tablet  (md / lg):       hiển thị subtitle, huy hiệu đầy đủ text "100% Local-First"
// desktop (xl / 2xl):      căn giữa theo max-w-7xl, spacing rộng rãi và border mờ cao cấp
// Interaction:             touch target an toàn, thông tin hệ thống trực quan không che khuất màn hình
