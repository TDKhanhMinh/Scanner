import { useState } from "react";
import {
  FolderOpen,
  Scan,
  FileCheck2,
  AlertCircle,
  ShieldCheck,
  Cpu,
  Layers,
  Sparkles,
} from "lucide-react";

export function App() {
  const [inputPath, setInputPath] = useState<string>("");
  const [outputPath] = useState<string>("");
  const [scanMode, setScanMode] = useState<"gray" | "bw" | "color">("gray");

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col antialiased selection:bg-blue-600 selection:text-white">
      {/* Top Navigation Bar */}
      <header className="border-b border-slate-800/80 bg-slate-900/60 backdrop-blur-md px-6 py-4 sticky top-0 z-50">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-blue-500/20 ring-1 ring-white/20">
              <Scan className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight text-white flex items-center gap-2">
                Attendance Scanner Desktop
                <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20">
                  v0.1.0 MVP
                </span>
              </h1>
              <p className="text-xs text-slate-400">
                Tự động hóa số hóa bảng chấm công theo nhân viên • 1 ảnh → 1 PDF
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              <ShieldCheck className="w-3.5 h-3.5" />
              100% Local-First
            </span>
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-slate-800 text-slate-300 border border-slate-700">
              <Cpu className="w-3.5 h-3.5 text-blue-400" />
              Python 3.11 Sidecar
            </span>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-6xl w-full mx-auto p-6 space-y-6">
        {/* Banner Informational Notice */}
        <div className="rounded-xl border border-blue-500/20 bg-blue-950/20 p-4 flex items-start gap-3">
          <Sparkles className="w-5 h-5 text-blue-400 shrink-0 mt-0.5" />
          <div className="text-xs sm:text-sm text-slate-300 leading-relaxed">
            <strong className="text-blue-300 font-semibold">Cơ chế quét thông minh (Incremental Scan): </strong>
            Hệ thống chỉ quét và chuyển đổi các ảnh mới thêm hoặc ảnh bị thay đổi, tự động bỏ qua các ảnh đã tạo PDF thành công trước đó để tối ưu thời gian.
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column: Configuration & Folder Pickers (2 cols wide on desktop) */}
          <div className="lg:col-span-2 space-y-6">
            {/* Step 1: Input Folder */}
            <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6 backdrop-blur-sm shadow-sm space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-sm sm:text-base font-semibold text-white flex items-center gap-2">
                  <span className="w-6 h-6 rounded-full bg-blue-600/20 text-blue-400 flex items-center justify-center text-xs font-bold">1</span>
                  Thư mục gốc chứa ảnh nhân viên (Input Root)
                </h2>
                <span className="text-xs text-slate-400">Yêu cầu thư mục con trực tiếp là tên nhân viên</span>
              </div>

              <div className="flex flex-col sm:flex-row gap-3">
                <div className="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-4 py-2.5 text-sm text-slate-300 font-mono truncate focus-within:border-blue-500">
                  {inputPath || <span className="text-slate-500 font-sans">Chưa chọn thư mục (ví dụ: D:\ChamCong\all)...</span>}
                </div>
                <button
                  type="button"
                  onClick={() => setInputPath("D:\\ChamCong\\all")}
                  className="px-4 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 active:bg-blue-700 text-white text-sm font-medium transition-colors flex items-center justify-center gap-2 shadow-sm min-h-[44px]"
                >
                  <FolderOpen className="w-4 h-4" />
                  Chọn thư mục
                </button>
              </div>

              {inputPath && (
                <div className="text-xs text-slate-400 flex items-center gap-2">
                  <span>Thư mục PDF đầu ra mặc định:</span>
                  <code className="text-blue-400 font-mono">{outputPath || `${inputPath}_pdf`}</code>
                </div>
              )}
            </div>

            {/* Step 2: Scan Enhancement Mode */}
            <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6 backdrop-blur-sm shadow-sm space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-sm sm:text-base font-semibold text-white flex items-center gap-2">
                  <span className="w-6 h-6 rounded-full bg-blue-600/20 text-blue-400 flex items-center justify-center text-xs font-bold">2</span>
                  Chế độ xử lý ảnh (Scan Mode)
                </h2>
                <span className="text-xs text-slate-400">OpenCV document enhancement</span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <button
                  type="button"
                  onClick={() => setScanMode("gray")}
                  className={`p-4 rounded-xl border text-left transition-all min-h-[44px] ${
                    scanMode === "gray"
                      ? "border-blue-500 bg-blue-500/10 text-white shadow-sm ring-1 ring-blue-500/30"
                      : "border-slate-800 bg-slate-950/40 text-slate-400 hover:border-slate-700 hover:text-slate-200"
                  }`}
                >
                  <div className="font-semibold text-sm flex items-center justify-between">
                    Grayscale
                    {scanMode === "gray" && <span className="text-[10px] bg-blue-500/20 text-blue-300 px-1.5 py-0.5 rounded">Mặc định</span>}
                  </div>
                  <div className="text-xs text-slate-400 mt-1.5 leading-relaxed">
                    Giữ sắc nét nét bảng, độ tương phản mượt mà, phù hợp mực bút bi nhạt.
                  </div>
                </button>

                <button
                  type="button"
                  onClick={() => setScanMode("bw")}
                  className={`p-4 rounded-xl border text-left transition-all min-h-[44px] ${
                    scanMode === "bw"
                      ? "border-blue-500 bg-blue-500/10 text-white shadow-sm ring-1 ring-blue-500/30"
                      : "border-slate-800 bg-slate-950/40 text-slate-400 hover:border-slate-700 hover:text-slate-200"
                  }`}
                >
                  <div className="font-semibold text-sm">B&W (Nhị phân)</div>
                  <div className="text-xs text-slate-400 mt-1.5 leading-relaxed">
                    Adaptive threshold lọc sạch nền xám, tạo trang trắng chữ đen sắc nét.
                  </div>
                </button>

                <button
                  type="button"
                  onClick={() => setScanMode("color")}
                  className={`p-4 rounded-xl border text-left transition-all min-h-[44px] ${
                    scanMode === "color"
                      ? "border-blue-500 bg-blue-500/10 text-white shadow-sm ring-1 ring-blue-500/30"
                      : "border-slate-800 bg-slate-950/40 text-slate-400 hover:border-slate-700 hover:text-slate-200"
                  }`}
                >
                  <div className="font-semibold text-sm">Color Enhanced</div>
                  <div className="text-xs text-slate-400 mt-1.5 leading-relaxed">
                    Giữ nguyên màu mực chữ ký và con dấu đỏ, tăng tương phản màu nền.
                  </div>
                </button>
              </div>
            </div>
          </div>

          {/* Right Column: Scan Plan Summary & Action */}
          <div className="space-y-6">
            <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6 backdrop-blur-sm shadow-sm space-y-5">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <Layers className="w-4 h-4 text-blue-400" />
                Tổng quan Lập kế hoạch (Scan Plan)
              </h3>

              <div className="space-y-3">
                <div className="flex items-center justify-between text-xs py-2 border-b border-slate-800">
                  <span className="text-slate-400">Nhân viên phát hiện:</span>
                  <span className="font-semibold text-white">0</span>
                </div>
                <div className="flex items-center justify-between text-xs py-2 border-b border-slate-800">
                  <span className="text-slate-400">Tổng số file ảnh:</span>
                  <span className="font-semibold text-white">0</span>
                </div>
                <div className="flex items-center justify-between text-xs py-2 border-b border-slate-800">
                  <span className="text-emerald-400 flex items-center gap-1">
                    <FileCheck2 className="w-3.5 h-3.5" /> File mới (New):
                  </span>
                  <span className="font-semibold text-emerald-400">0</span>
                </div>
                <div className="flex items-center justify-between text-xs py-2 border-b border-slate-800">
                  <span className="text-amber-400 flex items-center gap-1">
                    <AlertCircle className="w-3.5 h-3.5" /> Sửa đổi (Modified):
                  </span>
                  <span className="font-semibold text-amber-400">0</span>
                </div>
                <div className="flex items-center justify-between text-xs py-2">
                  <span className="text-slate-400">Bỏ qua (Unchanged):</span>
                  <span className="font-semibold text-slate-400">0</span>
                </div>
              </div>

              <button
                type="button"
                disabled={!inputPath}
                className="w-full py-3 px-4 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-semibold transition-all shadow-md shadow-blue-500/20 flex items-center justify-center gap-2 min-h-[44px]"
              >
                <Scan className="w-4 h-4" />
                Quét các file mới (Scan New Files)
              </button>

              <p className="text-[11px] text-slate-500 text-center leading-relaxed">
                Quá trình quét chạy trên tiến trình Python nền độc lập, hỗ trợ tự phục hồi và tiếp tục khi có lỗi.
              </p>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/60 bg-slate-950 py-4 px-6 mt-auto">
        <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between text-xs text-slate-500 gap-2">
          <span>Attendance Scanner Desktop • Tauri 2 + React TS + OpenCV Sidecar</span>
          <span>Định dạng hỗ trợ: .jpg, .jpeg, .png, .webp</span>
        </div>
      </footer>
    </div>
  );
}

export default App;
