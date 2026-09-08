# Attendance Scanner Desktop

Ứng dụng desktop nội bộ hỗ trợ quét và số hóa ảnh chấm công theo thư mục nhân viên, tự động phát hiện file mới/sửa đổi (incremental scan), nắn thẳng tài liệu bằng OpenCV và xuất **mỗi ảnh thành một PDF riêng**.

## 1. Cấu trúc Dự án (Monorepo Layout)

```plain text
attendance-scanner/
├── apps/
│   └── desktop/                  # Desktop Shell (Tauri 2 + React + TypeScript + Vite + Tailwind CSS)
│       ├── src/                  # React UI components & state
│       ├── src-tauri/            # Tauri 2 Rust configuration & commands
│       └── package.json
├── scanner/                      # Python Engine Core (OpenCV, NumPy, Pillow, Pydantic)
│   ├── pyproject.toml            # Cấu hình gói và dependencies
│   ├── src/attendance_scanner/   # CLI, Discovery, State manifest, Image pipeline
│   └── tests/                    # Pytest test suite
├── fixtures/
│   └── scanner/                  # Tập ảnh mẫu kiểm thử (không chứa PII)
├── scripts/
│   ├── setup-env.ps1             # Kịch bản khởi tạo môi trường tự động
│   ├── build-sidecar.ps1         # Build PyInstaller onefile cho Windows
│   └── test-sidecar.ps1          # Smoke test sidecar không cần Python
└── README.md
```

## 2. Yêu cầu Môi trường (Prerequisites)

- **Node.js**: v20+ (khuyến nghị v22+)
- **Rust & Cargo**: v1.75+ (hỗ trợ Tauri 2)
- **Python**: 3.11+ hoặc 3.12 (kèm `pip` và `venv`)
- **Hệ điều hành**: Windows 10/11 x64

## 3. Khởi tạo & Cài đặt Môi trường Nhanh

Chạy script PowerShell sau từ thư mục gốc của repo:

```powershell
.\scripts\setup-env.ps1
```

Hoặc cài đặt thủ công:

### Cài đặt Python Scanner
```powershell
cd scanner
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

### Cài đặt Desktop App
```powershell
cd apps\desktop
npm install
```

## 4. Chạy Phát triển (Development)

### Chạy kiểm thử Python Engine
```powershell
.\scanner\.venv\Scripts\pytest scanner\tests
```

### Kiểm tra CLI Sidecar
```powershell
.\scanner\.venv\Scripts\python -m attendance_scanner.cli --help
```

### Chạy Frontend Tests & Typecheck
```powershell
cd apps\desktop
npm run typecheck
npm test
npm run build
```

### Khởi chạy Desktop App (Tauri Dev Mode)
```powershell
cd apps\desktop
npm run tauri dev
```

### Build và smoke test Windows sidecar

AS-18 dùng PyInstaller onefile để Tauri phân phối trực tiếp một executable,
đồng thời vẫn giữ console stdout/stderr cho JSONL và diagnostics.

Chạy từ thư mục gốc:

    .\scanner\.venv\Scripts\python -m pip install -r scanner\requirements-sidecar.lock
    .\scripts\build-sidecar.ps1
    .\scripts\test-sidecar.ps1 -InputRoot "C:\path\to\representative\images"
    .\scripts\build-windows.ps1 -SmokeTestInputRoot "C:\path\to\representative\images"

Thư mục smoke test phải có ít nhất một file .jpg, .png và .webp. Script kiểm
tra --version, --help, plan, scan-batch, JSONL protocol v1 và PDF đầu ra; state
được cô lập trong thư mục tạm. Build tạo artifact theo tên
attendance-scanner-sidecar-x86_64-pc-windows-msvc.exe dưới
apps/desktop/src-tauri/binaries/.

File executable trong thư mục binaries là artifact được tạo tự động và được
ignore khỏi Git history. Workflow Windows sidecar thực hiện build, smoke test,
upload artifact và chạy tauri build để clean checkout luôn tái tạo đúng sidecar.

build-windows.ps1 là pipeline release chính: kiểm tra product metadata/icon,
build frontend, build sidecar, chạy smoke/relaunch test nếu có InputRoot, sau đó
tạo các installer MSI/NSIS và build/windows/release-manifest.json. Bản cài đặt
không cần Python, Node hoặc npm; state của scanner nằm trong
APPDATA/attendance-scanner/state. Uninstall không được xóa thư mục input/output
của người dùng.

## 5. Quy tắc Cam kết (Definition of Done)
- 100% xử lý hoàn toàn local-first, không OCR/AI/cloud ở MVP.
- Không hardcode đường dẫn tuyệt đối của máy lập trình viên.
- Toàn bộ giao tiếp giữa Tauri và Python Sidecar thực hiện qua chuẩn JSONL trên `stdout`; mọi log debug ghi ra `stderr`.
