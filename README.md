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
│   └── setup-env.ps1             # Kịch bản khởi tạo môi trường tự động
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

## 5. Quy tắc Cam kết (Definition of Done)
- 100% xử lý hoàn toàn local-first, không OCR/AI/cloud ở MVP.
- Không hardcode đường dẫn tuyệt đối của máy lập trình viên.
- Toàn bộ giao tiếp giữa Tauri và Python Sidecar thực hiện qua chuẩn JSONL trên `stdout`; mọi log debug ghi ra `stderr`.
