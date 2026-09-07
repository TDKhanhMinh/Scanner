# Attendance Scanner Engine

Core Python scanner engine running as a local sidecar for the Attendance Scanner Desktop application.

## CLI Usage

```bash
# Plan scan without processing
python -m attendance_scanner.cli plan --input "D:\ChamCong\all"

# Run batch processing
python -m attendance_scanner.cli scan-batch --input "D:\ChamCong\all" --output "D:\ChamCong\all_pdf" --mode gray --workers 3
```
