# Attendance Scanner Engine

Core Python scanner engine running as a local sidecar for the Attendance Scanner Desktop application.

## CLI Usage

```bash
# Plan scan without processing
python -m attendance_scanner.cli plan --input "<path_to_input_images>"

# Run batch processing
python -m attendance_scanner.cli scan-batch --input "<path_to_input_images>" --output "<path_to_output_pdf>" --mode gray --workers 3
```
