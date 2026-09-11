# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller onefile spec for the Attendance Scanner Windows sidecar."""

from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)


SCANNER_ROOT = Path(SPECPATH).resolve().parent
SOURCE_ROOT = SCANNER_ROOT / "src"

# Explicitly collect native/data assets for the CV and image stack. Keep the
# collection scoped to runtime assets so development/test modules do not inflate
# the production sidecar.
datas = []
binaries = collect_dynamic_libs("cv2") + collect_dynamic_libs("onnxruntime")
hiddenimports = collect_submodules("attendance_scanner")
for package_name in ("cv2", "numpy", "PIL", "onnxruntime"):
    datas.extend(collect_data_files(package_name))
models_dir = SOURCE_ROOT / "attendance_scanner" / "models"
if models_dir.is_dir():
    datas.append((str(models_dir), "attendance_scanner/models"))
hiddenimports.extend(
    [
        "cv2.cv2",
        "onnxruntime.capi._pybind_state",
        "pydantic_core._pydantic_core",
    ]
)


analysis = Analysis(
    [str(SOURCE_ROOT / "attendance_scanner" / "cli.py")],
    pathex=[str(SOURCE_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter"],
    noarchive=False,
)
pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    [],
    name="attendance-scanner-sidecar",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
)
