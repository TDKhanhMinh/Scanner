# V2 ONNX Runtime Infrastructure

AS-37 adds model-independent ONNX Runtime infrastructure for the Python
sidecar. It owns model discovery, session lifecycle, input/output validation,
provider/thread configuration, structured errors, and a small startup versus
steady-state benchmark. MobileNetV3/DeepLabV3 preprocessing, postprocessing,
and detector policy remain out of scope for this task.

## Session lifecycle

```python
import numpy as np

from attendance_scanner.onnx_runtime import OnnxInferenceService

service = OnnxInferenceService("models/document-segmentation.onnx")
info = service.load()  # lazy construction; safe to call repeatedly
result = service.infer(np.zeros((1, 3, 224, 224), dtype=np.float32))
service.close()  # explicit batch lifecycle boundary
```

The service creates one `onnxruntime.InferenceSession` per lifecycle and reuses
it for every `infer` call. Calls are serialized by a re-entrant lock so a
shared service behaves deterministically when a batch worker invokes it. The
default configuration is CPU-only, one intra-op thread, one inter-op thread,
sequential execution, and all graph optimizations. GPU/other providers may be
configured only when CPU remains present as the baseline and the requested
provider is installed.

## Model resolution and validation

Relative model paths are searched in explicit `search_roots`, current working
directory, the development package root, PyInstaller's `_MEIPASS` extraction
directory, and beside the packaged executable. Absolute paths are used as-is.
The model is never embedded in JSONL output.

On load, the service records the model SHA-256 and validates:

- at least one input and output;
- expected input name and optional static shape, when configured;
- expected output names, when configured;
- requested providers against `onnxruntime.get_available_providers()`.

Missing, corrupt/invalid, unsupported I/O, unavailable provider, invalid input,
and inference failures become `OnnxRuntimeError` with a stable code,
user-safe Vietnamese message, and a developer diagnostic. No expected
no-document or model failure is printed to sidecar stdout by this layer.

## Packaging

`scanner/requirements-sidecar.lock` pins `onnxruntime` and its direct runtime
dependencies. The PyInstaller spec explicitly collects ONNX Runtime native
libraries/data and the required hidden imports. Model files are not silently
bundled: a release configuration must choose an approved model artifact and
path, then verify its hash as part of the later model/packaging gates.

## Micro-benchmark

Run the development benchmark against a model and its input tensor shape:

```powershell
python scripts/benchmark-onnx-runtime.py `
  --model D:\models\document-segmentation.onnx `
  --input-shape 1,3,224,224 `
  --iterations 20
```

The JSON output separates session startup, first inference, and reused
inference mean/P50/P95/P99, and includes session creation count and runtime
provider facts. It intentionally does not claim production latency, RAM, or
packaged footprint; those measurements belong to AS-58/AS-59 with the approved
model and clean Windows artifact.

## Tests and boundary

`scanner/tests/test_onnx_runtime.py` uses the small ONNX Runtime sample model to
prove successful load/inference, lazy reuse, I/O/provider validation, structured
missing/corrupt/input errors, model path resolution, and the micro-benchmark.
No MobileNetV3/DeepLabV3 provider is implemented here, and no PDF/export or
sidecar JSONL behavior is changed by this infrastructure task.
