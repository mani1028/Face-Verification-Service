import os
import sys
from insightface.utils.storage import ensure_available

# Use the same path as set in Dockerfile ENV
INSIGHTFACE_HOME = os.getenv('INSIGHTFACE_HOME', '/root/.insightface')
os.environ['INSIGHTFACE_HOME'] = INSIGHTFACE_HOME
os.environ['INSIGHTFACE_ROOT'] = INSIGHTFACE_HOME

print(f"[preload] Using INSIGHTFACE_HOME={INSIGHTFACE_HOME}", flush=True)

model_name = os.getenv('FACE_MODEL_NAME', 'buffalo_l').strip()

print(f"[preload] Downloading {model_name} weights...", flush=True)
try:
    # ensure_available downloads and extracts to root/models/name
    model_dir = ensure_available('models', model_name, root=INSIGHTFACE_HOME)
    print(f"[preload] Model directory: {model_dir}", flush=True)

    # Verify the .onnx files are actually there
    onnx_files = [f for f in os.listdir(model_dir) if f.endswith('.onnx')]
    if not onnx_files:
        print(f"[preload] ERROR: No .onnx files found in {model_dir} after download!", file=sys.stderr, flush=True)
        sys.exit(1)

    print(f"[preload] ✅ {model_name} weights ready ({len(onnx_files)} .onnx files):", flush=True)
    for f in sorted(onnx_files):
        size = os.path.getsize(os.path.join(model_dir, f))
        print(f"[preload]    {f}  ({size // 1024 // 1024} MB)", flush=True)
        
except Exception as e:
    print(f"[preload] ERROR during model download/verify: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
