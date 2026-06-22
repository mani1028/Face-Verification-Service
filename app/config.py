import os


# =========================
# Model Configuration
# =========================

FACE_MODEL_NAME = os.getenv("FACE_MODEL_NAME", "buffalo_l").strip()

FACE_DEVICE = os.getenv("FACE_DEVICE", "cpu").strip().lower()
if FACE_DEVICE not in ["cpu", "gpu"]:
    FACE_DEVICE = "cpu"


# =========================
# Threshold Configuration
# =========================

def get_float_env(name: str, default: float) -> float:
    """
    Safely reads float values from environment variables.
    If the value is missing or invalid, it falls back to default.
    """

    value = os.getenv(name)

    if value is None or value.strip() == "":
        return default

    try:
        return float(value)
    except ValueError:
        return default


FACE_MATCH_THRESHOLD = get_float_env("FACE_MATCH_THRESHOLD", 0.60)

FACE_QUALITY_THRESHOLD = get_float_env("FACE_QUALITY_THRESHOLD", 0.70)

# Keep values inside safe ranges
if FACE_MATCH_THRESHOLD < 0.10 or FACE_MATCH_THRESHOLD > 1.00:
    FACE_MATCH_THRESHOLD = 0.60

if FACE_QUALITY_THRESHOLD < 0.10 or FACE_QUALITY_THRESHOLD > 1.00:
    FACE_QUALITY_THRESHOLD = 0.70



# =========================
# App Configuration
# =========================

APP_NAME = os.getenv(
    "APP_NAME",
    "Universal Face Recognition Service",
).strip()

APP_VERSION = os.getenv(
    "APP_VERSION",
    "1.1.0",
).strip()

APP_ENV = os.getenv(
    "APP_ENV",
    "production",
).strip().lower()


# =========================
# Runtime Configuration
# =========================

REQUEST_TIMEOUT_SECONDS = int(
    os.getenv("REQUEST_TIMEOUT_SECONDS", "180")
)

MAX_IMAGE_SIZE_MB = get_float_env("MAX_IMAGE_SIZE_MB", 8.0)

# =========================
# Detection Optimization
# =========================

FACE_DET_SIZE = int(os.getenv("FACE_DET_SIZE", "320"))

if FACE_DET_SIZE < 160 or FACE_DET_SIZE > 640:
    FACE_DET_SIZE = 320
