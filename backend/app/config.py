import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_env_file():
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env_file()

DATA_DIR = Path(os.environ.get("LH_DATA_DIR", ROOT / "data"))
OUTPUT_DIR = DATA_DIR / "outputs"
UPLOAD_DIR = DATA_DIR / "uploads"
BATCH_DIR = DATA_DIR / "batches"
DB_PATH = DATA_DIR / "local_higgs.db"
WORKER_TEMPLATE = ROOT / "kaggle" / "worker.py"
CACHE_TEMPLATE = ROOT / "kaggle" / "cache_builder.py"

for d in (OUTPUT_DIR, UPLOAD_DIR, BATCH_DIR):
    d.mkdir(parents=True, exist_ok=True)

# DRY_RUN=1 -> no Kaggle; fake videos via ffmpeg (UI testing)
DRY_RUN = os.environ.get("LH_DRY_RUN", "0") == "1"
KERNEL_SLUG = os.environ.get("LH_KERNEL_SLUG", "local-higgs-worker")
CACHE_PREFIX = os.environ.get("LH_CACHE_PREFIX", "local-higgs-cache")
MODEL_ID = os.environ.get("LH_MODEL_ID", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
POLL_SECONDS = int(os.environ.get("LH_POLL_SECONDS", "30"))
MAX_BATCH = int(os.environ.get("LH_MAX_BATCH", "4"))
# collect jobs for this long before pushing, so several go in one GPU session
BATCH_WAIT_SECONDS = int(os.environ.get("LH_BATCH_WAIT_SECONDS", "20"))
# a batch Kaggle keeps "queued" this long is failed (no GPU given: phone not verified, quota used up)
QUEUE_TIMEOUT = int(os.environ.get("LH_QUEUE_TIMEOUT", "1200"))
SESSION_TIMEOUT = int(os.environ.get("LH_SESSION_TIMEOUT", str(4 * 3600)))
MAX_IMAGE_BYTES = 6 * 1024 * 1024
