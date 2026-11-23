"""Talks to Kaggle (or fakes it in DRY_RUN mode).

submit(batch_id, jobs) -> ref      push a new kernel version that runs the batch
poll(ref)              -> (state, message)   state: running | done | failed
fetch(ref, dest)       -> result.json dict, files downloaded into dest
"""

import base64
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

from . import config


def _find(dest: Path, name: str):
    hits = list(dest.rglob(name))
    return hits[0] if hits else None


class KaggleRunner:
    def __init__(self):
        self._api = None
        self._lock = threading.Lock()

    def api(self):
        with self._lock:
            if self._api is None:
                # import late: the kaggle package tries to authenticate on import
                from kaggle.api.kaggle_api_extended import KaggleApi

                api = KaggleApi()
                api.authenticate()
                self._api = api
            return self._api

    def username(self):
        return self.api().get_config_value("username")

    def check(self):
        try:
            return {"ok": True, "user": self.username()}
        except SystemExit:  # authenticate() calls sys.exit when no token exists
            return {"ok": False, "error": "토큰 없음 (~/.kaggle/access_token)"}
        except Exception as e:
            return {"ok": False, "error": str(e) or e.__class__.__name__}

    def submit(self, batch_id, jobs):
        api = self.api()
        user = self.username()
        folder = config.BATCH_DIR / batch_id / "kernel"
        folder.mkdir(parents=True, exist_ok=True)

        payload = base64.b64encode(
            json.dumps({"batch_id": batch_id, "model": config.MODEL_ID, "jobs": jobs}).encode()
        ).decode()
        code = config.WORKER_TEMPLATE.read_text().replace(
            'JOBS_PAYLOAD = "__JOBS_PAYLOAD__"', f'JOBS_PAYLOAD = "{payload}"', 1
        )
        (folder / "worker.py").write_text(code)
        meta = {
            "id": f"{user}/{config.KERNEL_SLUG}",
            "title": config.KERNEL_SLUG,
            "code_file": "worker.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

        resp = api.kernels_push(str(folder), timeout=str(config.SESSION_TIMEOUT))
        err = getattr(resp, "error", None)
        if err:
            raise RuntimeError(f"Kaggle push failed: {err}")
        version = getattr(resp, "version_number", None)
        ref = f"{user}/{config.KERNEL_SLUG}"
        return f"{ref}/{version}" if version else ref

    def poll(self, ref):
        resp = self.api().kernels_status(ref)
        status = getattr(resp.status, "name", str(resp.status)).upper()
        msg = getattr(resp, "failure_message", "") or ""
        if "COMPLETE" in status:
            return "done", msg
        if "ERROR" in status or "CANCEL" in status:
            return "failed", msg or status
        return "running", status.lower()

    def fetch(self, ref, dest: Path):
        dest.mkdir(parents=True, exist_ok=True)
        self.api().kernels_output(ref, str(dest), force=True, quiet=True)
        f = _find(dest, "result.json")
        return json.loads(f.read_text()) if f else None

    def logs(self, ref):
        try:
            return self.api().kernels_logs(ref)[-3000:]
        except Exception as e:
            return f"(log unavailable: {e})"


class DryRunRunner:
    """Fakes Kaggle: after ~8s returns ffmpeg test videos."""

    def __init__(self):
        self._started = {}

    def check(self):
        return {"ok": True, "user": "dry-run"}

    def submit(self, batch_id, jobs):
        self._started[batch_id] = (time.time(), jobs)
        return batch_id

    def poll(self, ref):
        t0, _ = self._started.get(ref, (0, []))
        return ("done", "") if time.time() - t0 > 8 else ("running", "dry-run")

    def fetch(self, ref, dest: Path):
        _, jobs = self._started[ref]
        out = dest / "outputs"
        out.mkdir(parents=True, exist_ok=True)
        results = {}
        for j in jobs:
            secs = j["num_frames"] / j["fps"]
            path = out / f"{j['id']}.mp4"
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "lavfi", "-i", f"testsrc2=size={j['width']}x{j['height']}:rate={j['fps']}",
                "-t", f"{secs:.2f}", "-pix_fmt", "yuv420p", str(path),
            ]
            try:
                subprocess.run(cmd, check=True)
                results[j["id"]] = {"status": "done", "file": path.name, "seconds": 0}
            except Exception as e:
                results[j["id"]] = {"status": "failed", "error": f"ffmpeg: {e}"}
        return {"batch_id": ref, "results": results}

    def logs(self, ref):
        return "dry-run"


def make_runner():
    return DryRunRunner() if config.DRY_RUN else KaggleRunner()


def move_output(src_dir: Path, file_name: str):
    src = _find(src_dir, file_name)
    if not src:
        return None
    dst = config.OUTPUT_DIR / file_name
    shutil.move(str(src), dst)
    return dst.name
