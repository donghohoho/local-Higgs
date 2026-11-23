"""Saved model parts on Kaggle.

Each part is a small CPU notebook (kaggle/cache_builder.py) that downloads a
group of model files once. Its output is attached to every video run, so the
GPU session skips the download. Kaggle keeps up to 20GB of output per notebook,
hence the split.
"""

import threading
import time

from . import config

PARTS = [
    {"id": "wan-core", "label": "영상 모델 · VAE·설정", "gb": 3, "use": "video"},
    {"id": "wan-transformer", "label": "영상 모델 · 본체", "gb": 10, "use": "video"},
    {"id": "wan-text", "label": "영상 모델 · 텍스트 인코더", "gb": 11, "use": "video"},
    {"id": "speech", "label": "대사·립싱크", "gb": 9, "use": "speech"},
    {"id": "sfx", "label": "효과음", "gb": 9, "use": "sfx"},
]
IDS = [p["id"] for p in PARTS]
REFRESH_S = 60


class Cache:
    def __init__(self, runner):
        self.runner = runner
        self.lock = threading.Lock()
        self.state = {p: {"status": "unknown", "message": ""} for p in IDS}
        self.checked = 0.0

    def refresh(self, force=False):
        if not force and time.time() - self.checked < REFRESH_S:
            return
        self.checked = time.time()
        for p in IDS:
            try:
                st, msg = self.runner.cache_poll(p)
            except BaseException as e:  # not authenticated yet
                st, msg = "unknown", str(e)[-200:]
            with self.lock:
                self.state[p] = {"status": st, "message": msg}

    def build(self, parts=None):
        self.refresh(force=True)
        started = []
        for p in parts or IDS:
            if p not in IDS or self.state[p]["status"] in ("done", "running", "unknown"):
                continue
            try:
                self.runner.build_cache(p)
            except Exception as e:  # e.g. too many Kaggle sessions at once: try again later
                with self.lock:
                    self.state[p] = {"status": "failed", "message": f"시작 실패: {str(e)[-300:]}"}
                continue
            with self.lock:
                self.state[p] = {"status": "running", "message": "저장 중"}
            started.append(p)
        self.checked = time.time()
        return started

    def ready_sources(self):
        """kernel_sources for the video notebook: every finished part."""
        self.refresh()
        with self.lock:
            done = [p for p in IDS if self.state[p]["status"] == "done"]
        return [self.runner.kernel_ref(f"{config.CACHE_PREFIX}-{p}") for p in done]

    def video_ready(self):
        with self.lock:
            return all(self.state[p["id"]]["status"] == "done" for p in PARTS if p["use"] == "video")

    def public(self):
        self.refresh()
        with self.lock:
            return [{**p, **self.state[p["id"]]} for p in PARTS]
