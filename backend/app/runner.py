"""Talks to Kaggle (or fakes it in DRY_RUN mode).

submit(batch_id, jobs) -> ref      push a new kernel version that runs the batch
poll(ref)              -> (state, message)   state: running | done | failed
fetch(ref, dest)       -> result.json dict, files downloaded into dest
"""

import base64
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import config

TOKEN_FILE = Path("~/.kaggle/access_token").expanduser()
LEGACY_FILE = Path("~/.kaggle/kaggle.json").expanduser()


def explain_http_error(e):
    """Kaggle puts the real reason in the response body; show it."""
    body = ""
    resp = getattr(e, "response", None)
    if resp is not None:
        try:
            data = resp.json()
            body = data.get("message") or data.get("error") or str(data)
        except Exception:
            body = (resp.text or "")[:500]
    text = f"{e}" + (f"\nKaggle 응답: {body}" if body else "")
    low = text.lower()
    if "concurrent" in low or "running" in low or "session" in low:
        text += ("\n→ 같은 계정에서 GPU 노트북이 이미 돌고 있어요. kaggle.com/code 에서 "
                 "실행 중인 local-higgs 노트북을 Stop session 하거나, 다른 컴퓨터 작업이 끝난 뒤 재시도하세요.")
    elif "phone" in low or "verif" in low:
        text += "\n→ Kaggle 휴대폰 인증이 필요해요: kaggle.com → Settings → Phone Verification"
    elif "quota" in low:
        text += "\n→ 이번 주 GPU 시간을 다 썼어요. Kaggle Settings에서 남은 시간을 확인하세요."
    return text


def token_present():
    return bool(os.environ.get("KAGGLE_API_TOKEN")) or TOKEN_FILE.exists() or LEGACY_FILE.exists()


def _import_kaggle_api():
    """The kaggle package imports itself in a cycle; a failed/raced import leaves
    half-loaded modules behind. Drop them and try once more."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        for name in [m for m in sys.modules if m == "kaggle" or m.startswith("kaggle.")]:
            del sys.modules[name]
        import kaggle  # noqa: F401  (package first, then the submodule)
        from kaggle.api.kaggle_api_extended import KaggleApi
    return KaggleApi


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
                KaggleApi = _import_kaggle_api()
                api = KaggleApi()
                api.authenticate()
                self._api = api
            return self._api

    def username(self):
        return self.api().get_config_value("username")

    def check(self):
        if not token_present():
            return {"ok": False, "error": f"토큰 파일이 없어요: {TOKEN_FILE}"}
        try:
            return {"ok": True, "user": self.username()}
        except SystemExit:  # authenticate() calls sys.exit when the token is rejected
            return {"ok": False, "error": "토큰이 거부됐어요. Kaggle에서 새 토큰을 만들어 다시 저장하세요."}
        except Exception as e:
            return {"ok": False, "error": str(e) or e.__class__.__name__}

    def push(self, folder: Path, slug, code, gpu, kernel_sources=(), timeout=None):
        """Push a script kernel and start it. Returns 'user/slug/version'."""
        user = self.username()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "script.py").write_text(code, encoding="utf-8")
        meta = {
            "id": f"{user}/{slug}",
            "title": slug,
            "code_file": "script.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": gpu,
            "enable_tpu": False,
            "enable_internet": True,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": list(kernel_sources),
            "model_sources": [],
        }
        if gpu:
            meta["machine_shape"] = "NvidiaTeslaT4"
        (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        kw = {"timeout": str(timeout)} if timeout else {}
        try:
            resp = self.api().kernels_push(str(folder), **kw)
        except Exception as e:
            raise RuntimeError(explain_http_error(e)) from e
        err = getattr(resp, "error", None)
        if err:
            raise RuntimeError(f"Kaggle push failed: {err}")
        version = getattr(resp, "version_number", None)
        ref = f"{user}/{slug}"
        return f"{ref}/{version}" if version else ref

    def submit(self, batch_id, jobs, kernel_sources=()):
        payload = base64.b64encode(
            json.dumps({"batch_id": batch_id, "model": config.MODEL_ID, "jobs": jobs}).encode()
        ).decode()
        code = config.WORKER_TEMPLATE.read_text(encoding="utf-8").replace(
            'JOBS_PAYLOAD = "__JOBS_PAYLOAD__"', f'JOBS_PAYLOAD = "{payload}"', 1
        )
        return self.push(config.BATCH_DIR / batch_id / "kernel", config.KERNEL_SLUG, code, gpu=True,
                         kernel_sources=kernel_sources, timeout=config.SESSION_TIMEOUT)

    def build_cache(self, part):
        code = (config.CACHE_TEMPLATE.read_text(encoding="utf-8")
                .replace('PART = "__PART__"', f'PART = "{part}"', 1)
                .replace('WAN = "__MODEL__"', f'WAN = "{config.MODEL_ID}"', 1))
        return self.push(config.DATA_DIR / "cache_kernels" / part, f"{config.CACHE_PREFIX}-{part}", code, gpu=False)

    def kernel_ref(self, slug):
        return f"{self.username()}/{slug}"

    def cache_poll(self, part):
        """State of a cache notebook: none | running | done | failed."""
        try:
            return self.poll(self.kernel_ref(f"{config.CACHE_PREFIX}-{part}"))
        except Exception as e:
            text = str(e).lower()
            if "404" in text or "not found" in text:
                return "none", ""
            return "unknown", str(e)[-300:]

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
        return json.loads(f.read_text(encoding="utf-8")) if f else None

    def logs(self, ref):
        try:
            return self.api().kernels_logs(ref)[-3000:]
        except Exception as e:
            return f"(log unavailable: {e})"

    def stream_events(self, ref):
        """Live progress markers from the running kernel's log."""
        for ev in self.api().kernels_logs_stream(ref):
            yield from parse_markers(ev.get("data") or "")


def parse_markers(text):
    for line in text.splitlines():
        k = line.find("LHP {")
        if k < 0:
            continue
        try:
            yield json.loads(line[k + 4:])
        except ValueError:
            continue


class DryRunRunner:
    """Fakes Kaggle with a timeline of progress events and ffmpeg test videos."""

    SETUP_S = 6.0
    STEP_S = 0.25
    DECODE_S = 1.5
    POST_SETUP_S = 4.0
    POST_S = 3.0

    def __init__(self):
        self._started = {}

    def check(self):
        return {"ok": True, "user": "dry-run"}

    def build_cache(self, part):
        self._cache = getattr(self, "_cache", {})
        self._cache[part] = time.time()
        return part

    def kernel_ref(self, slug):
        return slug

    def cache_poll(self, part):
        t = getattr(self, "_cache", {}).get(part)
        if t is None:
            return "none", ""
        return ("done", "") if time.time() - t > 8 else ("running", "dry-run")

    def submit(self, batch_id, jobs, kernel_sources=()):
        """Same event order as kaggle/worker.py."""
        t = time.time()
        ev = []

        def add(event, **kw):
            ev.append({"t": event, "ts": t, **kw})

        add("setup")
        t += self.SETUP_S
        if any(j["speech"] for j in jobs):
            add("phase", stage="speech")
            t += 2.0 * sum(1 for j in jobs if j["speech"])
        ordered = [j for j in jobs if not j["image_b64"]] + [j for j in jobs if j["image_b64"]]
        kind = None
        for j in ordered:
            k = "i2v" if j["image_b64"] else "t2v"
            if k != kind:
                add("load", kind=k)
                t += 1.0
                kind = k
            add("start", id=j["id"], n=j["steps"])
            for i in range(1, j["steps"] + 1):
                t += self.STEP_S
                add("step", id=j["id"], i=i, n=j["steps"])
            t += self.DECODE_S
            add("gen_done", id=j["id"], secs=1)
        for stage in ("lipsync", "sfx"):
            todo = [j for j in ordered if (j["speech"] if stage == "lipsync" else j["sfx"])]
            if not todo:
                continue
            add("phase", stage=f"{stage}_setup")
            t += self.POST_SETUP_S
            for j in todo:
                add("post", id=j["id"], stage=stage)
                t += self.POST_S
                add("post_done", id=j["id"], stage=stage)
        for j in ordered:
            add("done", id=j["id"])
        self._started[batch_id] = (t, jobs, ev)
        return batch_id

    def poll(self, ref):
        end, _, _ = self._started.get(ref, (0, [], []))
        return ("done", "") if time.time() > end + 1 else ("running", "dry-run")

    def stream_events(self, ref):
        _, _, events = self._started[ref]
        for ev in events:
            wait = ev["ts"] - time.time()
            if wait > 0:
                time.sleep(wait)
            yield ev

    def fetch(self, ref, dest: Path):
        _, jobs, _ = self._started[ref]
        out = dest / "outputs"
        out.mkdir(parents=True, exist_ok=True)
        results = {}
        for j in jobs:
            secs = j["num_frames"] / j["fps"]
            path = out / f"{j['id']}.mp4"
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "lavfi", "-i", f"testsrc2=size={j['width']}x{j['height']}:rate={j['fps']}",
            ]
            if j["speech"] or j["sfx"]:  # beep instead of real audio
                cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100", "-c:a", "aac"]
            cmd += ["-t", f"{secs:.2f}", "-pix_fmt", "yuv420p", str(path)]
            try:
                subprocess.run(cmd, check=True)
                results[j["id"]] = {"status": "done", "file": path.name, "seconds": 0}
            except Exception as e:
                results[j["id"]] = {"status": "failed", "error": f"ffmpeg: {e}"}
        return {"batch_id": ref, "results": results}

    def logs(self, ref):
        return "dry-run"


def make_runner():
    if config.DRY_RUN:
        return DryRunRunner()
    try:
        _import_kaggle_api()  # import on the main thread, before worker threads start
    except Exception:
        pass
    return KaggleRunner()


def move_output(src_dir: Path, file_name: str):
    src = _find(src_dir, file_name)
    if not src:
        return None
    dst = config.OUTPUT_DIR / file_name
    shutil.move(str(src), dst)
    return dst.name
