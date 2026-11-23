"""Mac local engine: FastMetal Wan 2.2 5B on Apple Silicon (FastVideo MLX).

Runs one job at a time in a subprocess on this Mac. Text -> video only
(FastMetal 5B is T2V); images and audio stay on Kaggle.
Install once with ./setup_local.sh.
"""

import json
import platform
import shutil
import subprocess
import threading
import time
import traceback

from . import config, db

LOCAL = config.ROOT / "local"
PY = LOCAL / ".venv" / "bin" / "python"
REPO = LOCAL / "FastVideo"
MODEL = LOCAL / "models" / "FastMetal-5B-QAD"
SCRIPT = REPO / "examples" / "inference" / "basic" / "mlx_wan22_generate.py"
STATS = config.DATA_DIR / ("local-stats-dry.json" if config.DRY_RUN else "local-stats.json")
DEFAULTS = {"load_s": 60.0, "s_per_unit": 4.0}  # unit = megapixels x frames; MacBook Air guess
QUALITY_FLAGS = {"draft": ["--fast"], "standard": [], "high": ["--refine"]}

_state = {"job": None, "started": None, "proc": None}
_wake = threading.Event()


def status():
    if config.DRY_RUN:
        return {"available": True, "reason": "dry-run"}
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return {"available": False, "reason": "Apple Silicon 맥에서만 쓸 수 있어요"}
    if not (LOCAL / ".ready").exists() or not PY.exists() or not MODEL.exists():
        return {"available": False, "reason": "설치 안 됨: 터미널에서 ./setup_local.sh 실행"}
    return {"available": True, "reason": ""}


def _stats():
    try:
        return {**DEFAULTS, **json.loads(STATS.read_text(encoding="utf-8"))}
    except Exception:
        return dict(DEFAULTS)


def _learn(units, seconds):
    s = _stats()
    per = max(seconds - s["load_s"], 5) / max(units, 0.1)
    s["s_per_unit"] = 0.5 * s["s_per_unit"] + 0.5 * per
    try:
        STATS.write_text(json.dumps(s, indent=2), encoding="utf-8")
    except Exception:
        pass


def _units(job):
    u = job["width"] * job["height"] * job["num_frames"] / 1e6
    return u * {"draft": 0.55, "standard": 1.0, "high": 1.8}.get(job["quality"], 1.0)


def progress(job):
    """{percent, eta_s, stage} for a local job (time-based estimate)."""
    s = _stats()
    est = s["load_s"] + s["s_per_unit"] * _units(job)
    if _state["job"] == job["id"] and _state["started"]:
        el = time.time() - _state["started"]
        pct = int(min(99, 100 * el / max(est, el + 5)))
        return {"percent": pct, "eta_s": int(max(est - el, 5)), "stage": "맥에서 생성 중"}
    # waiting behind the running local job
    ahead = 0.0
    if _state["job"] and _state["started"]:
        cur = db.get_job(_state["job"])
        if cur:
            ahead = max(s["load_s"] + s["s_per_unit"] * _units(cur) - (time.time() - _state["started"]), 5)
    return {"percent": 0, "eta_s": int(ahead + est), "stage": "맥 로컬 대기 중"}


def _run(job):
    out = config.OUTPUT_DIR / f"{job['id']}.mp4"
    if config.DRY_RUN:
        time.sleep(6)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        f"testsrc2=size={job['width']}x{job['height']}:rate=24",
                        "-t", f"{job['num_frames'] / 24:.2f}", "-pix_fmt", "yuv420p", str(out)], check=True)
        return out
    cmd = [
        str(PY), str(SCRIPT),
        "--mlx-checkpoint", str(MODEL),
        "--text-encoder-root", str(MODEL),
        "--vae-root", str(MODEL / "vae"),
        "--prompt", job["final_prompt"],
        "--height", str(job["height"]), "--width", str(job["width"]),
        "--num-frames", str(job["num_frames"]),
        "--seed", str(job["seed"] % (2**31)),
        "--output-path", str(out),
        *QUALITY_FLAGS.get(job["quality"], []),
    ]
    log = config.DATA_DIR / "local.log"
    with open(log, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT)
        _state["proc"] = proc
        code = proc.wait()
    _state["proc"] = None
    if code != 0 or not out.exists():
        tail = log.read_text(encoding="utf-8", errors="replace")[-2500:]
        raise RuntimeError(f"맥 로컬 생성 실패 (exit {code}):\n{tail}")
    return out


def loop():
    while True:
        jobs = db.jobs_by_status("queued", limit=1, engine="local")
        if not jobs or not status()["available"]:
            _wake.wait(3)
            _wake.clear()
            continue
        job = jobs[0]
        _state.update(job=job["id"], started=time.time())
        db.update_job(job["id"], status="running", batch_id="local", error=None)
        try:
            out = _run(job)
            secs = time.time() - _state["started"]
            db.update_job(job["id"], status="done", output_file=out.name, gen_seconds=round(secs, 1))
            _learn(_units(job), secs)
        except Exception as e:
            db.update_job(job["id"], status="failed", error=str(e)[-3000:] or traceback.format_exc()[-2000:])
        finally:
            _state.update(job=None, started=None)


def cancel():
    p = _state.get("proc")
    if p and p.poll() is None:
        p.terminate()
        return True
    return False


def start():
    if shutil.which("ffmpeg") is None and not config.DRY_RUN:
        pass  # FastVideo needs ffmpeg; setup_local.sh installs it
    threading.Thread(target=loop, daemon=True).start()


def wake():
    _wake.set()
