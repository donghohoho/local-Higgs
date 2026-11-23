"""Background loop: queue -> Kaggle batch -> results.

Only one batch runs at a time (one GPU session). Jobs that arrive while a
batch is running wait for the next one.
"""

import base64
import shutil
import threading
import time
import traceback
from pathlib import Path

from . import config, db
from .presets import FPS, NEGATIVE_PROMPT
from .runner import make_runner, move_output

runner = make_runner()
state = {"last_error": None, "kaggle": None, "batch_ref": None, "batch_detail": None}
_wake = threading.Event()


def wake():
    _wake.set()


def _payload(job):
    image_b64 = None
    if job["image_path"]:
        image_b64 = base64.b64encode(Path(job["image_path"]).read_bytes()).decode()
    return {
        "id": job["id"],
        "prompt": job["final_prompt"],
        "negative_prompt": NEGATIVE_PROMPT,
        "width": job["width"],
        "height": job["height"],
        "num_frames": job["num_frames"],
        "fps": FPS,
        "steps": job["steps"],
        "guidance": 5.0,
        "seed": job["seed"],
        "image_b64": image_b64,
    }


def _start_batch():
    queued = db.jobs_by_status("queued", limit=config.MAX_BATCH)
    if not queued:
        return
    # wait a little so several clicks end up in one GPU session
    if time.time() - queued[-1]["created_at"] < config.BATCH_WAIT_SECONDS and len(queued) < config.MAX_BATCH:
        return

    batch_id = db.new_id()
    db.insert_batch(batch_id)
    for j in queued:
        db.update_job(j["id"], status="submitted", batch_id=batch_id, error=None)
    try:
        ref = runner.submit(batch_id, [_payload(j) for j in queued])
    except BaseException as e:
        msg = f"제출 실패: {e}"
        db.update_batch(batch_id, "failed", msg)
        for j in queued:
            db.update_job(j["id"], status="failed", error=msg)
        state["last_error"] = msg
        return
    db.update_batch(batch_id, "submitted", ref)
    state["batch_ref"] = ref


def _check_batch(batch):
    ref = batch["message"] or batch["id"]
    status, msg = runner.poll(ref)
    state["batch_detail"] = msg
    if status == "running":
        if batch["status"] != "running":
            db.update_batch(batch["id"], "running", ref)
            for j in db.jobs_in_batch(batch["id"]):
                if j["status"] == "submitted":
                    db.update_job(j["id"], status="running")
        return

    dest = config.BATCH_DIR / batch["id"] / "output"
    result = None
    try:
        result = runner.fetch(ref, dest)
    except Exception as e:
        msg = f"{msg} / 결과 다운로드 실패: {e}"

    if result and result.get("batch_id") not in (None, batch["id"]):
        # Kaggle still reports a previous version: keep waiting
        return

    results = (result or {}).get("results", {})
    log_tail = None
    for j in db.jobs_in_batch(batch["id"]):
        r = results.get(j["id"])
        if r and r.get("status") == "done":
            name = move_output(dest, r["file"])
            if name:
                db.update_job(j["id"], status="done", output_file=name, gen_seconds=r.get("seconds"))
                continue
            r = {"error": "결과 파일을 찾지 못함"}
        if log_tail is None and not (r and r.get("error")):
            log_tail = runner.logs(ref)
        err = (r or {}).get("error") or f"Kaggle: {msg}\n{log_tail or ''}"
        db.update_job(j["id"], status="failed", error=err[-3000:])

    db.update_batch(batch["id"], "done" if status == "done" else "failed", ref)
    shutil.rmtree(config.BATCH_DIR / batch["id"] / "kernel", ignore_errors=True)
    state["batch_ref"] = None
    state["batch_detail"] = None


def tick():
    batch = db.active_batch()
    if batch:
        _check_batch(batch)
    else:
        _start_batch()


def loop():
    state["kaggle"] = runner.check()
    last_check = time.time()
    while True:
        if not state["kaggle"]["ok"] and time.time() - last_check > 30:
            state["kaggle"] = runner.check()  # token may have been added since startup
            last_check = time.time()
        if not state["kaggle"]["ok"]:
            _wake.wait(5)
            _wake.clear()
            continue
        try:
            tick()
            state["last_error"] = None if db.active_batch() or not state["last_error"] else state["last_error"]
        except Exception:
            state["last_error"] = traceback.format_exc()[-1500:]
        interval = 3 if config.DRY_RUN or not db.active_batch() else config.POLL_SECONDS
        _wake.wait(interval)
        _wake.clear()


def start():
    t = threading.Thread(target=loop, daemon=True)
    t.start()
