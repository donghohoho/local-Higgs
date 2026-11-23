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
from .progress import tracker
from . import media
from .cache import Cache
from .runner import make_runner, move_output

runner = make_runner()
cache = Cache(runner)
state = {"last_error": None, "kaggle": None, "batch_ref": None, "batch_detail": None}
_wake = threading.Event()


def wake():
    _wake.set()


def _payload(job):
    image_b64 = None
    if job["image_path"]:
        data = Path(job["image_path"]).read_bytes()
        if len(data) > media.IMAGE_MAX_BYTES:  # also covers images uploaded before this check
            data = media.shrink_image(data, job["width"], job["height"])
        image_b64 = base64.b64encode(data).decode()
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
        "speech": _speech(job),
        "sfx": {"prompt": job["sfx_prompt"] or ""} if job["audio_mode"] == "sfx" else None,
    }


def _speech(job):
    if not job["speech_text"]:
        return None
    ref = None
    if job["voice"] == "clone" and job["voice_ref_path"]:
        p = Path(job["voice_ref_path"])
        ref = base64.b64encode(media.shrink_voice(p.read_bytes(), p.suffix)).decode()
    return {"text": job["speech_text"], "voice": job["voice"] or "female", "ref_b64": ref}


def _start_batch():
    queued = db.jobs_by_status("queued", limit=config.MAX_BATCH)
    if not queued:
        return
    # wait a little so several clicks end up in one GPU session
    if time.time() - queued[-1]["created_at"] < config.BATCH_WAIT_SECONDS and len(queued) < config.MAX_BATCH:
        return

    # build inputs; keep the batch under Kaggle's 1MB script limit
    payloads, picked, used, seen = [], [], 0, set()
    for j in queued:
        try:
            pl = _payload(j)
        except Exception as e:
            db.update_job(j["id"], status="failed", error=f"입력 파일 처리 실패: {e}")
            continue
        size = 0
        for blob in (pl["image_b64"], (pl["speech"] or {}).get("ref_b64")):
            if blob and blob not in seen:
                size += len(blob)
        seen.update(b for b in (pl["image_b64"], (pl["speech"] or {}).get("ref_b64")) if b)
        if picked and used + size > media.BATCH_BUDGET:
            break  # goes in the next batch
        if size > media.BATCH_BUDGET:
            db.update_job(j["id"], status="failed", error="입력 파일이 너무 커요 (이미지/음성을 줄여주세요)")
            continue
        payloads.append(pl)
        picked.append(j)
        used += size
    queued = picked
    if not queued:
        return

    batch_id = db.new_id()
    db.insert_batch(batch_id)
    for j in queued:
        db.update_job(j["id"], status="submitted", batch_id=batch_id, error=None)
    try:
        sources = cache.ready_sources()
        ref = runner.submit(batch_id, payloads, kernel_sources=sources)
    except BaseException as e:
        msg = f"제출 실패: {e}"
        db.update_batch(batch_id, "failed", msg)
        for j in queued:
            db.update_job(j["id"], status="failed", error=msg)
        state["last_error"] = msg
        return
    db.update_batch(batch_id, "submitted", ref)
    state["batch_ref"] = ref
    _track(batch_id, ref, time.time(), cached=cache.video_ready())


def _track(batch_id, ref, submit_ts, cached=False):
    tracker.start(batch_id, db.jobs_in_batch(batch_id), submit_ts, lambda: runner.stream_events(ref), cached=cached)


QUEUE_HINT = (
    "Kaggle이 GPU를 {m}분 동안 배정하지 않았어요.\n"
    "· kaggle.com → Settings → Phone Verification (휴대폰 인증) 확인\n"
    "· Settings에서 이번 주 GPU 남은 시간 확인\n"
    "· kaggle.com/code 에서 local-higgs-worker 노트북이 멈춰 있으면 Stop session\n"
    "확인 후 '재시도'를 누르세요."
)


def _fail_batch(batch, msg):
    for j in db.jobs_in_batch(batch["id"]):
        if j["status"] in ("submitted", "running"):
            db.update_job(j["id"], status="failed", error=msg)
    db.update_batch(batch["id"], "failed", batch["message"])
    tracker.finish(batch["id"])
    state["batch_ref"] = None
    state["batch_detail"] = None


def cancel_active():
    b = db.active_batch()
    if not b:
        return False
    _fail_batch(b, "취소됨. Kaggle에서 이미 실행 중이었다면 kaggle.com/code 에서 Stop session 해주세요.")
    return True


def _check_batch(batch):
    ref = batch["message"] or batch["id"]
    status, msg = runner.poll(ref)
    state["batch_detail"] = msg
    tracker.set_remote(batch["id"], msg if status == "running" else status)
    if status == "running" and msg == "queued" and time.time() - batch["created_at"] > config.QUEUE_TIMEOUT:
        _fail_batch(batch, QUEUE_HINT.format(m=config.QUEUE_TIMEOUT // 60))
        return
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
                db.update_job(j["id"], status="done", output_file=name, gen_seconds=r.get("seconds"),
                              error=r.get("warning"))
                continue
            r = {"error": "결과 파일을 찾지 못함"}
        if log_tail is None and not (r and r.get("error")):
            log_tail = runner.logs(ref)
        err = (r or {}).get("error") or f"Kaggle: {msg}\n{log_tail or ''}"
        db.update_job(j["id"], status="failed", error=err[-3000:])

    db.update_batch(batch["id"], "done" if status == "done" else "failed", ref)
    tracker.finish(batch["id"])
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
    b = db.active_batch()
    if b and not config.DRY_RUN:  # app restarted while Kaggle was running
        _track(b["id"], b["message"] or b["id"], b["created_at"])
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
