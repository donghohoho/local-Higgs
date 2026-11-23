import random
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from . import config, db, local_engine, media, scheduler
from .presets import DURATIONS, QUALITY_STEPS, SIZES, build_prompt, public_options
from .progress import tracker


@asynccontextmanager
async def lifespan(app):
    db.init()
    scheduler.start()
    local_engine.start()
    yield


app = FastAPI(title="local-Higgs", lifespan=lifespan)
app.mount("/outputs", StaticFiles(directory=config.OUTPUT_DIR), name="outputs")


def _public(job):
    job = dict(job)
    job["has_image"] = bool(job.pop("image_path", None))
    job.pop("voice_ref_path", None)
    # a finished job may still carry a warning (e.g. lipsync failed, voice only)
    job["warning"] = job.pop("error") if job["status"] == "done" else None
    job["error"] = job.get("error") if job["status"] != "done" else None
    job["video_url"] = f"/outputs/{job['output_file']}" if job.get("output_file") else None
    return job


@app.get("/api/options")
def options():
    return public_options()


@app.get("/api/status")
def status():
    b = db.active_batch()
    return {
        "dry_run": config.DRY_RUN,
        "model": config.MODEL_ID,
        "kaggle": scheduler.state["kaggle"],
        "active_batch": b,
        "batch_detail": scheduler.state["batch_detail"],
        "queued": len(db.jobs_by_status("queued")),
        "local": local_engine.status(),
        "last_error": scheduler.state["last_error"],
    }


@app.post("/api/batch/cancel")
def cancel_batch():
    if not scheduler.cancel_active():
        raise HTTPException(400, "실행 중인 작업이 없어요")
    scheduler.wake()
    return {"ok": True}


@app.get("/api/cache")
def cache_status():
    return {"parts": scheduler.cache.public(), "video_ready": scheduler.cache.video_ready()}


@app.post("/api/cache/build")
def cache_build():
    if not scheduler.state["kaggle"] or not scheduler.state["kaggle"].get("ok"):
        raise HTTPException(400, "Kaggle에 먼저 연결해주세요")
    try:
        started = scheduler.cache.build()
    except Exception as e:
        raise HTTPException(500, f"저장 시작 실패: {e}")
    return {"started": started}


@app.get("/api/jobs")
def jobs():
    rows = db.list_jobs()
    prog = {}
    for bid in {j["batch_id"] for j in rows if j["status"] in ("submitted", "running") and j["engine"] == "kaggle"}:
        prog.update(tracker.progress(bid, db.jobs_in_batch(bid)))
    for j in rows:
        if j["engine"] == "local" and j["status"] in ("queued", "running"):
            prog[j["id"]] = local_engine.progress(j)
    out = []
    for j in rows:
        pj = _public(j)
        active = j["status"] in ("submitted", "running") or (j["engine"] == "local" and j["status"] == "queued")
        pj["progress"] = prog.get(j["id"]) if active else None
        out.append(pj)
    return out


@app.post("/api/jobs")
async def create_job(
    prompt: str = Form(...),
    preset: str = Form("none"),
    resolution: str = Form("480p"),
    aspect: str = Form("16:9"),
    duration: str = Form("2"),
    quality: str = Form("standard"),
    seed: int = Form(-1),
    count: int = Form(1),
    engine: str = Form("kaggle"),
    audio_mode: str = Form("none"),
    sfx_prompt: str = Form(""),
    speech_text: str = Form(""),
    voice: str = Form("female"),
    image: UploadFile | None = File(None),
    voice_ref: UploadFile | None = File(None),
):
    if not prompt.strip():
        raise HTTPException(400, "프롬프트를 입력해주세요")
    if resolution not in SIZES or aspect not in SIZES[resolution]:
        raise HTTPException(400, "잘못된 해상도/비율")
    if duration not in DURATIONS or quality not in QUALITY_STEPS:
        raise HTTPException(400, "잘못된 길이/품질")
    count = max(1, min(count, config.MAX_BATCH))
    if engine not in ("kaggle", "local"):
        raise HTTPException(400, "잘못된 엔진")
    if engine == "local":
        st = local_engine.status()
        if not st["available"]:
            raise HTTPException(400, f"맥 로컬을 쓸 수 없어요: {st['reason']}")
        if (image is not None and image.filename) or speech_text.strip() or audio_mode != "none":
            raise HTTPException(400, "맥 로컬은 텍스트 → 영상만 돼요. 이미지·오디오는 Kaggle을 고르세요.")

    image_path = None
    if image is not None and image.filename:
        data = await image.read()
        if len(data) > config.MAX_IMAGE_BYTES:
            raise HTTPException(400, "이미지는 20MB 이하만 가능해요")
        ext = Path(image.filename).suffix.lower() or ".png"
        if ext not in (".png", ".jpg", ".jpeg", ".webp"):
            raise HTTPException(400, "png/jpg/webp만 가능해요")
        if resolution not in SIZES or aspect not in SIZES[resolution]:
            raise HTTPException(400, "잘못된 해상도/비율")
        try:  # resize now: Kaggle takes at most 1MB per job script
            data = media.shrink_image(data, *SIZES[resolution][aspect])
        except Exception:
            raise HTTPException(400, "이미지를 읽지 못했어요")
        p = config.UPLOAD_DIR / f"{db.new_id()}.jpg"
        p.write_bytes(data)
        image_path = str(p)

    if audio_mode not in ("none", "sfx"):
        raise HTTPException(400, "잘못된 오디오 옵션")
    speech_text = speech_text.strip()
    if len(speech_text) > 120:
        raise HTTPException(400, "대사는 120자 이하로 써주세요 (최대 5초)")
    if voice not in ("female", "male", "clone"):
        raise HTTPException(400, "잘못된 목소리 옵션")
    voice_ref_path = None
    if speech_text and voice == "clone":
        if voice_ref is None or not voice_ref.filename:
            raise HTTPException(400, "내 목소리를 쓰려면 3~10초 음성 파일을 올려주세요")
        data = await voice_ref.read()
        if len(data) > 5 * 1024 * 1024:
            raise HTTPException(400, "음성 파일은 5MB 이하만 가능해요")
        ext = Path(voice_ref.filename).suffix.lower() or ".wav"
        if ext not in (".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".webm"):
            raise HTTPException(400, "음성은 wav/mp3/m4a/aac/ogg/flac/webm만 가능해요")
        try:
            small = media.shrink_voice(data, ext)
            if small is not data:  # converted by ffmpeg
                ext = ".mp3"
            data = small
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception:
            raise HTTPException(400, "음성 파일을 읽지 못했어요")
        p = config.UPLOAD_DIR / f"{db.new_id()}{ext}"
        p.write_bytes(data)
        voice_ref_path = str(p)
    if speech_text:
        duration = "auto"  # the worker sets the length from the speech (max 5s)

    w, h = SIZES[resolution][aspect]
    created = []
    for i in range(count):
        s = seed + i if seed >= 0 else random.randint(0, 2**31 - 1)
        job = db.insert_job({
            "id": db.new_id(),
            "status": "queued",
            "prompt": prompt.strip(),
            "final_prompt": build_prompt(prompt, preset),
            "preset": preset,
            "resolution": resolution,
            "aspect": aspect,
            "duration": duration,
            "quality": quality,
            "width": w,
            "height": h,
            "num_frames": DURATIONS.get(duration, DURATIONS["5"]),
            "steps": QUALITY_STEPS[quality],
            "seed": s,
            "image_path": image_path,
            "audio_mode": audio_mode,
            "sfx_prompt": sfx_prompt.strip() or None,
            "speech_text": speech_text or None,
            "voice": voice if speech_text else None,
            "voice_ref_path": voice_ref_path,
            "engine": engine,
        })
        created.append(_public(job))
    scheduler.wake()
    local_engine.wake()
    return created


@app.post("/api/jobs/{job_id}/retry")
def retry(job_id: str):
    j = db.get_job(job_id)
    if not j:
        raise HTTPException(404)
    if j["status"] != "failed":
        raise HTTPException(400, "실패한 작업만 재시도할 수 있어요")
    db.update_job(job_id, status="queued", batch_id=None, error=None)
    scheduler.wake()
    local_engine.wake()
    return {"ok": True}


@app.delete("/api/jobs/{job_id}")
def delete(job_id: str):
    j = db.get_job(job_id)
    if not j:
        raise HTTPException(404)
    if j["status"] in ("submitted", "running"):
        if j["engine"] == "local" and local_engine.cancel():
            return {"ok": True, "cancelled": True}  # the loop marks it failed
        raise HTTPException(400, "실행 중인 작업은 삭제할 수 없어요")
    if j.get("output_file"):
        (config.OUTPUT_DIR / j["output_file"]).unlink(missing_ok=True)
    db.delete_job(job_id)
    return {"ok": True}
