import random
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from . import config, db, scheduler
from .presets import DURATIONS, QUALITY_STEPS, SIZES, build_prompt, public_options


@asynccontextmanager
async def lifespan(app):
    db.init()
    scheduler.start()
    yield


app = FastAPI(title="local-Higgs", lifespan=lifespan)
app.mount("/outputs", StaticFiles(directory=config.OUTPUT_DIR), name="outputs")


def _public(job):
    job = dict(job)
    job["has_image"] = bool(job.pop("image_path", None))
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
        "last_error": scheduler.state["last_error"],
    }


@app.get("/api/jobs")
def jobs():
    return [_public(j) for j in db.list_jobs()]


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
    image: UploadFile | None = File(None),
):
    if not prompt.strip():
        raise HTTPException(400, "프롬프트를 입력해주세요")
    if resolution not in SIZES or aspect not in SIZES[resolution]:
        raise HTTPException(400, "잘못된 해상도/비율")
    if duration not in DURATIONS or quality not in QUALITY_STEPS:
        raise HTTPException(400, "잘못된 길이/품질")
    count = max(1, min(count, config.MAX_BATCH))

    image_path = None
    if image is not None and image.filename:
        data = await image.read()
        if len(data) > config.MAX_IMAGE_BYTES:
            raise HTTPException(400, "이미지는 6MB 이하만 가능해요")
        ext = Path(image.filename).suffix.lower() or ".png"
        if ext not in (".png", ".jpg", ".jpeg", ".webp"):
            raise HTTPException(400, "png/jpg/webp만 가능해요")
        p = config.UPLOAD_DIR / f"{db.new_id()}{ext}"
        p.write_bytes(data)
        image_path = str(p)

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
            "num_frames": DURATIONS[duration],
            "steps": QUALITY_STEPS[quality],
            "seed": s,
            "image_path": image_path,
        })
        created.append(_public(job))
    scheduler.wake()
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
    return {"ok": True}


@app.delete("/api/jobs/{job_id}")
def delete(job_id: str):
    j = db.get_job(job_id)
    if not j:
        raise HTTPException(404)
    if j["status"] in ("submitted", "running"):
        raise HTTPException(400, "Kaggle에서 실행 중인 작업은 삭제할 수 없어요")
    if j.get("output_file"):
        (config.OUTPUT_DIR / j["output_file"]).unlink(missing_ok=True)
    db.delete_job(job_id)
    return {"ok": True}
