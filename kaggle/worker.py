"""local-Higgs Kaggle worker.

This file is a template. The backend replaces the JOBS_PAYLOAD line below with
a base64-encoded JSON batch, then pushes it to Kaggle as a GPU script kernel.

Output (in /kaggle/working):
  outputs/<job_id>.mp4   generated videos
  result.json            per-job status + the batch id it belongs to
"""

import base64
import gc
import io
import json
import os
import subprocess
import sys
import time
import traceback

JOBS_PAYLOAD = "__JOBS_PAYLOAD__"

WORK = "/kaggle/working"
OUT_DIR = os.path.join(WORK, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def log(msg):
    print(f"[local-higgs {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def emit(kind, **kw):
    """Progress marker. The app reads these from the live Kaggle log."""
    print("LHP " + json.dumps({"t": kind, "ts": time.time(), **kw}), flush=True)


def install_deps():
    pkgs = [
        "diffusers>=0.35.0",
        "transformers>=4.51.0",
        "accelerate>=1.6.0",
        "ftfy",
        "imageio",
        "imageio-ffmpeg",
    ]
    log("installing: " + " ".join(pkgs))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *pkgs], check=True)


def load_batch():
    data = json.loads(base64.b64decode(JOBS_PAYLOAD).decode("utf-8"))
    return data["batch_id"], data["model"], data["jobs"]


def write_result(batch_id, results):
    with open(os.path.join(WORK, "result.json"), "w") as f:
        json.dump({"batch_id": batch_id, "results": results}, f, ensure_ascii=False, indent=2)


def prepare_image(b64, width, height):
    from PIL import Image

    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    # center-crop to target aspect, then resize
    src_ratio = img.width / img.height
    dst_ratio = width / height
    if src_ratio > dst_ratio:
        new_w = int(img.height * dst_ratio)
        left = (img.width - new_w) // 2
        img = img.crop((left, 0, left + new_w, img.height))
    else:
        new_h = int(img.width / dst_ratio)
        top = (img.height - new_h) // 2
        img = img.crop((0, top, img.width, top + new_h))
    return img.resize((width, height), Image.LANCZOS)


def load_pipeline(kind, model_id, dtype):
    import torch
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline

    log(f"loading {kind} pipeline: {model_id}")
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    if kind == "i2v":
        pipe = WanImageToVideoPipeline.from_pretrained(
            model_id, vae=vae, torch_dtype=dtype, expand_timesteps=True
        )
    else:
        pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=dtype)

    # Kaggle T4 = 16GB VRAM, so stream model parts between CPU and GPU.
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)  # progress goes through emit()
    try:
        pipe.vae.enable_tiling()
    except Exception as e:  # older diffusers
        log(f"vae tiling unavailable: {e}")
    return pipe


def run_job(pipe, job, kind):
    import torch
    from diffusers.utils import export_to_video

    gen = torch.Generator(device="cpu").manual_seed(int(job["seed"]))
    n = job["steps"]

    def on_step(_pipe, step, _timestep, cb_kwargs):
        emit("step", id=job["id"], i=step + 1, n=n)
        return cb_kwargs

    emit("start", id=job["id"], n=n)
    kwargs = dict(
        callback_on_step_end=on_step,
        prompt=job["prompt"],
        negative_prompt=job["negative_prompt"],
        height=job["height"],
        width=job["width"],
        num_frames=job["num_frames"],
        num_inference_steps=job["steps"],
        guidance_scale=job["guidance"],
        generator=gen,
    )
    if kind == "i2v":
        kwargs["image"] = prepare_image(job["image_b64"], job["width"], job["height"])

    t0 = time.time()
    frames = pipe(**kwargs).frames[0]
    path = os.path.join(OUT_DIR, f"{job['id']}.mp4")
    export_to_video(frames, path, fps=job["fps"])
    return path, time.time() - t0


def main():
    batch_id, model_id, jobs = load_batch()
    emit("setup", batch=batch_id)
    results = {j["id"]: {"status": "pending"} for j in jobs}
    write_result(batch_id, results)
    log(f"batch {batch_id}: {len(jobs)} job(s)")

    try:
        install_deps()
        import torch

        log(f"GPUs: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}")
        dtype = torch.bfloat16
    except Exception:
        err = traceback.format_exc()
        for j in jobs:
            results[j["id"]] = {"status": "failed", "error": "setup failed:\n" + err[-1500:]}
        write_result(batch_id, results)
        raise

    for kind in ("t2v", "i2v"):
        group = [j for j in jobs if (j.get("image_b64") is not None) == (kind == "i2v")]
        if not group:
            continue
        try:
            emit("load", kind=kind)
            pipe = load_pipeline(kind, model_id, dtype)
        except Exception:
            err = traceback.format_exc()
            log(err)
            for j in group:
                emit("fail", id=j["id"])
                results[j["id"]] = {"status": "failed", "error": "model load failed:\n" + err[-1500:]}
            write_result(batch_id, results)
            continue

        for j in group:
            log(f"job {j['id']} ({kind}) {j['width']}x{j['height']} {j['num_frames']}f")
            try:
                path, secs = run_job(pipe, j, kind)
                results[j["id"]] = {"status": "done", "file": os.path.basename(path), "seconds": round(secs, 1)}
                emit("done", id=j["id"], secs=round(secs, 1))
                log(f"done in {secs:.0f}s")
            except Exception:
                err = traceback.format_exc()
                log(err)
                emit("fail", id=j["id"])
                results[j["id"]] = {"status": "failed", "error": err[-1500:]}
            write_result(batch_id, results)
            torch.cuda.empty_cache()

        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    log("batch finished")


if __name__ == "__main__":
    main()
