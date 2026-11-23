"""local-Higgs Kaggle worker.

This file is a template. The backend replaces the JOBS_PAYLOAD line below with
a base64-encoded JSON batch, then pushes it to Kaggle as a GPU script kernel.

Stages (each only if some job needs it):
  1. speech   OmniVoice TTS -> wav; video length follows the speech
  2. video    Wan 2.2 TI2V-5B (text->video / image->video)
  3. lipsync  LatentSync 1.5 (mouth follows the speech)
  4. sfx      MMAudio (sound effects that match the video)
  5. mix      ffmpeg -> outputs/<job_id>.mp4

Output (in /kaggle/working):
  outputs/<job_id>.mp4   final videos
  result.json            per-job status + the batch id it belongs to
"""

import base64
import gc
import io
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback

JOBS_PAYLOAD = "__JOBS_PAYLOAD__"

# Hugging Face cache on local disk; saved model parts get linked in (attach_caches)
os.environ.setdefault("HF_HOME", "/tmp/hf")
MMAUDIO_CACHE = None

WORK = "/kaggle/working"
OUT_DIR = os.path.join(WORK, "outputs")
TMP = "/tmp/lh"  # not saved as Kaggle output
for d in (OUT_DIR, TMP):
    os.makedirs(d, exist_ok=True)

MAX_SPEECH_S = 5.0
LATENTSYNC_REPO = "https://github.com/bytedance/LatentSync.git"
LATENTSYNC_HF = "ByteDance/LatentSync-1.5"
MMAUDIO_REPO = "https://github.com/hkchengrex/MMAudio.git"
FFMPEG = "ffmpeg"


def log(msg):
    print(f"[local-higgs {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def emit(event, **kw):
    """Progress marker. The app reads these from the live Kaggle log."""
    print("LHP " + json.dumps({"t": event, "ts": time.time(), **kw}), flush=True)


def sh(cmd, cwd=None, tail=1500):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd[:4]))} failed:\n{(r.stdout + r.stderr)[-tail:]}")
    return r.stdout


def pip(*args, python=sys.executable):
    sh([python, "-m", "pip", "install", "-q", *args])


def install_deps(need_speech):
    pkgs = [
        "diffusers>=0.35.0",
        "transformers>=4.51.0",
        "accelerate>=1.6.0",
        "ftfy",
        "imageio",
        "imageio-ffmpeg",
        "soundfile",
    ]
    if need_speech:
        pkgs.append("omnivoice")
    log("installing: " + " ".join(pkgs))
    pip("-U", *pkgs)

    global FFMPEG
    if shutil.which("ffmpeg"):
        FFMPEG = shutil.which("ffmpeg")
    else:
        import imageio_ffmpeg

        FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
        os.makedirs("/usr/local/bin", exist_ok=True)
        try:
            os.symlink(FFMPEG, "/usr/local/bin/ffmpeg")  # LatentSync calls plain `ffmpeg`
        except OSError:
            pass


def _link_tree(src, dst):
    """Mirror src into dst with symlinks. Small files under refs/ are copied so
    the HF hub can update them (the Kaggle input folder is read-only)."""
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        target = os.path.join(dst, rel)
        os.makedirs(target, exist_ok=True)
        for f in files:
            s_path, d_path = os.path.join(root, f), os.path.join(target, f)
            if os.path.lexists(d_path):
                continue
            if f"{os.sep}refs" in root or rel.startswith("refs"):
                shutil.copy(s_path, d_path)
            elif os.path.islink(s_path):
                # HF snapshot links are relative (../../blobs/x): keep them relative
                os.symlink(os.readlink(s_path), d_path)
            else:
                os.symlink(s_path, d_path)


def attach_caches(root="/kaggle/input"):
    """Find saved model parts (LH_CACHE.json) and link them into HF_HOME."""
    global MMAUDIO_CACHE
    found = []
    for dirpath, dirs, files in os.walk(root):
        if dirpath.count(os.sep) - root.count(os.sep) > 4:
            dirs[:] = []
            continue
        if "LH_CACHE.json" not in files:
            continue
        dirs[:] = []
        try:
            part = json.load(open(os.path.join(dirpath, "LH_CACHE.json")))["part"]
        except Exception:
            continue
        if os.path.isdir(os.path.join(dirpath, "hf")):
            _link_tree(os.path.join(dirpath, "hf"), os.environ["HF_HOME"])
        if os.path.isdir(os.path.join(dirpath, "mmaudio")):
            MMAUDIO_CACHE = os.path.join(dirpath, "mmaudio")
        found.append(part)
    return found


def load_batch():
    data = json.loads(base64.b64decode(JOBS_PAYLOAD).decode("utf-8"))
    return data["batch_id"], data["model"], data["jobs"]


def write_result(batch_id, results):
    with open(os.path.join(WORK, "result.json"), "w") as f:
        json.dump({"batch_id": batch_id, "results": results}, f, ensure_ascii=False, indent=2)


def frames_for(seconds, fps):
    """Smallest valid Wan frame count (4k+1) that covers `seconds`, 2s..5s."""
    n = math.ceil(seconds * fps)
    n = 4 * math.ceil((n - 1) / 4) + 1
    return max(49, min(121, n))


# ---------------------------------------------------------------- 1. speech
def has_hangul(s):
    return any("가" <= ch <= "힣" for ch in s)


def run_speech(jobs):
    import soundfile as sf
    import torch
    from omnivoice import OmniVoice

    log("loading OmniVoice")
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
    sr = getattr(model, "sampling_rate", None) or 24000
    errors = {}
    for j in jobs:
        sp = j["speech"]
        try:
            kw = {"text": sp["text"]}
            if has_hangul(sp["text"]):
                kw["language"] = "ko"
            if sp.get("ref_b64"):
                raw = os.path.join(TMP, f"{j['id']}_ref.bin")
                ref = os.path.join(TMP, f"{j['id']}_ref.wav")
                with open(raw, "wb") as f:
                    f.write(base64.b64decode(sp["ref_b64"]))
                sh([FFMPEG, "-y", "-loglevel", "error", "-i", raw, "-ac", "1", "-ar", str(sr), "-t", "10", ref])
                kw["ref_audio"] = ref
            elif sp.get("voice") in ("female", "male"):
                kw["instruct"] = f"{sp['voice']}, young adult"
            audio = model.generate(**kw)[0]
            if len(audio) / sr > MAX_SPEECH_S:
                audio = model.generate(**kw, duration=MAX_SPEECH_S)[0]
            path = os.path.join(TMP, f"{j['id']}_speech.wav")
            sf.write(path, audio, sr)
            secs = len(audio) / sr
            j["_speech"] = path
            j["num_frames"] = frames_for(secs + 0.3, j["fps"])
            log(f"speech {j['id']}: {secs:.1f}s -> {j['num_frames']} frames")
        except Exception:
            errors[j["id"]] = "음성 생성 실패:\n" + traceback.format_exc()[-1200:]
            log(errors[j["id"]])
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return errors


# ---------------------------------------------------------------- 2. video
def prepare_image(b64, width, height):
    from PIL import Image

    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
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


def run_video(pipe, job, kind):
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
        num_inference_steps=n,
        guidance_scale=job["guidance"],
        generator=gen,
    )
    if kind == "i2v":
        kwargs["image"] = prepare_image(job["image_b64"], job["width"], job["height"])

    t0 = time.time()
    frames = pipe(**kwargs).frames[0]
    path = os.path.join(TMP, f"{job['id']}_raw.mp4")
    export_to_video(frames, path, fps=job["fps"])
    return path, time.time() - t0


# ---------------------------------------------------------------- 3. lipsync
def setup_latentsync():
    from huggingface_hub import hf_hub_download

    repo = os.path.join(TMP, "LatentSync")
    env = os.path.join(TMP, "ls_env")
    if not os.path.isdir(repo):
        sh(["git", "clone", "-q", "--depth", "1", LATENTSYNC_REPO, repo])
    if not os.path.isdir(env):
        sh([sys.executable, "-m", "venv", "--system-site-packages", env])
    py = os.path.join(env, "bin", "python")
    # keep Kaggle's torch; skip packages inference does not need
    skip = ("torch==", "torchvision==", "gradio", "mediapipe", "--extra-index-url")
    alt = {"decord": "eva-decord"}
    with open(os.path.join(repo, "requirements.txt")) as f:
        reqs = [r.strip() for r in f if r.strip() and not r.startswith("#")]
    for r in reqs:
        if r.startswith(skip):
            continue
        name = r.split("==")[0]
        for spec in (r, name, alt.get(name)):
            if not spec:
                continue
            try:
                pip(spec, python=py)
                break
            except RuntimeError as e:
                log(f"pip {spec} failed, trying fallback: {str(e)[-200:]}")
    ck = os.path.join(repo, "checkpoints")
    for name in ("latentsync_unet.pt", "whisper/tiny.pt"):
        dst = os.path.join(ck, name)
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(hf_hub_download(LATENTSYNC_HF, name), dst)  # cached part or download
    return repo, py


def run_lipsync(repo, py, job, video, speech):
    out = os.path.join(TMP, f"{job['id']}_lips.mp4")
    sh([
        py, "-m", "scripts.inference",
        "--unet_config_path", "configs/unet/stage2.yaml",
        "--inference_ckpt_path", "checkpoints/latentsync_unet.pt",
        "--inference_steps", "20",
        "--guidance_scale", "1.5",
        "--enable_deepcache",
        "--temp_dir", os.path.join(TMP, f"ls_{job['id']}"),
        "--video_path", video,
        "--audio_path", speech,
        "--video_out_path", out,
    ], cwd=repo)
    return out


# ---------------------------------------------------------------- 4. sfx
SFX_RUNNER = r'''
import sys, json, torch, soundfile as sf
from pathlib import Path
from mmaudio.eval_utils import all_model_cfg, generate, load_video
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils

tasks = json.loads(Path(sys.argv[1]).read_text())
cfg = all_model_cfg["large_44k_v2"]
cfg.download_if_needed()
seq = cfg.seq_cfg
dev, dt = "cuda", torch.float32
net = get_my_mmaudio(cfg.model_name).to(dev, dt).eval()
net.load_weights(torch.load(cfg.model_path, map_location=dev, weights_only=True))
fu = FeaturesUtils(tod_vae_ckpt=cfg.vae_path, synchformer_ckpt=cfg.synchformer_ckpt,
                   enable_conditions=True, mode=cfg.mode,
                   bigvgan_vocoder_ckpt=cfg.bigvgan_16k_path, need_vae_encoder=False).to(dev, dt).eval()
for t in tasks:
    try:
        with torch.inference_mode():
            info = load_video(Path(t["video"]), t["duration"])
            seq.duration = info.duration_sec
            net.update_seq_lengths(seq.latent_seq_len, seq.clip_seq_len, seq.sync_seq_len)
            rng = torch.Generator(device=dev); rng.manual_seed(t["seed"])
            fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=25)
            audio = generate(info.clip_frames.unsqueeze(0), info.sync_frames.unsqueeze(0), [t["prompt"]],
                             negative_text=["music, speech, talking"], feature_utils=fu, net=net, fm=fm,
                             rng=rng, cfg_strength=4.5)
        sf.write(t["out"], audio.float().cpu()[0].numpy().T, seq.sampling_rate)
        print("SFX_OK", t["id"], flush=True)
    except Exception as e:
        print("SFX_FAIL", t["id"], repr(e)[-500:], flush=True)
'''


def setup_mmaudio():
    repo = os.path.join(TMP, "MMAudio")
    env = os.path.join(TMP, "mm_env")
    if not os.path.isdir(repo):
        sh(["git", "clone", "-q", "--depth", "1", MMAUDIO_REPO, repo])
    if not os.path.isdir(env):
        sh([sys.executable, "-m", "venv", "--system-site-packages", env])
    py = os.path.join(env, "bin", "python")
    if MMAUDIO_CACHE:  # saved weights: skip the download in cfg.download_if_needed()
        _link_tree(MMAUDIO_CACHE, repo)
    # torchcodec / gradio / tensorboard are not needed for inference
    pip("--no-deps", "-e", repo, python=py)
    pip("torchdiffeq>=0.2.5", "open_clip_torch>=2.29.0", "av>=14.0.1", "timm>=1.0.12", "einops",
        "hydra-core>=1.3.2", "hydra_colorlog", "colorlog", "nitrous-ema", "tensordict>=0.6.1",
        "librosa", "soundfile", "python-dotenv", "gitpython", python=py)
    return repo, py


def run_sfx_all(repo, py, tasks):
    spec = os.path.join(TMP, "sfx_tasks.json")
    with open(spec, "w") as f:
        json.dump(tasks, f)
    with open(os.path.join(TMP, "sfx_runner.py"), "w") as f:
        f.write(SFX_RUNNER)
    proc = subprocess.Popen([py, os.path.join(TMP, "sfx_runner.py"), spec], cwd=repo,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    status, tail = {}, []
    for line in proc.stdout:
        tail = (tail + [line])[-30:]
        if line.startswith("SFX_OK"):
            jid = line.split()[1]
            status[jid] = None
            emit("post_done", id=jid, stage="sfx")
        elif line.startswith("SFX_FAIL"):
            parts = line.split(maxsplit=2)
            status[parts[1]] = parts[2] if len(parts) > 2 else "unknown"
    proc.wait()
    for t in tasks:
        if t["id"] not in status:
            status[t["id"]] = "MMAudio 종료됨:\n" + "".join(tail)[-1000:]
    return status


# ---------------------------------------------------------------- 5. mix
def mux(video, audio, out):
    """Put audio on a video; the video length wins (short audio is padded with silence)."""
    sh([FFMPEG, "-y", "-loglevel", "error", "-i", video, "-i", audio,
        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-af", "apad", "-c:a", "aac", "-shortest", out])


def mix_sfx(video_with_audio, sfx, out):
    """Speech stays loud, sound effects sit underneath."""
    sh([FFMPEG, "-y", "-loglevel", "error", "-i", video_with_audio, "-i", sfx,
        "-filter_complex",
        "[0:a]apad[v];[1:a]volume=0.35[s];[v][s]amix=inputs=2:duration=shortest:dropout_transition=0,volume=2[a]",
        "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-shortest", out])


# ---------------------------------------------------------------- main
def main():
    batch_id, model_id, jobs = load_batch()
    emit("setup", batch=batch_id)
    try:
        parts = attach_caches()
        log(f"saved model parts: {parts or 'none (downloading)'}")
    except Exception:
        log("cache attach failed, downloading instead:\n" + traceback.format_exc()[-800:])
    results = {j["id"]: {"status": "pending"} for j in jobs}
    warnings = {j["id"]: [] for j in jobs}
    write_result(batch_id, results)
    log(f"batch {batch_id}: {len(jobs)} job(s)")

    def fail(j, err):
        results[j["id"]] = {"status": "failed", "error": err[-1500:]}
        emit("fail", id=j["id"])
        write_result(batch_id, results)

    speech_jobs = [j for j in jobs if j.get("speech")]
    try:
        install_deps(bool(speech_jobs))
        import torch

        log(f"GPUs: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}")
        dtype = torch.bfloat16
    except Exception:
        err = "setup failed:\n" + traceback.format_exc()
        for j in jobs:
            fail(j, err)
        raise

    # 1. speech first: it decides the video length
    if speech_jobs:
        emit("phase", stage="speech")
        for jid, err in run_speech(speech_jobs).items():
            warnings[jid].append(err)

    # 2. video
    raw = {}
    for kind in ("t2v", "i2v"):
        group = [j for j in jobs if (j.get("image_b64") is not None) == (kind == "i2v")]
        if not group:
            continue
        try:
            emit("load", kind=kind)
            pipe = load_pipeline(kind, model_id, dtype)
        except Exception:
            err = "model load failed:\n" + traceback.format_exc()
            log(err)
            for j in group:
                fail(j, err)
            continue
        for j in group:
            log(f"job {j['id']} ({kind}) {j['width']}x{j['height']} {j['num_frames']}f")
            try:
                path, secs = run_video(pipe, j, kind)
                raw[j["id"]] = path
                j["_gen_seconds"] = round(secs, 1)
                emit("gen_done", id=j["id"], secs=round(secs, 1))
                log(f"video done in {secs:.0f}s")
            except Exception:
                err = traceback.format_exc()
                log(err)
                fail(j, err)
            torch.cuda.empty_cache()
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    current = dict(raw)  # job id -> latest video file (may carry audio)

    # 3. lipsync
    lips = [j for j in jobs if j["id"] in raw and j.get("_speech")]
    if lips:
        emit("phase", stage="lipsync_setup")
        ls_err = None
        try:
            repo, py = setup_latentsync()
        except Exception:
            repo = None
            ls_err = "립싱크 설치 실패:\n" + traceback.format_exc()[-1000:]
            log(ls_err)
        for j in lips:
            emit("post", id=j["id"], stage="lipsync")
            try:
                if not repo:
                    raise RuntimeError(ls_err or "립싱크 설치 실패")
                current[j["id"]] = run_lipsync(repo, py, j, raw[j["id"]], j["_speech"])
            except Exception as e:
                warnings[j["id"]].append(f"립싱크 실패 (음성만 넣음): {str(e)[-800:]}")
                out = os.path.join(TMP, f"{j['id']}_voice.mp4")
                try:
                    mux(raw[j["id"]], j["_speech"], out)
                    current[j["id"]] = out
                except Exception as e2:
                    warnings[j["id"]].append(f"음성 합치기 실패: {e2}")
            emit("post_done", id=j["id"], stage="lipsync")

    # 4. sfx
    sfx_jobs = [j for j in jobs if j["id"] in raw and j.get("sfx")]
    if sfx_jobs:
        emit("phase", stage="sfx_setup")
        tasks = [{
            "id": j["id"],
            "video": raw[j["id"]],
            "duration": j["num_frames"] / j["fps"],
            "prompt": j["sfx"].get("prompt") or j["prompt"],
            "seed": int(j["seed"]) % (2**31),
            "out": os.path.join(TMP, f"{j['id']}_sfx.wav"),
        } for j in sfx_jobs]
        try:
            repo, py = setup_mmaudio()
            for t in tasks:
                emit("post", id=t["id"], stage="sfx")
            status = run_sfx_all(repo, py, tasks)
        except Exception:
            err = "효과음 설치 실패:\n" + traceback.format_exc()[-1000:]
            log(err)
            status = {t["id"]: err for t in tasks}
        for t in tasks:
            jid = t["id"]
            if status.get(jid):
                warnings[jid].append(f"효과음 실패: {status[jid]}")
                continue
            out = os.path.join(TMP, f"{jid}_final.mp4")
            try:
                if current[jid] == raw[jid]:
                    mux(raw[jid], t["out"], out)
                else:
                    mix_sfx(current[jid], t["out"], out)
                current[jid] = out
            except Exception as e:
                warnings[jid].append(f"효과음 합치기 실패: {e}")

    # 5. deliver
    for j in jobs:
        jid = j["id"]
        if jid not in current:
            continue
        dst = os.path.join(OUT_DIR, f"{jid}.mp4")
        shutil.copy(current[jid], dst)
        results[jid] = {"status": "done", "file": os.path.basename(dst), "seconds": j.get("_gen_seconds")}
        if warnings[jid]:
            results[jid]["warning"] = "\n".join(warnings[jid])[-2000:]
        emit("done", id=jid)
        write_result(batch_id, results)

    log("batch finished")


if __name__ == "__main__":
    main()
