"""local-Higgs model cache builder (runs once per part on a Kaggle CPU session).

Downloads one group of model files into /kaggle/working/hf (a Hugging Face
cache). The video worker attaches this notebook's output as an input and
links it into its own HF cache, so the files are not downloaded again.

Kaggle saves at most 20GB of output per notebook, so the models are split
into parts. If a Wan part is still too big, float32 weights become bfloat16.
"""

import json
import os
import shutil
import subprocess
import sys
import time

PART = "__PART__"
WAN = "__MODEL__"

OUT = "/kaggle/working"
os.environ["HF_HOME"] = os.path.join(OUT, "hf")
LIMIT = 19.5e9


def log(msg):
    print(f"[local-higgs cache {PART} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)


def du(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            p = os.path.join(root, f)
            if not os.path.islink(p):
                total += os.path.getsize(p)
    return total


def shrink_to_bf16(path):
    import torch
    from safetensors.torch import load_file, save_file

    for root, _, files in os.walk(path):
        for f in files:
            p = os.path.join(root, f)
            if not p.endswith(".safetensors"):
                continue
            try:
                tensors = load_file(p)
            except Exception:
                continue  # not a safetensors file
            if not any(t.dtype == torch.float32 for t in tensors.values()):
                continue
            log(f"float32 -> bfloat16: {f}")
            tensors = {k: (v.to(torch.bfloat16) if v.dtype == torch.float32 else v) for k, v in tensors.items()}
            save_file(tensors, p + ".tmp", metadata={"format": "pt"})
            os.replace(p + ".tmp", p)


def materialize(hub):
    """Replace HF snapshot symlinks with the real files and drop blobs/.
    Kaggle output may not keep symlinks; the HF hub reads plain files fine."""
    for root, _, files in os.walk(hub):
        if f"{os.sep}snapshots" not in root:
            continue
        for f in files:
            p = os.path.join(root, f)
            if os.path.islink(p):
                real = os.path.realpath(p)
                os.unlink(p)
                shutil.move(real, p)
    for name in os.listdir(hub):
        shutil.rmtree(os.path.join(hub, name, "blobs"), ignore_errors=True)


def build():
    pip("-U", "huggingface_hub>=0.30", "safetensors", "requests", "tqdm")
    from huggingface_hub import hf_hub_download, snapshot_download

    if PART == "wan-transformer":
        snapshot_download(WAN, allow_patterns=["transformer/*"])
    elif PART == "wan-text":
        snapshot_download(WAN, allow_patterns=["text_encoder/*"])
    elif PART == "wan-core":
        snapshot_download(WAN, allow_patterns=["model_index.json", "scheduler/*", "tokenizer/*", "vae/*"])
    elif PART == "speech":
        pip("omnivoice")
        import torch
        from omnivoice import OmniVoice

        # loading once pulls every file OmniVoice needs (incl. its audio tokenizer)
        OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cpu", dtype=torch.float32)
        snapshot_download("openai/whisper-large-v3-turbo",
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.tiktoken"])
        hf_hub_download("ByteDance/LatentSync-1.5", "latentsync_unet.pt")
        hf_hub_download("ByteDance/LatentSync-1.5", "whisper/tiny.pt")
    elif PART == "sfx":
        hf_hub_download("hkchengrex/MMAudio", "weights/mmaudio_large_44k_v2.pth")
        snapshot_download("apple/DFN5B-CLIP-ViT-H-14-384", allow_patterns=["open_clip_*", "*.json", "*.txt"])
        snapshot_download("nvidia/bigvgan_v2_44khz_128band_512x", allow_patterns=["config.json", "bigvgan_generator.pt"])
        # MMAudio's own weights (some come from GitHub releases, not the HF cache)
        repo = "/tmp/MMAudio"
        subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/hkchengrex/MMAudio.git", repo], check=True)
        pip("--no-deps", "-e", repo)
        mm = os.path.join(OUT, "mmaudio")
        os.makedirs(mm, exist_ok=True)
        os.chdir(mm)
        from pathlib import Path

        from mmaudio.utils.download_utils import download_model_if_needed

        for p in ("weights/mmaudio_large_44k_v2.pth", "ext_weights/v1-44.pth", "ext_weights/synchformer_state_dict.pth"):
            Path(p).parent.mkdir(parents=True, exist_ok=True)
            download_model_if_needed(Path(p))
        os.chdir(OUT)
    else:
        raise SystemExit(f"unknown part {PART}")

    hub = os.path.join(OUT, "hf", "hub")
    shutil.rmtree(os.path.join(hub, ".locks"), ignore_errors=True)
    materialize(hub)
    size = du(OUT)
    log(f"size {size / 1e9:.2f} GB")
    if size > LIMIT and PART.startswith("wan"):
        shrink_to_bf16(os.path.join(OUT, "hf"))
        size = du(OUT)
        log(f"after bf16: {size / 1e9:.2f} GB")
    if size > LIMIT:
        raise SystemExit(f"part {PART} is {size / 1e9:.1f} GB, over Kaggle's 20GB output limit")

    with open(os.path.join(OUT, "LH_CACHE.json"), "w") as f:
        json.dump({"part": PART, "model": WAN, "bytes": size, "ts": time.time()}, f)
    log("ready")


if __name__ == "__main__":
    build()
