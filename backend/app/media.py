"""Keep inputs small: Kaggle rejects a kernel script over 1MB, and images and
voice samples travel inside the script."""

import io
import shutil
import subprocess
import tempfile
from pathlib import Path

IMAGE_MAX_BYTES = 250_000
VOICE_MAX_BYTES = 200_000
# base64 budget for all inputs of one batch (script limit 1MB, the code itself ~30KB)
BATCH_BUDGET = 850_000


def shrink_image(data: bytes, width: int, height: int) -> bytes:
    """Center-crop to the video's aspect, resize to the video size, JPEG."""
    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    src, dst = img.width / img.height, width / height
    if src > dst:
        w = int(img.height * dst)
        img = img.crop(((img.width - w) // 2, 0, (img.width - w) // 2 + w, img.height))
    else:
        h = int(img.width / dst)
        img = img.crop((0, (img.height - h) // 2, img.width, (img.height - h) // 2 + h))
    img = img.resize((width, height), Image.LANCZOS)
    for q in (90, 80, 70, 60):
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=q, optimize=True)
        if buf.tell() <= IMAGE_MAX_BYTES:
            break
    return buf.getvalue()


def shrink_voice(data: bytes, suffix: str) -> bytes:
    """First 10s, mono, small mp3. Needs ffmpeg; without it, small files pass as-is."""
    if len(data) <= VOICE_MAX_BYTES:
        return data
    ff = shutil.which("ffmpeg")
    if not ff:
        raise ValueError("음성 파일이 너무 커요. 10초 이하 m4a/mp3로 올려주세요 (200KB 이하).")
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / f"in{suffix}", Path(d) / "out.mp3"
        src.write_bytes(data)
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(src), "-t", "10", "-ac", "1",
                        "-ar", "22050", "-b:a", "48k", str(dst)], check=True)
        return dst.read_bytes()
