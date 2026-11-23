"""Camera presets, sizes and durations.

Wan 2.2 follows camera language in the prompt well, so presets are prompt
suffixes. Phase 2 can swap some of these for a camera-control model.
"""

CAMERA_PRESETS = [
    {"id": "none", "name": "없음", "desc": "프롬프트 그대로", "suffix": ""},
    {"id": "static", "name": "고정 샷", "desc": "카메라 고정",
     "suffix": "static camera, locked-off tripod shot, no camera movement"},
    {"id": "dolly_in", "name": "돌리 인", "desc": "피사체로 천천히 다가감",
     "suffix": "slow dolly in, the camera smoothly pushes forward toward the subject"},
    {"id": "dolly_out", "name": "돌리 아웃", "desc": "뒤로 빠지며 전경 공개",
     "suffix": "slow dolly out, the camera pulls back to reveal the surroundings"},
    {"id": "pan_left", "name": "팬 왼쪽", "desc": "왼쪽으로 회전",
     "suffix": "smooth camera pan to the left"},
    {"id": "pan_right", "name": "팬 오른쪽", "desc": "오른쪽으로 회전",
     "suffix": "smooth camera pan to the right"},
    {"id": "tilt_up", "name": "틸트 업", "desc": "아래에서 위로",
     "suffix": "camera tilts up slowly from the ground to the sky"},
    {"id": "orbit", "name": "오빗", "desc": "피사체 주위를 회전",
     "suffix": "the camera orbits around the subject in a smooth 180-degree arc"},
    {"id": "crane_up", "name": "크레인 업", "desc": "위로 솟아오름",
     "suffix": "crane shot, the camera rises upward revealing a high-angle view"},
    {"id": "fpv", "name": "FPV 드론", "desc": "빠른 드론 비행",
     "suffix": "fast FPV drone shot flying forward, dynamic motion, slight banking"},
    {"id": "handheld", "name": "핸드헬드", "desc": "자연스러운 흔들림",
     "suffix": "handheld camera, subtle natural shake, documentary style"},
]

# width/height must be divisible by 32 for Wan 2.2 TI2V
SIZES = {
    "480p": {"16:9": (832, 480), "9:16": (480, 832), "1:1": (640, 640)},
    "720p": {"16:9": (1280, 704), "9:16": (704, 1280), "1:1": (960, 960)},
}

FPS = 24
# num_frames must be 4k+1
DURATIONS = {"2": 49, "3": 73, "5": 121}

QUALITY_STEPS = {"draft": 20, "standard": 30, "high": 50}

NEGATIVE_PROMPT = (
    "blurry, low quality, worst quality, jpeg artifacts, overexposed, static frame, "
    "subtitles, watermark, text, logo, deformed, disfigured, extra fingers, bad hands, "
    "bad anatomy, mutated limbs, fused fingers, messy background, three legs, "
    "walking backwards, flickering"
)


def get_preset(preset_id):
    for p in CAMERA_PRESETS:
        if p["id"] == preset_id:
            return p
    return CAMERA_PRESETS[0]


def build_prompt(prompt, preset_id):
    suffix = get_preset(preset_id)["suffix"]
    prompt = prompt.strip().rstrip(".")
    return f"{prompt}. {suffix}." if suffix else prompt + "."


def public_options():
    return {
        "camera_presets": [{k: p[k] for k in ("id", "name", "desc")} for p in CAMERA_PRESETS],
        "resolutions": list(SIZES.keys()),
        "aspects": list(SIZES["480p"].keys()),
        "durations": list(DURATIONS.keys()),
        "qualities": list(QUALITY_STEPS.keys()),
    }
