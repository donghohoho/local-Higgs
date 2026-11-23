#!/usr/bin/env bash
# 맥 로컬 엔진 설치 (Apple Silicon 맥 전용, 한 번만): ./setup_local.sh
# FastVideo MLX + FastMetal Wan 2.2 5B (약 19.5GB 다운로드)
set -e
cd "$(dirname "$0")"

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "✖ Apple Silicon(M1~M5) 맥에서만 쓸 수 있어요."; exit 1
fi
PY=$(command -v python3.12 || true)
[ -z "$PY" ] && { echo "✖ python3.12가 없어요: brew install python@3.12"; exit 1; }
command -v ffmpeg >/dev/null || { echo "▶ ffmpeg 설치 중…"; brew install ffmpeg; }

FREE_GB=$(df -g . | awk 'NR==2 {print $4}')
if [ ! -d local/models/FastMetal-5B-QAD ] && [ "$FREE_GB" -lt 25 ]; then
  echo "✖ 저장공간이 25GB 이상 필요해요 (지금 ${FREE_GB}GB 남음)."; exit 1
fi

mkdir -p local/models
if [ ! -d local/FastVideo ]; then
  echo "▶ FastVideo 받는 중…"
  git clone -q --depth 1 https://github.com/hao-ai-lab/FastVideo.git local/FastVideo
fi
if [ ! -d local/.venv ]; then
  echo "▶ 로컬 엔진용 Python 환경 만드는 중…"
  "$PY" -m venv local/.venv
fi
echo "▶ 패키지 설치 중… (몇 분 걸려요)"
local/.venv/bin/pip install -q --upgrade pip
local/.venv/bin/pip install -q -e "local/FastVideo[mlx]"
local/.venv/bin/pip install -q -U "huggingface_hub[cli]>=0.34"

echo "▶ FastMetal Wan 2.2 5B 받는 중… (약 19.5GB, 인터넷 속도에 따라 10~40분)"
local/.venv/bin/hf download FastVideo/FastMetal-5B-QAD --local-dir local/models/FastMetal-5B-QAD

touch local/.ready
echo "✔ 설치 완료. ./start.sh 로 켜고, 화면에서 '맥 로컬'을 고르세요."
