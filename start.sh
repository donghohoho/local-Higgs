#!/usr/bin/env bash
# local-Higgs 실행: ./start.sh        (Kaggle GPU 사용)
#                  ./start.sh --dry  (Kaggle 없이 화면 테스트)
set -e
cd "$(dirname "$0")"

if [ "$1" = "--dry" ]; then export LH_DRY_RUN=1; echo "▶ DRY RUN 모드 (가짜 영상)"; fi

PY=$(command -v python3.12 || command -v python3)
if [ ! -d backend/.venv ]; then
  echo "▶ Python 가상환경 만드는 중…"
  "$PY" -m venv backend/.venv
fi
REQ_HASH=$(shasum backend/requirements.txt | cut -d' ' -f1)
if [ ! -f backend/.venv/.req-$REQ_HASH ]; then
  echo "▶ 백엔드 패키지 설치 중…"
  backend/.venv/bin/pip install -q --upgrade pip
  backend/.venv/bin/pip install -q -r backend/requirements.txt
  rm -f backend/.venv/.req-*; touch backend/.venv/.req-$REQ_HASH
fi
if [ ! -d frontend/node_modules ]; then
  echo "▶ 프론트엔드 패키지 설치 중…"
  (cd frontend && npm install --no-audit --no-fund)
fi

trap 'kill 0' EXIT
backend/.venv/bin/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --log-level warning &
(cd frontend && npm run dev -- --hostname 127.0.0.1 >/dev/null) &
sleep 4
echo "▶ http://localhost:3000 열림 (종료: Ctrl + C)"
command -v open >/dev/null && open http://localhost:3000
wait
