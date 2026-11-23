#!/usr/bin/env bash
# local-Higgs 실행: ./start.sh        (Kaggle GPU 사용)
#                  ./start.sh --dry  (Kaggle 없이 화면 테스트)
set -e
cd "$(dirname "$0")"

if [ "$1" = "--dry" ]; then export LH_DRY_RUN=1; echo "▶ DRY RUN 모드 (가짜 영상)"; fi

# 이전에 켜둔 local-Higgs가 포트를 잡고 있으면 끄기 (다른 프로그램이면 멈추고 안내)
free_port() {
  local port=$1 pid cmd
  for pid in $(lsof -ti tcp:"$port" -sTCP:LISTEN 2>/dev/null); do
    cmd=$(ps -o command= -p "$pid" 2>/dev/null || true)
    if echo "$cmd" | grep -qE "uvicorn backend.app.main|next-server|next dev|next/dist"; then
      echo "▶ 이전에 켜둔 앱 종료 (포트 $port)"
      kill "$pid" 2>/dev/null || true
    else
      echo "✖ 포트 $port 를 다른 프로그램이 쓰고 있어요: $cmd"
      echo "  그 프로그램을 끄고 다시 실행하세요."
      exit 1
    fi
  done
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    lsof -ti tcp:"$port" -sTCP:LISTEN >/dev/null 2>&1 || return 0
    sleep 0.5
  done
}
free_port 8000
free_port 3000

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
LOCK_HASH=$(shasum frontend/package-lock.json | cut -d' ' -f1)
if [ ! -f frontend/node_modules/.lock-$LOCK_HASH ]; then
  echo "▶ 프론트엔드 패키지 설치 중…"
  (cd frontend && npm install --no-audit --no-fund)
  rm -f frontend/node_modules/.lock-*; touch frontend/node_modules/.lock-$LOCK_HASH
fi

trap 'kill 0' EXIT
backend/.venv/bin/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --log-level warning &
(cd frontend && npm run dev -- --hostname 127.0.0.1 >/dev/null) &

echo "▶ 켜는 중…"
for _ in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8000/api/options >/dev/null && curl -sf -o /dev/null http://127.0.0.1:3000; then
    echo "▶ http://localhost:3000 열림 (종료: Ctrl + C)"
    command -v open >/dev/null && open http://localhost:3000
    wait
    exit 0
  fi
  sleep 1
done
echo "✖ 60초 안에 켜지지 않았어요. 위의 오류 메시지를 확인하세요."
exit 1
