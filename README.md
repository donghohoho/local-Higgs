# local-Higgs

Higgsfield처럼 AI 영상을 만드는 개인용 웹앱이에요.
**맥은 화면만**, 무거운 생성은 **Kaggle 무료 GPU**에서 해요. 맥 저장공간을 거의 안 써요.

## 작동 원리

```
[맥 웹앱] 프롬프트 입력 → 대기열
   ↓ Kaggle API: 작업을 담은 노트북 업로드 + 실행
[Kaggle T4 GPU] 모델 다운로드 → 영상 생성 → 결과 저장
   ↓ Kaggle API: 상태 확인 → 완료 시 mp4 다운로드
[맥 웹앱] 갤러리에 표시
```

- 여러 작업은 **한 번의 GPU 세션**에 모아서 보내요 (기본 최대 4개, 20초 대기).
- 한 번에 하나의 배치만 실행돼요. 그동안 추가한 작업은 다음 배치로 가요.

## 기능

**1단계 — 영상**
- 텍스트 → 영상 / 이미지 → 영상 (Wan 2.2 TI2V-5B, Apache 2.0)
- 카메라 무빙 프리셋 10종
- 480p·720p, 16:9·9:16·1:1, 2·3·5초
- 대기열, 갤러리, 재시도, 설정 재사용
- 진행률(%)과 남은 시간 (Kaggle 실시간 로그 기반, 쓸수록 정확해짐)

**2단계 — 오디오**
- 대사: OmniVoice TTS (Apache 2.0, 한국어 지원) → 여성 / 남성 / 내 목소리(3~10초 샘플)
- 립싱크: LatentSync 1.5 (Apache 2.0, T4에서 동작하는 버전)
- 효과음: MMAudio (코드 MIT, **가중치 CC-BY-NC 4.0 → 비상업용**)
- 대사가 있으면 영상 길이가 대사에 맞춰 자동으로 정해져요 (최대 5초)
- 립싱크나 효과음이 실패해도 영상은 나와요. 카드의 "경고 보기"에 이유가 떠요.

> LTX-2.5(영상+소리 동시 생성)는 22B 모델 + 12B 텍스트 인코더라 Kaggle T4(16GB)에 맞지 않아서 쓰지 않았어요.

## 준비물

1. Kaggle 가입 + 휴대폰 인증 (GPU 사용 조건)
2. Kaggle 토큰: kaggle.com/settings/api → **Generate New Token** → 복사 후

   ```bash
   mkdir -p ~/.kaggle
   pbpaste > ~/.kaggle/access_token
   chmod 600 ~/.kaggle/access_token
   ```

3. `brew install node python@3.12 ffmpeg git`

## 실행

```bash
git clone https://github.com/donghohoho/local-higgs.git
cd local-higgs
chmod +x start.sh
./start.sh --dry   # 먼저 화면 테스트 (가짜 영상, Kaggle 안 씀)
./start.sh         # 실제 생성
```

브라우저에서 http://localhost:3000 이 열려요. 종료는 터미널에서 `Ctrl + C`.

## 모델 저장 (다운로드 시간 줄이기)

화면 오른쪽 위 **"모델 저장 0/5"** → **Kaggle에 모델 저장하기**.

- 모델을 Kaggle 노트북 5개에 나눠서 한 번만 받아둬요 (Kaggle 노트북 결과물은 하나당 20GB까지).
- CPU 노트북이라 **GPU 시간을 안 써요**. 10~20분 걸려요.
- 저장된 뒤에는 영상 생성 때 자동으로 연결돼서 매번 받던 약 24GB 다운로드를 건너뛰어요.
- 저장 안 된 모델은 예전처럼 그때그때 받아요.

| 파트 | 용량 (예상) |
|---|---|
| 영상 모델 · VAE·설정 | 약 3GB |
| 영상 모델 · 본체 | 약 10GB |
| 영상 모델 · 텍스트 인코더 | 약 11GB |
| 대사·립싱크 | 약 9GB |
| 효과음 | 약 9GB |

## 소요 시간 (예상)

| 단계 | 시간 |
|---|---|
| Kaggle 세션 시작 + 패키지 설치 | 2~4분 |
| 모델 다운로드 (첫 작업마다) | 3~6분 |
| 480p · 2초 · 표준 | 수 분 |
| 720p · 5초 · 고품질 | 수십 분 |
| 대사 추가 (배치당 1회 설치) | +5~8분, 영상당 +1~2분 |
| 효과음 추가 (배치당 1회 설치) | +3~5분, 영상당 +1분 |

T4에서의 정확한 시간은 첫 실행 후 갤러리에 표시돼요. 처음엔 **480p · 2초 · 초안**으로 테스트하세요.

## 문제 해결

- **Kaggle 인증 실패**: `~/.kaggle/access_token` 확인 → 앱 재시작.
- **생성 실패**: 카드의 "오류 보기"를 확인하세요. kaggle.com/code 에서 `local-higgs-worker` 노트북 로그도 볼 수 있어요.
- **GPU 한도**: Kaggle 설정 화면에서 주간 남은 시간을 확인하세요.

## 구조

```
backend/   FastAPI · SQLite 대기열 · Kaggle 연동
frontend/  Next.js 화면
kaggle/    Kaggle에서 실행되는 생성 스크립트 (worker.py), 모델 저장 스크립트 (cache_builder.py)
data/      결과 영상·DB (git 제외)
```

## 다음 단계

- 3단계: 캐릭터 일관성 (Wan-Animate-2, 캐릭터 LoRA)
- 4단계: 3D (TRELLIS.2 + Blender 카메라 경로)
