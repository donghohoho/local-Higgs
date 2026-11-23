# local-Higgs (Windows): start.bat 을 더블클릭하거나, PowerShell에서 .\start.ps1  (-dry: 화면 테스트)
param([switch]$dry)
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location $PSScriptRoot

function Say($msg) { Write-Host "▶ $msg" }
function Fail($msg) { Write-Host "✖ $msg" -ForegroundColor Red; exit 1 }

if ($dry) { $env:LH_DRY_RUN = "1"; Say "DRY RUN 모드 (가짜 영상)" }
$env:PYTHONUTF8 = "1"          # 윈도우 기본 인코딩(cp949) 대신 UTF-8로 파일 읽기
$env:PYTHONIOENCODING = "utf-8"

# 필요한 프로그램 확인
foreach ($cmd in "node", "npm", "git") {
  if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
    Fail "$cmd 이(가) 없어요. README의 '윈도우 준비물'을 먼저 설치하세요."
  }
}
$py = $null
if (Get-Command py -ErrorAction SilentlyContinue) { $py = @("py", "-3") }
elseif (Get-Command python -ErrorAction SilentlyContinue) { $py = @("python") }
else { Fail "Python이 없어요. python.org에서 설치하세요 (설치할 때 'Add python.exe to PATH' 체크)." }

# 이전에 켜둔 local-Higgs가 포트를 잡고 있으면 끄기
foreach ($port in 8000, 3000) {
  $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
  foreach ($c in $conns) {
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$($c.OwningProcess)"
    if ($proc.CommandLine -match "uvicorn|backend\.app\.main|next") {
      Say "이전에 켜둔 앱 종료 (포트 $port)"
      & taskkill /PID $proc.ProcessId /T /F | Out-Null
    } else {
      Fail "포트 $port 를 다른 프로그램($($proc.Name))이 쓰고 있어요. 그 프로그램을 끄고 다시 실행하세요."
    }
  }
}

# 백엔드 (Python 가상환경)
$venvPy = "backend\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
  Say "Python 가상환경 만드는 중…"
  & $py[0] $py[1..9] -m venv backend\.venv
  if ($LASTEXITCODE -ne 0) { Fail "가상환경을 만들지 못했어요." }
}
$reqHash = (Get-FileHash backend\requirements.txt -Algorithm SHA1).Hash
$reqMark = "backend\.venv\.req-$reqHash"
if (-not (Test-Path $reqMark)) {
  Say "백엔드 패키지 설치 중…"
  & $venvPy -m pip install -q --upgrade pip
  & $venvPy -m pip install -q -r backend\requirements.txt
  if ($LASTEXITCODE -ne 0) { Fail "백엔드 패키지 설치 실패" }
  Remove-Item backend\.venv\.req-* -ErrorAction SilentlyContinue
  New-Item $reqMark -ItemType File | Out-Null
}

# 프론트엔드
$lockHash = (Get-FileHash frontend\package-lock.json -Algorithm SHA1).Hash
$lockMark = "frontend\node_modules\.lock-$lockHash"
if (-not (Test-Path $lockMark)) {
  Say "프론트엔드 패키지 설치 중… (처음엔 몇 분 걸려요)"
  Push-Location frontend
  & npm install --no-audit --no-fund
  $code = $LASTEXITCODE
  Pop-Location
  if ($code -ne 0) { Fail "프론트엔드 패키지 설치 실패" }
  Remove-Item frontend\node_modules\.lock-* -ErrorAction SilentlyContinue
  New-Item $lockMark -ItemType File | Out-Null
}

New-Item data -ItemType Directory -Force | Out-Null
$backend = Start-Process -FilePath $venvPy -NoNewWindow -PassThru `
  -ArgumentList "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "8000", "--log-level", "warning"
$frontend = Start-Process -FilePath "cmd.exe" -WorkingDirectory "frontend" -WindowStyle Hidden -PassThru `
  -ArgumentList "/c", "npm run dev -- --hostname 127.0.0.1 > ..\data\frontend.log 2>&1"

function Stop-All {
  foreach ($p in $backend, $frontend) {
    if ($p -and -not $p.HasExited) { & taskkill /PID $p.Id /T /F 2>$null | Out-Null }
  }
}

try {
  Say "켜는 중…"
  $ok = $false
  for ($i = 0; $i -lt 90; $i++) {
    Start-Sleep -Seconds 1
    try {
      Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://127.0.0.1:8000/api/options | Out-Null
      Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 http://127.0.0.1:3000 | Out-Null
      $ok = $true; break
    } catch { }
    if ($backend.HasExited) { Fail "백엔드가 꺼졌어요. 위의 오류 메시지를 확인하세요." }
  }
  if (-not $ok) { Fail "90초 안에 켜지지 않았어요. data\frontend.log 를 확인하세요." }
  Say "http://localhost:3000 열림. 이 창을 닫거나 Ctrl + C 를 누르면 종료돼요."
  Start-Process "http://localhost:3000"
  while (-not $backend.HasExited) { Start-Sleep -Seconds 2 }
} finally {
  Stop-All
}
