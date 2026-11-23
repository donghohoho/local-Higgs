"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type Preset = { id: string; name: string; desc: string };
type Options = {
  camera_presets: Preset[];
  resolutions: string[];
  aspects: string[];
  durations: string[];
  qualities: string[];
};
type Job = {
  id: string;
  status: "queued" | "submitted" | "running" | "done" | "failed";
  prompt: string;
  preset: string;
  resolution: string;
  aspect: string;
  duration: string;
  quality: string;
  seed: number;
  has_image: boolean;
  video_url: string | null;
  gen_seconds: number | null;
  error: string | null;
  created_at: number;
  progress: { percent: number; eta_s: number; stage: string } | null;
  warning: string | null;
  audio_mode: string;
  speech_text: string | null;
  voice: string | null;
  sfx_prompt: string | null;
};
type CachePart = { id: string; label: string; gb: number; status: string; message: string };
const CACHE_LABEL: Record<string, string> = {
  none: "저장 안 됨", running: "저장 중", done: "저장됨", failed: "실패", unknown: "확인 중",
};
type AudioChoice = "none" | "sfx" | "speech" | "both";
const AUDIO_LABEL: Record<AudioChoice, string> = { none: "없음", sfx: "효과음", speech: "대사", both: "대사+효과음" };
const VOICE_LABEL: Record<string, string> = { female: "여성", male: "남성", clone: "내 목소리" };

function fmtEta(s: number) {
  if (s <= 0) return "곧 완료";
  if (s < 60) return `${s}초 남음`;
  const m = Math.round(s / 60);
  return m < 60 ? `약 ${m}분 남음` : `약 ${Math.floor(m / 60)}시간 ${m % 60}분 남음`;
}
type Status = {
  dry_run: boolean;
  model: string;
  kaggle: { ok: boolean; user?: string; error?: string } | null;
  active_batch: { id: string; status: string } | null;
  batch_detail: string | null;
  queued: number;
  last_error: string | null;
};

const STATUS_LABEL: Record<Job["status"], string> = {
  queued: "대기 중",
  submitted: "Kaggle 전송됨",
  running: "생성 중",
  done: "완료",
  failed: "실패",
};
const QUALITY_LABEL: Record<string, string> = { draft: "초안", standard: "표준", high: "고품질" };

function Segmented({
  value, options, onChange, label = (v: string) => v,
}: { value: string; options: string[]; onChange: (v: string) => void; label?: (v: string) => string }) {
  return (
    <div className="seg">
      {options.map((o) => (
        <button key={o} type="button" className={o === value ? "on" : ""} onClick={() => onChange(o)}>
          {label(o)}
        </button>
      ))}
    </div>
  );
}

export default function Home() {
  const [opts, setOpts] = useState<Options | null>(null);
  const [status, setStatus] = useState<Status | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [apiDown, setApiDown] = useState(false);
  const [cache, setCache] = useState<CachePart[]>([]);
  const [cacheOpen, setCacheOpen] = useState(false);
  const [cacheMsg, setCacheMsg] = useState<string | null>(null);

  const [prompt, setPrompt] = useState("");
  const [preset, setPreset] = useState("none");
  const [resolution, setResolution] = useState("480p");
  const [aspect, setAspect] = useState("16:9");
  const [duration, setDuration] = useState("2");
  const [quality, setQuality] = useState("standard");
  const [count, setCount] = useState(1);
  const [seed, setSeed] = useState("");
  const [image, setImage] = useState<File | null>(null);
  const [audio, setAudio] = useState<AudioChoice>("none");
  const [speech, setSpeech] = useState("");
  const [voice, setVoice] = useState("female");
  const [voiceRef, setVoiceRef] = useState<File | null>(null);
  const [sfxPrompt, setSfxPrompt] = useState("");
  const [preview, setPreview] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      const [s, j] = await Promise.all([
        fetch("/api/status").then((r) => r.json()),
        fetch("/api/jobs").then((r) => r.json()),
      ]);
      setStatus(s);
      setJobs(j);
      setApiDown(false);
    } catch {
      setApiDown(true);
    }
  }, []);

  useEffect(() => {
    fetch("/api/options").then((r) => r.json()).then(setOpts).catch(() => setApiDown(true));
    refresh();
    const t = setInterval(refresh, 5000);
    const loadCache = () =>
      fetch("/api/cache").then((r) => r.json()).then((d) => setCache(d.parts ?? [])).catch(() => {});
    loadCache();
    const c = setInterval(loadCache, 30000);
    return () => {
      clearInterval(t);
      clearInterval(c);
    };
  }, [refresh]);

  async function buildCache() {
    setCacheMsg("Kaggle에 저장 노트북 보내는 중…");
    const r = await fetch("/api/cache/build", { method: "POST" });
    const d = await r.json();
    if (!r.ok) return setCacheMsg(d.detail ?? "실패");
    setCacheMsg(d.started.length ? `${d.started.length}개 저장 시작 (10~20분, GPU 시간 안 씀)` : "이미 저장됐거나 저장 중이에요");
    fetch("/api/cache").then((x) => x.json()).then((x) => setCache(x.parts ?? []));
  }

  function pickImage(f: File | null) {
    setImage(f);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(f ? URL.createObjectURL(f) : null);
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!prompt.trim()) return setMsg("프롬프트를 입력해주세요");
    setBusy(true);
    setMsg(null);
    const fd = new FormData();
    fd.append("prompt", prompt);
    fd.append("preset", preset);
    fd.append("resolution", resolution);
    fd.append("aspect", aspect);
    fd.append("duration", duration);
    fd.append("quality", quality);
    fd.append("count", String(count));
    fd.append("seed", seed.trim() === "" ? "-1" : seed.trim());
    if (image) fd.append("image", image);
    const wantSpeech = audio === "speech" || audio === "both";
    if (wantSpeech && !speech.trim()) {
      setBusy(false);
      return setMsg("대사를 입력해주세요");
    }
    fd.append("audio_mode", audio === "sfx" || audio === "both" ? "sfx" : "none");
    fd.append("sfx_prompt", sfxPrompt);
    if (wantSpeech) {
      fd.append("speech_text", speech);
      fd.append("voice", voice);
      if (voice === "clone" && voiceRef) fd.append("voice_ref", voiceRef);
    }
    try {
      const r = await fetch("/api/jobs", { method: "POST", body: fd });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || "요청 실패");
      setMsg(`${data.length}개 대기열에 추가됨`);
      refresh();
    } catch (err) {
      setMsg(String((err as Error).message));
    } finally {
      setBusy(false);
    }
  }

  async function act(id: string, kind: "retry" | "delete") {
    const r = await fetch(kind === "retry" ? `/api/jobs/${id}/retry` : `/api/jobs/${id}`, {
      method: kind === "retry" ? "POST" : "DELETE",
    });
    if (!r.ok) setMsg((await r.json()).detail);
    refresh();
  }

  function reuse(j: Job) {
    setPrompt(j.prompt);
    setPreset(j.preset);
    setResolution(j.resolution);
    setAspect(j.aspect);
    setDuration(j.duration);
    setQuality(j.quality);
    setSeed(String(j.seed));
    const sp = !!j.speech_text, fx = j.audio_mode === "sfx";
    setAudio(sp && fx ? "both" : sp ? "speech" : fx ? "sfx" : "none");
    setSpeech(j.speech_text ?? "");
    setVoice(j.voice ?? "female");
    setSfxPrompt(j.sfx_prompt ?? "");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  const k = status?.kaggle;
  const statusText = apiDown
    ? "백엔드 연결 안 됨"
    : !status
      ? "연결 중…"
      : status.dry_run
        ? "DRY RUN (테스트 모드)"
        : k && !k.ok
          ? "Kaggle 인증 실패"
          : `Kaggle: ${k?.user ?? "…"}`;
  const statusOk = !apiDown && !!status && (status.dry_run || !!k?.ok);

  return (
    <main>
      <header>
        <div className="logo">local<span>Higgs</span></div>
        <div className="status">
          <span className={`dot ${statusOk ? "ok" : "bad"}`} />
          {statusText}
          {status?.active_batch && (
            <span className="pill">GPU 작업 중 · {status.batch_detail ?? status.active_batch.status}</span>
          )}
          {!!status?.queued && <span className="pill">대기 {status.queued}</span>}
          {cache.length > 0 && (
            <button className="pill btn" onClick={() => setCacheOpen((v) => !v)}>
              모델 저장 {cache.filter((c) => c.status === "done").length}/{cache.length}
            </button>
          )}
        </div>
      </header>

      {cacheOpen && (
        <div className="cachebox">
          <p>
            모델을 Kaggle에 한 번 저장해두면, 생성할 때마다 받던 다운로드(약 24GB)를 건너뛰어요.
            저장은 GPU 시간을 쓰지 않아요.
          </p>
          <ul>
            {cache.map((c) => (
              <li key={c.id}>
                <span>{c.label} <small>약 {c.gb}GB</small></span>
                <b className={`cs ${c.status}`}>{CACHE_LABEL[c.status] ?? c.status}</b>
              </li>
            ))}
          </ul>
          <button className="go small" onClick={buildCache}>Kaggle에 모델 저장하기</button>
          {cacheMsg && <p className="hint">{cacheMsg}</p>}
        </div>
      )}

      {k && !k.ok && !status?.dry_run && (
        <div className="banner">
          Kaggle 연결 안 됨: {k.error}
          <br />
          kaggle.com/settings/api → Generate New Token → 복사 후 터미널에서{" "}
          <code>mkdir -p ~/.kaggle && pbpaste &gt; ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token</code>
          {" "}실행. 30초 안에 자동으로 다시 확인해요.
        </div>
      )}
      {status?.last_error && <div className="banner">{status.last_error}</div>}

      <div className="layout">
        <form className="panel" onSubmit={submit}>
          <label className="lbl">프롬프트</label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="예: a girl with pink hair walking through a university campus at sunset, cinematic"
            rows={5}
          />
          <p className="hint">영어 프롬프트가 더 정확해요.</p>

          <label className="lbl">시작 이미지 (선택)</label>
          <div
            className={`drop ${preview ? "has" : ""}`}
            onClick={() => fileRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              pickImage(e.dataTransfer.files?.[0] ?? null);
            }}
          >
            {preview ? <img src={preview} alt="" /> : <span>클릭하거나 이미지를 끌어다 놓기</span>}
            <input
              ref={fileRef}
              type="file"
              accept="image/png,image/jpeg,image/webp"
              hidden
              onChange={(e) => pickImage(e.target.files?.[0] ?? null)}
            />
          </div>
          {image && (
            <button type="button" className="link" onClick={() => pickImage(null)}>
              이미지 제거
            </button>
          )}

          <label className="lbl">카메라 무빙</label>
          <div className="presets">
            {opts?.camera_presets.map((p) => (
              <button
                type="button"
                key={p.id}
                className={`preset ${preset === p.id ? "on" : ""}`}
                onClick={() => setPreset(p.id)}
                title={p.desc}
              >
                <b>{p.name}</b>
                <small>{p.desc}</small>
              </button>
            ))}
          </div>

          {opts && (
            <div className="grid2">
              <div>
                <label className="lbl">해상도</label>
                <Segmented value={resolution} options={opts.resolutions} onChange={setResolution} />
              </div>
              <div>
                <label className="lbl">비율</label>
                <Segmented value={aspect} options={opts.aspects} onChange={setAspect} />
              </div>
              <div>
                <label className="lbl">길이</label>
                <Segmented value={duration} options={opts.durations} onChange={setDuration} label={(v) => `${v}초`} />
              </div>
              <div>
                <label className="lbl">품질</label>
                <Segmented value={quality} options={opts.qualities} onChange={setQuality} label={(v) => QUALITY_LABEL[v] ?? v} />
              </div>
              <div>
                <label className="lbl">개수</label>
                <Segmented value={String(count)} options={["1", "2", "3", "4"]} onChange={(v) => setCount(Number(v))} />
              </div>
              <div>
                <label className="lbl">시드 (비우면 랜덤)</label>
                <input
                  className="input"
                  value={seed}
                  onChange={(e) => setSeed(e.target.value.replace(/[^0-9]/g, ""))}
                  placeholder="랜덤"
                />
              </div>
            </div>
          )}
          <label className="lbl">오디오</label>
          <Segmented
            value={audio}
            options={["none", "sfx", "speech", "both"]}
            onChange={(v) => setAudio(v as AudioChoice)}
            label={(v) => AUDIO_LABEL[v as AudioChoice]}
          />
          {(audio === "speech" || audio === "both") && (
            <>
              <label className="lbl">대사 (최대 5초)</label>
              <textarea
                value={speech}
                onChange={(e) => setSpeech(e.target.value.slice(0, 120))}
                placeholder="예: 안녕! 여기가 경희대 국제캠퍼스야."
                rows={2}
              />
              <label className="lbl">목소리</label>
              <Segmented value={voice} options={["female", "male", "clone"]} onChange={setVoice} label={(v) => VOICE_LABEL[v]} />
              {voice === "clone" && (
                <input
                  className="input file"
                  type="file"
                  accept="audio/*"
                  onChange={(e) => setVoiceRef(e.target.files?.[0] ?? null)}
                />
              )}
              <p className="hint">
                영상 길이는 대사에 맞춰 자동으로 정해져요. 얼굴이 정면으로 크게 보여야 입 모양이 잘 맞아요.
                {voice === "clone" ? " 내 목소리는 3~10초 녹음 파일을 올려주세요." : " 한국어는 '내 목소리'가 가장 자연스러워요."}
              </p>
            </>
          )}
          {(audio === "sfx" || audio === "both") && (
            <>
              <label className="lbl">효과음 설명 (선택)</label>
              <input
                className="input"
                value={sfxPrompt}
                onChange={(e) => setSfxPrompt(e.target.value)}
                placeholder="비우면 영상 프롬프트 사용. 예: rain, footsteps, wind"
              />
            </>
          )}

          {resolution === "720p" && (
            <p className="hint warn">720p·5초는 T4에서 오래 걸려요. 먼저 480p로 테스트하세요.</p>
          )}

          <button className="go" disabled={busy}>
            {busy ? "추가 중…" : "생성하기"}
          </button>
          {msg && <p className="hint">{msg}</p>}
        </form>

        <section className="gallery">
          {jobs.length === 0 && <div className="empty">아직 생성한 영상이 없어요.</div>}
          {jobs.map((j) => (
            <article key={j.id} className={`card ${j.status}`}>
              <div className={`media r${j.aspect.replace(":", "x")}`}>
                {j.video_url ? (
                  <video src={j.video_url} controls loop playsInline preload="metadata"
                    muted={!j.speech_text && j.audio_mode !== "sfx"} />
                ) : (
                  <div className="placeholder">
                    {j.status === "failed" ? "⚠" : j.progress ? (
                      <b className="pct">{j.progress.percent}%</b>
                    ) : (
                      <span className="spin" />
                    )}
                    <span>{j.progress ? j.progress.stage : STATUS_LABEL[j.status]}</span>
                    {j.progress && (
                      <>
                        <div className="bar"><i style={{ width: `${j.progress.percent}%` }} /></div>
                        <small>{fmtEta(j.progress.eta_s)} (예상)</small>
                      </>
                    )}
                  </div>
                )}
              </div>
              <div className="meta">
                <p className="prompt" title={j.prompt}>{j.prompt}</p>
                <p className="tags">
                  <span>{STATUS_LABEL[j.status]}</span>
                  <span>{j.resolution} · {j.aspect} · {j.duration === "auto" ? "자동" : `${j.duration}초`}</span>
                  {j.speech_text && <span>대사 · {VOICE_LABEL[j.voice ?? "female"]}</span>}
                  {j.audio_mode === "sfx" && <span>효과음</span>}
                  {j.preset !== "none" && <span>{opts?.camera_presets.find((p) => p.id === j.preset)?.name}</span>}
                  {j.has_image && <span>I2V</span>}
                  {j.gen_seconds ? <span>{Math.round(j.gen_seconds)}s</span> : null}
                </p>
                {j.warning && (
                  <details className="warn">
                    <summary>경고 보기</summary>
                    <pre>{j.warning}</pre>
                  </details>
                )}
                {j.error && (
                  <details>
                    <summary>오류 보기</summary>
                    <pre>{j.error}</pre>
                  </details>
                )}
                <div className="actions">
                  <button onClick={() => reuse(j)}>설정 재사용</button>
                  {j.video_url && <a href={j.video_url} download>다운로드</a>}
                  {j.status === "failed" && <button onClick={() => act(j.id, "retry")}>재시도</button>}
                  {!["submitted", "running"].includes(j.status) && (
                    <button className="del" onClick={() => act(j.id, "delete")}>삭제</button>
                  )}
                </div>
              </div>
            </article>
          ))}
        </section>
      </div>
    </main>
  );
}
