"""Progress % and time left for jobs running on Kaggle.

The worker prints `LHP {...}` markers. A follower thread reads the live
Kaggle log and keeps the latest state. The estimate walks the worker's order:

  setup -> speech -> video (t2v, then i2v) -> lipsync -> sfx

  percent = elapsed / (elapsed + remaining)

Live speed is used where available; learned averages (data/stats.json) for
everything not started yet. Each finished stage makes the next guess better.
"""

import json
import threading
import time

from . import config

STATS_PATH = config.DATA_DIR / ("stats-dry.json" if config.DRY_RUN else "stats.json")
# first guesses for a Kaggle T4; replaced by real measurements after one run
DEFAULT_STATS = {
    "setup_s": 420.0,            # submit -> ready (queue, pip, model download)
    "setup_cached_s": 200.0,     # same, with the video model saved on Kaggle
    "load_s": 90.0,              # loading the second pipeline (t2v -> i2v)
    "step_s_per_unit": 0.45,     # seconds per denoise step per unit
    "decode_s_per_unit": 1.2,
    "speech_s_per_job": 40.0,    # includes OmniVoice download/load, spread over jobs
    "lipsync_setup_s": 360.0,
    "lipsync_s": 90.0,
    "sfx_setup_s": 240.0,
    "sfx_s": 40.0,
}
ALPHA = 0.5
STAGE_LABEL = {"lipsync": "립싱크 중", "sfx": "효과음 생성 중"}


def unit(job):
    """Work size: megapixels x frames."""
    return job["width"] * job["height"] * job["num_frames"] / 1e6


def needs(job):
    speech = bool(job.get("speech_text"))
    return {"speech": speech, "lipsync": speech, "sfx": job.get("audio_mode") == "sfx"}


def _load_stats():
    try:
        return {**DEFAULT_STATS, **json.loads(STATS_PATH.read_text())}
    except Exception:
        return dict(DEFAULT_STATS)


class Tracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.stats = _load_stats()
        self.batches = {}
        self.max_pct = {}

    def _learn(self, key, sample):
        if sample <= 0:
            return
        self.stats[key] = (1 - ALPHA) * self.stats[key] + ALPHA * sample
        try:
            STATS_PATH.write_text(json.dumps(self.stats, indent=2))
        except Exception:
            pass

    # ---------- lifecycle ----------
    def start(self, batch_id, jobs, submit_ts, stream_factory, cached=False):
        with self.lock:
            self.batches[batch_id] = {
                "setup_key": "setup_cached_s" if cached else "setup_s",
                "submit_ts": submit_ts,
                "units": {j["id"]: unit(j) for j in jobs},
                "n_speech": sum(needs(j)["speech"] for j in jobs),
                "jobs": {},
                "phase": {},          # stage -> ts first seen
                "first_load_ts": None,
                "load_ts": None,
                "load_kind": None,
                "started_any": False,
                "last_sfx_ts": None,
                "stop": False,
                "remote": None,        # Kaggle's own status: queued / running ...
            }
        threading.Thread(target=self._follow, args=(batch_id, stream_factory), daemon=True).start()

    def set_remote(self, batch_id, status):
        with self.lock:
            if batch_id in self.batches:
                self.batches[batch_id]["remote"] = status

    def finish(self, batch_id):
        with self.lock:
            b = self.batches.pop(batch_id, None)
            if b:
                b["stop"] = True

    def _follow(self, batch_id, stream_factory):
        while True:
            with self.lock:
                b = self.batches.get(batch_id)
                if not b or b["stop"]:
                    return
            try:
                for ev in stream_factory():
                    self.handle(batch_id, ev)
                    with self.lock:
                        if batch_id not in self.batches:
                            return
            except Exception:
                pass
            time.sleep(15)

    # ---------- events ----------
    def handle(self, batch_id, ev):
        with self.lock:
            b = self.batches.get(batch_id)
            if not b:
                return
            t, ts = ev.get("t"), float(ev.get("ts", time.time()))
            jid = ev.get("id")
            j = b["jobs"].setdefault(jid, {"post": {}, "post_done": {}}) if jid else None

            if t == "phase":
                b["phase"].setdefault(ev.get("stage"), ts)
            elif t == "load":
                if b["first_load_ts"] is None:
                    b["first_load_ts"] = ts
                    sp = b["phase"].get("speech")
                    if sp and b["n_speech"]:
                        self._learn("speech_s_per_job", (ts - sp) / b["n_speech"])
                    self._learn(b["setup_key"], (sp or ts) - b["submit_ts"])
                elif b["load_kind"] != ev.get("kind"):
                    b["load_ts"] = ts
                b["load_kind"] = ev.get("kind")
            elif t == "start":
                if "start" in j:
                    return
                j.update(start=ts, n=int(ev.get("n", 1)))
                if b["started_any"] and b["load_ts"]:
                    self._learn("load_s", ts - b["load_ts"])
                    b["load_ts"] = None
                b["started_any"] = True
            elif t == "step":
                j.setdefault("start", ts)
                j.setdefault("n", int(ev.get("n", 1)))
                i = int(ev.get("i", 0))
                if "first" not in j:
                    j["first"] = (i, ts)
                if i >= j.get("last", (0, 0))[0]:
                    j["last"] = (i, ts)
            elif t == "gen_done":
                if j.get("gen_done"):
                    return
                j["gen_done"] = ts
                u = b["units"].get(jid, 1) or 1
                if "first" in j and "last" in j:
                    (i0, t0), (i1, t1) = j["first"], j["last"]
                    if i1 > i0:
                        self._learn("step_s_per_unit", (t1 - t0) / (i1 - i0) / u)
                    self._learn("decode_s_per_unit", (ts - t1) / u)
            elif t == "post":
                stage = ev.get("stage")
                if stage in j["post"]:
                    return
                j["post"][stage] = ts
                setup_ts = b["phase"].get(f"{stage}_setup")
                if setup_ts and not b.get(f"{stage}_ready"):
                    b[f"{stage}_ready"] = True
                    self._learn(f"{stage}_setup_s", ts - setup_ts)
            elif t == "post_done":
                stage = ev.get("stage")
                if stage in j["post_done"]:
                    return
                j["post_done"][stage] = ts
                if stage == "lipsync" and stage in j["post"]:
                    self._learn("lipsync_s", ts - j["post"][stage])
                elif stage == "sfx":
                    prev = b["last_sfx_ts"] or j["post"].get("sfx")
                    if prev:
                        self._learn("sfx_s", ts - prev)
                    b["last_sfx_ts"] = ts
            elif t in ("done", "fail"):
                j["done"] = ts

    # ---------- estimate ----------
    def progress(self, batch_id, jobs):
        now = time.time()
        with self.lock:
            b = self.batches.get(batch_id)
            if not b:
                return {}
            s = self.stats
            ordered = [j for j in jobs if not j["image_path"]] + [j for j in jobs if j["image_path"]]
            has_t2v = any(not j["image_path"] for j in jobs)
            st = lambda j: b["jobs"].get(j["id"], {"post": {}, "post_done": {}})  # noqa: E731
            finish, stage = {}, {}

            # setup + speech
            cursor = 0.0
            if b["first_load_ts"] is None:
                pre = s[b["setup_key"]] + b["n_speech"] * s["speech_s_per_job"]
                cursor = max(pre - (now - b["submit_ts"]), 20)
                in_speech = "speech" in b["phase"]
                for j in ordered:
                    if b["remote"] == "queued":
                        stage[j["id"]] = "Kaggle GPU 배정 기다리는 중"
                    elif in_speech and needs(j)["speech"]:
                        stage[j["id"]] = "음성 생성 중"
                    else:
                        stage[j["id"]] = "GPU 준비 중"

            # video
            prev_kind = None
            for j in ordered:
                x, u, kind = st(j), unit(j), ("i2v" if j["image_path"] else "t2v")
                step_s, decode_s = s["step_s_per_unit"] * u, s["decode_s_per_unit"] * u
                n = x.get("n", j["steps"])
                if x.get("gen_done") or x.get("done"):
                    finish[j["id"]] = cursor
                    prev_kind = kind
                    continue
                if "last" in x:
                    (i0, t0), (i1, t1) = x["first"], x["last"]
                    if i1 > i0:
                        step_s = (t1 - t0) / (i1 - i0)
                    left = max(n - i1, 0)
                    rem = left * step_s + decode_s - min(now - t1, step_s if left else decode_s) * 0.9
                    stage[j["id"]] = f"영상 생성 {i1}/{n} 스텝" if left else "영상 저장 중"
                else:
                    rem = n * step_s + decode_s
                    if prev_kind == "t2v" and kind == "i2v" and has_t2v and "start" not in x:
                        rem += s["load_s"]
                    if "start" in x:
                        stage[j["id"]] = "영상 생성 시작"
                    else:
                        stage.setdefault(j["id"], "앞 작업 대기 중")
                prev_kind = kind
                cursor += max(rem, 5)
                finish[j["id"]] = cursor

            # lipsync, then sfx (each: one-time setup + per job)
            for name in ("lipsync", "sfx"):
                todo = [j for j in ordered if needs(j)[name] and not st(j).get("done")
                        and name not in st(j)["post_done"]]
                if not todo:
                    continue
                if not b.get(f"{name}_ready"):
                    ph = b["phase"].get(f"{name}_setup")
                    cursor += max(s[f"{name}_setup_s"] - ((now - ph) if ph else 0), 10)
                for j in todo:
                    x = st(j)
                    per = s[f"{name}_s"]
                    if name in x["post"] and (name == "lipsync" or j is todo[0]):
                        rem = max(per - (now - (b["last_sfx_ts"] or x["post"][name]
                                                if name == "sfx" else x["post"][name])), 5)
                        stage[j["id"]] = STAGE_LABEL[name]
                    else:
                        rem = per
                        if x.get("gen_done"):
                            stage.setdefault(j["id"], "후처리 대기 중")
                    cursor += rem
                    finish[j["id"]] = cursor

            out = {}
            elapsed = now - b["submit_ts"]
            for j in ordered:
                x = st(j)
                if x.get("done"):
                    out[j["id"]] = {"percent": 100, "eta_s": 0, "stage": "다운로드 대기 중"}
                    continue
                eta = finish.get(j["id"], cursor) + 3
                pct = int(min(99, 100 * elapsed / (elapsed + eta)))
                pct = max(pct, self.max_pct.get(j["id"], 0))
                self.max_pct[j["id"]] = pct
                out[j["id"]] = {"percent": pct, "eta_s": int(eta), "stage": stage.get(j["id"], "후처리 대기 중")}
            return out


tracker = Tracker()
