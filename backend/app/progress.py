"""Progress % and time left for jobs running on Kaggle.

The worker prints `LHP {...}` markers. A follower thread reads the live
Kaggle log, keeps the latest state per job, and the API turns that into:

  percent = elapsed / (elapsed + remaining)

`remaining` uses the live step speed of the current job, and learned
averages (data/stats.json) for steps not started yet. Every finished job
makes the estimate better.
"""

import json
import threading
import time

from . import config

STATS_PATH = config.DATA_DIR / ("stats-dry.json" if config.DRY_RUN else "stats.json")
# first guesses for a Kaggle T4; replaced by real measurements after one run
DEFAULT_STATS = {
    "setup_s": 420.0,        # submit -> first job starts (queue, pip, model download, load)
    "load_s": 90.0,          # loading the second pipeline (t2v -> i2v)
    "step_s_per_unit": 0.45,  # seconds per denoise step per unit
    "decode_s_per_unit": 1.2,
}
ALPHA = 0.5


def unit(job):
    """Work size: megapixels x frames."""
    return job["width"] * job["height"] * job["num_frames"] / 1e6


def _load_stats():
    try:
        return {**DEFAULT_STATS, **json.loads(STATS_PATH.read_text())}
    except Exception:
        return dict(DEFAULT_STATS)


class Tracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.stats = _load_stats()
        self.batches = {}  # batch_id -> state
        self.max_pct = {}  # job_id -> highest percent shown (never go backwards)

    # ---------- learning ----------
    def _learn(self, key, sample):
        if sample <= 0:
            return
        self.stats[key] = (1 - ALPHA) * self.stats[key] + ALPHA * sample
        try:
            STATS_PATH.write_text(json.dumps(self.stats, indent=2))
        except Exception:
            pass

    # ---------- batch lifecycle ----------
    def start(self, batch_id, jobs, submit_ts, stream_factory):
        with self.lock:
            self.batches[batch_id] = {
                "submit_ts": submit_ts,
                "units": {j["id"]: unit(j) for j in jobs},
                "jobs": {},            # id -> {start, n, first, last, done}
                "load_ts": None,
                "load_kind": None,
                "started_any": False,
                "stop": False,
            }
        t = threading.Thread(target=self._follow, args=(batch_id, stream_factory), daemon=True)
        t.start()

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
            time.sleep(15)  # stream dropped or not ready yet: reconnect

    # ---------- events ----------
    def handle(self, batch_id, ev):
        with self.lock:
            b = self.batches.get(batch_id)
            if not b:
                return
            t, ts = ev.get("t"), float(ev.get("ts", time.time()))
            jid = ev.get("id")
            if t == "load":
                if b["started_any"] and b["load_kind"] != ev.get("kind"):
                    b["load_ts"] = ts
                b["load_kind"] = ev.get("kind")
            elif t == "start" and jid:
                j = b["jobs"].setdefault(jid, {})
                if "start" in j:
                    return  # replayed event
                j.update(start=ts, n=int(ev.get("n", 1)))
                if not b["started_any"]:
                    b["started_any"] = True
                    self._learn("setup_s", ts - b["submit_ts"])
                elif b["load_ts"]:
                    self._learn("load_s", ts - b["load_ts"])
                    b["load_ts"] = None
            elif t == "step" and jid:
                j = b["jobs"].setdefault(jid, {"start": ts, "n": int(ev.get("n", 1))})
                i = int(ev.get("i", 0))
                if "first" not in j:
                    j["first"] = (i, ts)
                if i >= j.get("last", (0, 0))[0]:
                    j["last"] = (i, ts)
            elif t in ("done", "fail") and jid:
                j = b["jobs"].setdefault(jid, {})
                if j.get("done"):
                    return
                j["done"] = ts
                u = b["units"].get(jid, 1) or 1
                if t == "done" and "first" in j and "last" in j:
                    (i0, t0), (i1, t1) = j["first"], j["last"]
                    if i1 > i0:
                        self._learn("step_s_per_unit", (t1 - t0) / (i1 - i0) / u)
                    self._learn("decode_s_per_unit", (ts - t1) / u)

    # ---------- estimate ----------
    def progress(self, batch_id, jobs):
        """jobs: DB rows of this batch. Returns {job_id: {...}}."""
        now = time.time()
        with self.lock:
            b = self.batches.get(batch_id)
            if not b:
                return {}
            s = self.stats
            # same order as the worker: text->video first, then image->video
            ordered = [j for j in jobs if not j["image_path"]] + [j for j in jobs if j["image_path"]]
            has_t2v = any(not j["image_path"] for j in jobs)

            cursor = 0.0  # seconds from now
            if not b["started_any"]:
                cursor += max(s["setup_s"] - (now - b["submit_ts"]), 20)
            out = {}
            prev_kind = None
            for j in ordered:
                kind = "i2v" if j["image_path"] else "t2v"
                u = unit(j)
                st = b["jobs"].get(j["id"], {})
                step_s = s["step_s_per_unit"] * u
                decode_s = s["decode_s_per_unit"] * u
                n = st.get("n", j["steps"])

                if st.get("done"):
                    out[j["id"]] = {"percent": 100, "eta_s": 0, "stage": "마무리 중"}
                    prev_kind = kind
                    continue

                if "last" in st:
                    (i0, t0), (i1, t1) = st["first"], st["last"]
                    if i1 > i0:
                        step_s = (t1 - t0) / (i1 - i0)  # live speed
                    left_steps = max(n - i1, 0)
                    rem = left_steps * step_s + decode_s
                    rem -= min(now - t1, step_s if left_steps else decode_s) * 0.9
                    stage = f"생성 중 {i1}/{n} 스텝" if left_steps else "영상 저장 중"
                elif "start" in st:
                    rem = n * step_s + decode_s
                    stage = "생성 시작"
                else:
                    rem = n * step_s + decode_s
                    if prev_kind == "t2v" and kind == "i2v" and has_t2v:
                        rem += s["load_s"]
                    stage = "GPU 준비 중" if not b["started_any"] else "앞 작업 대기 중"
                prev_kind = kind

                cursor += max(rem, 5)
                elapsed = now - b["submit_ts"]
                pct = int(min(99, 100 * elapsed / (elapsed + cursor)))
                pct = max(pct, self.max_pct.get(j["id"], 0))
                self.max_pct[j["id"]] = pct
                out[j["id"]] = {"percent": pct, "eta_s": int(cursor), "stage": stage}
            return out


tracker = Tracker()
