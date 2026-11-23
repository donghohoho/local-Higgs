import sqlite3
import threading
import time
import uuid

from .config import DB_PATH

_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    status TEXT NOT NULL,          -- queued | submitted | running | done | failed
    prompt TEXT NOT NULL,
    final_prompt TEXT NOT NULL,
    preset TEXT NOT NULL,
    resolution TEXT NOT NULL,
    aspect TEXT NOT NULL,
    duration TEXT NOT NULL,
    quality TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    num_frames INTEGER NOT NULL,
    steps INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    image_path TEXT,
    batch_id TEXT,
    output_file TEXT,
    gen_seconds REAL,
    error TEXT,
    audio_mode TEXT NOT NULL DEFAULT 'none',  -- none | sfx
    sfx_prompt TEXT,
    speech_text TEXT,
    voice TEXT,                    -- female | male | clone
    voice_ref_path TEXT,
    engine TEXT NOT NULL DEFAULT 'kaggle'      -- kaggle | local
);
CREATE TABLE IF NOT EXISTS batches (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    status TEXT NOT NULL,          -- submitted | running | done | failed
    message TEXT
);
"""


MIGRATIONS = [
    ("audio_mode", "TEXT NOT NULL DEFAULT 'none'"),
    ("sfx_prompt", "TEXT"),
    ("speech_text", "TEXT"),
    ("voice", "TEXT"),
    ("voice_ref_path", "TEXT"),
    ("engine", "TEXT NOT NULL DEFAULT 'kaggle'"),
]


def _conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init():
    with _lock, _conn() as c:
        c.executescript(SCHEMA)
        # older databases: add the audio columns
        have = {r["name"] for r in c.execute("PRAGMA table_info(jobs)")}
        for col, typ in MIGRATIONS:
            if col not in have:
                c.execute(f"ALTER TABLE jobs ADD COLUMN {col} {typ}")


def new_id():
    return uuid.uuid4().hex[:10]


def insert_job(job):
    now = time.time()
    job = {**job, "created_at": now, "updated_at": now}
    cols = ", ".join(job)
    qs = ", ".join("?" for _ in job)
    with _lock, _conn() as c:
        c.execute(f"INSERT INTO jobs ({cols}) VALUES ({qs})", list(job.values()))
    return job


def update_job(job_id, **fields):
    fields["updated_at"] = time.time()
    sets = ", ".join(f"{k}=?" for k in fields)
    with _lock, _conn() as c:
        c.execute(f"UPDATE jobs SET {sets} WHERE id=?", [*fields.values(), job_id])


def get_job(job_id):
    with _lock, _conn() as c:
        r = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return dict(r) if r else None


def list_jobs(limit=200):
    with _lock, _conn() as c:
        rows = c.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def jobs_by_status(status, limit=100, engine=None):
    with _lock, _conn() as c:
        if engine:
            rows = c.execute(
                "SELECT * FROM jobs WHERE status=? AND engine=? ORDER BY created_at ASC LIMIT ?",
                (status, engine, limit),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM jobs WHERE status=? ORDER BY created_at ASC LIMIT ?", (status, limit)
            ).fetchall()
    return [dict(r) for r in rows]


def jobs_in_batch(batch_id):
    with _lock, _conn() as c:
        rows = c.execute("SELECT * FROM jobs WHERE batch_id=?", (batch_id,)).fetchall()
    return [dict(r) for r in rows]


def delete_job(job_id):
    with _lock, _conn() as c:
        c.execute("DELETE FROM jobs WHERE id=?", (job_id,))


def insert_batch(batch_id):
    with _lock, _conn() as c:
        c.execute(
            "INSERT INTO batches (id, created_at, status) VALUES (?, ?, 'submitted')",
            (batch_id, time.time()),
        )


def update_batch(batch_id, status, message=None):
    with _lock, _conn() as c:
        c.execute("UPDATE batches SET status=?, message=? WHERE id=?", (status, message, batch_id))


def active_batch():
    with _lock, _conn() as c:
        r = c.execute(
            "SELECT * FROM batches WHERE status IN ('submitted','running') ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    return dict(r) if r else None


def last_batch():
    with _lock, _conn() as c:
        r = c.execute("SELECT * FROM batches ORDER BY created_at DESC LIMIT 1").fetchone()
    return dict(r) if r else None
