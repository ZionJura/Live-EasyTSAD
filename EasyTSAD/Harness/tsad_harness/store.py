"""Single-writer domain state and replayable events (DSH owns conversation logs)."""
from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
import uuid
from pathlib import Path

from .task import now


def identity(prefix: str) -> str:
    return prefix + "_" + uuid.uuid4().hex[:16]


class Store:
    KINDS = {"batches", "jobs", "attempts", "bindings", "diagnostics", "metrics", "cursors", "processes", "messages"}

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=30)
        self.db.row_factory = sqlite3.Row
        # DELETE journal works on the existing fuse filesystem; no WAL shared-memory dependency.
        self.db.executescript("""
            PRAGMA foreign_keys=ON;
            PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS objects(kind TEXT, id TEXT, body TEXT NOT NULL,
                PRIMARY KEY(kind,id));
            CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT,
                at TEXT NOT NULL, type TEXT NOT NULL, body TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS requests(id TEXT PRIMARY KEY, operation TEXT NOT NULL,
                input TEXT NOT NULL, result TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS leases(resource TEXT PRIMARY KEY, owner TEXT NOT NULL,
                body TEXT NOT NULL);
        """)

    @contextlib.contextmanager
    def transaction(self):
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise

    def get(self, kind: str, key: str) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT body FROM objects WHERE kind=? AND id=?", (kind, key)).fetchone()
            return json.loads(row[0]) if row else None

    def all(self, kind: str) -> list[dict]:
        with self.lock:
            return [json.loads(r[0]) for r in self.db.execute(
                "SELECT body FROM objects WHERE kind=? ORDER BY rowid", (kind,))]

    def put(self, kind: str, obj: dict) -> dict:
        if kind not in self.KINDS:
            raise ValueError(kind)
        obj = {**obj, "updatedAt": now()}
        with self.lock:
            self.db.execute("INSERT INTO objects VALUES(?,?,?) ON CONFLICT(kind,id) DO UPDATE SET body=excluded.body",
                            (kind, obj["id"], json.dumps(obj, ensure_ascii=False, allow_nan=False)))
        return obj

    def update(self, kind: str, key: str, **fields) -> dict:
        with self.lock:
            obj = self.get(kind, key)
            if obj is None:
                raise KeyError(key)
            return self.put(kind, {**obj, **fields})

    def update_active(self, kind: str, key: str, **fields) -> dict | None:
        """CAS-style worker write: user pause/cancel always wins stale worker updates."""
        with self.lock:
            current = self.get(kind, key)
            if not current or not current.get("authorized") or current.get("status") in ("paused", "cancelled", "failed", "done"):
                return None
            return self.put(kind, {**current, **fields})

    def event(self, event_type: str, **data) -> int:
        with self.lock:
            cur = self.db.execute("INSERT INTO events(at,type,body) VALUES(?,?,?)",
                                  (now(), event_type, json.dumps(data, ensure_ascii=False, default=str, allow_nan=False)))
            return cur.lastrowid

    def events(self, after: int = 0, limit: int = 300) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT * FROM events WHERE seq>? ORDER BY seq LIMIT ?", (after, limit))
            return [{"id": r["seq"], "at": r["at"], "type": r["type"], **json.loads(r["body"])} for r in rows]

    def cursor(self) -> int:
        with self.lock:
            return self.db.execute("SELECT coalesce(max(seq),0) FROM events").fetchone()[0]

    def once(self, request_id: str, operation: str, arguments: dict, fn):
        """Call within the same transaction as object creation; different payloads conflict."""
        encoded = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
        with self.transaction():
            old = self.db.execute("SELECT * FROM requests WHERE id=?", (request_id,)).fetchone()
            if old:
                if old["operation"] != operation or old["input"] != encoded:
                    raise ValueError("requestId was already used with a different request")
                return json.loads(old["result"])
            result = fn()
            self.db.execute("INSERT INTO requests VALUES(?,?,?,?)",
                            (request_id, operation, encoded, json.dumps(result, ensure_ascii=False)))
            return result

    def acquire(self, resources: list[str], owner: str, **details) -> bool:
        with self.transaction():
            for resource in resources:
                row = self.db.execute("SELECT owner FROM leases WHERE resource=?", (resource,)).fetchone()
                if row and row[0] != owner:
                    return False
            for resource in resources:
                self.db.execute("INSERT OR REPLACE INTO leases VALUES(?,?,?)",
                                (resource, owner, json.dumps(details)))
            return True

    def release(self, owner: str):
        with self.lock:
            self.db.execute("DELETE FROM leases WHERE owner=?", (owner,))

    def leases(self) -> list[dict]:
        with self.lock:
            return [{"resource": r[0], "owner": r[1], **json.loads(r[2])}
                    for r in self.db.execute("SELECT * FROM leases")]

    def close(self):
        self.db.close()
