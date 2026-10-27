"""SQLite storage. IDs: starter notes keep N001-N012, user notes start at N013.

Command line:  python -m app.db load-starter
"""
import json
import os
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STARTER_FILE = ROOT / "data" / "starter_notes.json"
LAST_STARTER_SEQ = 12  # user notes can never use N001-N012


def db_path() -> Path:
    return Path(os.environ.get("SMART_NOTES_DB", ROOT / "data" / "notes.db"))


def connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init() -> None:
    with closing(connect()) as conn, conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS notes (
            id TEXT PRIMARY KEY, seq INTEGER NOT NULL, text TEXT NOT NULL,
            category TEXT NOT NULL, uncertain INTEGER NOT NULL,
            confidence REAL NOT NULL, created_at TEXT NOT NULL)""")


def _row(r: sqlite3.Row) -> dict:
    return {"id": r["id"], "text": r["text"], "category": r["category"],
            "uncertain": bool(r["uncertain"]), "confidence": r["confidence"], "created_at": r["created_at"]}


def add_note(text: str, result: dict) -> dict:
    """Save a user note. New ID = N + (highest number so far, but at least 12) + 1."""
    with closing(connect()) as conn, conn:
        top = conn.execute("SELECT COALESCE(MAX(seq), 0) FROM notes").fetchone()[0]
        seq = max(LAST_STARTER_SEQ, top) + 1
        note_id = f"N{seq:03d}"
        conn.execute("INSERT INTO notes VALUES (?,?,?,?,?,?,?)",
                     (note_id, seq, text, result["category"], int(result["uncertain"]),
                      result["confidence"], datetime.now(timezone.utc).isoformat()))
        row = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
    return _row(row)


def list_notes(category: str | None = None) -> list[dict]:
    """category: one of the 4 labels, or 'Uncertain' for flagged notes, or None for all."""
    sql, args = "SELECT * FROM notes", ()
    if category == "Uncertain":
        sql += " WHERE uncertain = 1"
    elif category:
        sql, args = sql + " WHERE category = ?", (category,)
    with closing(connect()) as conn:
        return [_row(r) for r in conn.execute(sql + " ORDER BY seq", args)]


def load_starter(classifier) -> dict:
    """Insert N001-N012 if absent (INSERT OR IGNORE), classified by the trained model."""
    notes = json.loads(STARTER_FILE.read_text(encoding="utf-8"))
    added = 0
    with closing(connect()) as conn, conn:
        for n in notes:
            res = classifier.predict(n["text"])
            cur = conn.execute("INSERT OR IGNORE INTO notes VALUES (?,?,?,?,?,?,?)",
                               (n["id"], int(n["id"][1:]), n["text"], res["category"], int(res["uncertain"]),
                                res["confidence"], datetime.now(timezone.utc).isoformat()))
            added += cur.rowcount
    return {"added": added, "already_present": len(notes) - added, "total_starter": len(notes)}


if __name__ == "__main__":
    if sys.argv[1:] == ["load-starter"]:
        from app.classifier import Classifier
        init()
        print(load_starter(Classifier()))
    else:
        sys.exit("usage: python -m app.db load-starter")
