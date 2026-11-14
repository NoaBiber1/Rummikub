"""SQLite persistence for game sessions, turn history, community challenges and the leaderboard.

Board and rack states are stored only as JSON text (Tile schema: ``{id, color, value, isJoker}``),
never as image bytes.
"""
import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

DB_PATH = os.environ.get("RUMMIKUB_DB_PATH", "rummikub.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS game_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'packed')),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS turn_history (
    id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL REFERENCES game_sessions(id),
    turn_index INTEGER NOT NULL,
    board_state_json TEXT NOT NULL,
    rack_state_json TEXT NOT NULL,
    move_applied_json TEXT,
    UNIQUE (game_id, turn_index)
);

CREATE TABLE IF NOT EXISTS community_challenges (
    id TEXT PRIMARY KEY,
    initial_board_json TEXT NOT NULL,
    initial_rack_json TEXT NOT NULL,
    board_hash TEXT NOT NULL,
    ai_score INTEGER NOT NULL,
    created_from_session_id TEXT NOT NULL REFERENCES game_sessions(id),
    turn_index INTEGER NOT NULL,
    UNIQUE (created_from_session_id, turn_index),
    UNIQUE (board_hash)
);

CREATE TABLE IF NOT EXISTS challenge_submissions (
    challenge_id TEXT NOT NULL REFERENCES community_challenges(id),
    user_id TEXT NOT NULL,
    human_placed INTEGER NOT NULL,
    score INTEGER NOT NULL,
    PRIMARY KEY (challenge_id, user_id)
);

CREATE TABLE IF NOT EXISTS leaderboard (
    user_id TEXT PRIMARY KEY,
    total_points INTEGER NOT NULL DEFAULT 0,
    challenges_won INTEGER NOT NULL DEFAULT 0
);
"""


def connect(path: Optional[str] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(path or DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def _tile_json(tile: dict[str, Any]) -> dict[str, Any]:
    color = tile["color"]
    return {
        "id": tile["id"],
        "color": color,
        "value": tile["value"],
        "isJoker": bool(tile.get("isJoker", color == "JOKER")),
    }


def to_json(data: Any) -> str:
    """Serialize a board (list of sets) or rack (list of tiles) to JSON, normalizing tiles."""

    def norm(node: Any) -> Any:
        if isinstance(node, dict):
            if "color" in node and "value" in node:
                return _tile_json(node)
            return {k: norm(v) for k, v in node.items()}
        if isinstance(node, list):
            return [norm(v) for v in node]
        return node

    return json.dumps(norm(data), sort_keys=True, separators=(",", ":"))


def board_hash(board: list[dict[str, Any]], rack: list[dict[str, Any]]) -> str:
    """Order-independent hash of the tile types (color, value) on the board and in the rack."""
    board_tiles = sorted(
        (t["color"], t["value"]) for s in board for t in s.get("tiles", [])
    )
    rack_tiles = sorted((t["color"], t["value"]) for t in rack)
    raw = json.dumps([board_tiles, rack_tiles])
    return hashlib.sha256(raw.encode()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_session(conn: sqlite3.Connection, user_id: str) -> str:
    sid = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO game_sessions (id, user_id, status, created_at) VALUES (?, ?, 'active', ?)",
        (sid, user_id, _now()),
    )
    conn.commit()
    return sid


def get_session(conn: sqlite3.Connection, session_id: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM game_sessions WHERE id = ?", (session_id,)).fetchone()


def end_session(conn: sqlite3.Connection, session_id: str) -> None:
    conn.execute("UPDATE game_sessions SET status = 'packed' WHERE id = ?", (session_id,))
    conn.commit()


def next_turn_index(conn: sqlite3.Connection, session_id: str) -> int:
    row = conn.execute(
        "SELECT MAX(turn_index) AS last FROM turn_history WHERE game_id = ?", (session_id,)
    ).fetchone()
    return 0 if row["last"] is None else row["last"] + 1


def record_turn(
    conn: sqlite3.Connection,
    game_id: str,
    turn_index: int,
    board: list[dict[str, Any]],
    rack: list[dict[str, Any]],
    move_applied: Optional[Any] = None,
) -> str:
    tid = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO turn_history (id, game_id, turn_index, board_state_json, rack_state_json, move_applied_json)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (
            tid,
            game_id,
            turn_index,
            to_json(board),
            to_json(rack),
            None if move_applied is None else to_json(move_applied),
        ),
    )
    conn.commit()
    return tid


def create_challenge(
    conn: sqlite3.Connection,
    board: list[dict[str, Any]],
    rack: list[dict[str, Any]],
    ai_score: int,
    session_id: str,
    turn_index: int,
) -> str:
    cid = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO community_challenges (id, initial_board_json, initial_rack_json, board_hash,"
        " ai_score, created_from_session_id, turn_index) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (cid, to_json(board), to_json(rack), board_hash(board, rack), ai_score, session_id, turn_index),
    )
    conn.commit()
    return cid


def get_challenge(conn: sqlite3.Connection, challenge_id: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM community_challenges WHERE id = ?", (challenge_id,)).fetchone()


def list_challenges(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM community_challenges ORDER BY rowid DESC").fetchall()


def record_submission(
    conn: sqlite3.Connection, challenge_id: str, user_id: str, human_placed: int, ai_score: int
) -> int:
    """Store a user's answer and add its score to the leaderboard. Returns the score.

    ``score = human_placed - ai_score``. The challenge counts as won when the score is positive.
    Raises ``sqlite3.IntegrityError`` if the user already submitted for this challenge.
    """
    score = human_placed - ai_score
    try:
        conn.execute(
            "INSERT INTO challenge_submissions (challenge_id, user_id, human_placed, score)"
            " VALUES (?, ?, ?, ?)",
            (challenge_id, user_id, human_placed, score),
        )
        conn.execute(
            "INSERT INTO leaderboard (user_id, total_points, challenges_won) VALUES (?, ?, ?)"
            " ON CONFLICT(user_id) DO UPDATE SET total_points = total_points + excluded.total_points,"
            " challenges_won = challenges_won + excluded.challenges_won",
            (user_id, score, 1 if score > 0 else 0),
        )
        conn.commit()
    except sqlite3.Error:
        conn.rollback()
        raise
    return score


def get_leaderboard(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM leaderboard ORDER BY total_points DESC, challenges_won DESC, user_id ASC"
    ).fetchall()
