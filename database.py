"""SQLite persistence layer for users and attendance records."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from config import DATABASE_PATH, TIMEZONE


def get_connection() -> sqlite3.Connection:
    """Open a configured SQLite connection.

    Foreign-key enforcement is per connection in SQLite, so it is enabled each
    time. The timeout/busy_timeout also helps when the API and camera process
    access the database at roughly the same time.
    """
    db_path = Path(DATABASE_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(str(db_path), timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 10000")
    return connection


def _now() -> datetime:
    return datetime.now(ZoneInfo(TIMEZONE))


def init_db() -> None:
    """Create tables and safely migrate the original attendance schema.

    Older versions stored only `timestamp`; this migration adds
    `attendance_date`, fills it from existing timestamps, removes duplicate
    same-day records (keeping the earliest row), then enforces one attendance
    record per user per date with a unique index.
    """
    connection = get_connection()
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                embedding BLOB NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                attendance_date TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )

        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(attendance)").fetchall()
        }
        if "attendance_date" not in columns:
            connection.execute(
                "ALTER TABLE attendance ADD COLUMN attendance_date TEXT"
            )

        connection.execute(
            """
            UPDATE attendance
            SET attendance_date = substr(timestamp, 1, 10)
            WHERE attendance_date IS NULL OR attendance_date = ''
            """
        )

        # Repair duplicates from older databases before creating the unique index.
        connection.execute(
            """
            DELETE FROM attendance
            WHERE id NOT IN (
                SELECT MIN(id)
                FROM attendance
                WHERE attendance_date IS NOT NULL
                GROUP BY user_id, attendance_date
            )
            AND attendance_date IS NOT NULL
            """
        )
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS
                ux_attendance_user_date
            ON attendance(user_id, attendance_date)
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS ix_attendance_timestamp ON attendance(timestamp)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS ix_attendance_date ON attendance(attendance_date)"
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def save_user(name: str, embedding: Any) -> int | None:
    """Save a user's normalized face embedding.

    Returns the new user ID. Returns None if the unique name already exists.
    Invalid names or embeddings raise ValueError so callers can show a useful
    validation message rather than silently saving bad data.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("User name cannot be empty")
    name = name.strip()

    vector = np.asarray(embedding, dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("Embedding must be a non-empty one-dimensional vector")
    if not np.isfinite(vector).all():
        raise ValueError("Embedding contains invalid numeric values")

    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise ValueError("Embedding cannot be a zero vector")
    vector = vector / norm
    embedding_bytes = vector.astype(np.float32, copy=False).tobytes()
    created_at = _now().strftime("%Y-%m-%d %H:%M:%S")

    connection = get_connection()
    try:
        cursor = connection.execute(
            "INSERT INTO users (name, embedding, created_at) VALUES (?, ?, ?)",
            (name, sqlite3.Binary(embedding_bytes), created_at),
        )
        connection.commit()
        return int(cursor.lastrowid)
    except sqlite3.IntegrityError as exc:
        connection.rollback()
        # Treat a duplicate name as the expected non-exceptional outcome.
        if "users.name" in str(exc) or "UNIQUE constraint failed" in str(exc):
            return None
        raise
    finally:
        connection.close()


def get_all_users() -> list[dict[str, Any]]:
    """Return registered users including decoded embeddings for face matching."""
    connection = get_connection()
    try:
        rows = connection.execute(
            "SELECT id, name, embedding, created_at FROM users ORDER BY id"
        ).fetchall()
        users = []
        for row in rows:
            embedding = np.frombuffer(row["embedding"], dtype=np.float32).copy()
            users.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "embedding": embedding,
                    "created_at": row["created_at"],
                }
            )
        return users
    finally:
        connection.close()


def get_users(limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
    """Return public user fields only; embeddings are never returned by this query."""
    connection = get_connection()
    try:
        rows = connection.execute(
            """
            SELECT id, name, created_at
            FROM users
            ORDER BY id
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def get_user(user_id: int) -> dict[str, Any] | None:
    """Look up a user without exposing their biometric embedding."""
    connection = get_connection()
    try:
        row = connection.execute(
            "SELECT id, name, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        return dict(row) if row is not None else None
    finally:
        connection.close()


def get_all_attendance(
    limit: int = 100,
    offset: int = 0,
    user_id: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict[str, Any]]:
    """Return paginated attendance joined with the user's name.

    Date filters use YYYY-MM-DD strings and are inclusive.
    """
    conditions: list[str] = []
    parameters: list[Any] = []

    if user_id is not None:
        conditions.append("attendance.user_id = ?")
        parameters.append(user_id)
    if start_date is not None:
        conditions.append("attendance.attendance_date >= ?")
        parameters.append(start_date)
    if end_date is not None:
        conditions.append("attendance.attendance_date <= ?")
        parameters.append(end_date)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    parameters.extend([limit, offset])

    connection = get_connection()
    try:
        rows = connection.execute(
            f"""
            SELECT
                attendance.id,
                users.id AS user_id,
                users.name,
                attendance.timestamp,
                attendance.attendance_date
            FROM attendance
            JOIN users ON users.id = attendance.user_id
            {where_clause}
            ORDER BY attendance.timestamp DESC, attendance.id DESC
            LIMIT ? OFFSET ?
            """,
            parameters,
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


if __name__ == "__main__":
    init_db()
    print(f"FaceID database initialized at: {DATABASE_PATH}")
