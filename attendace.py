"""Attendance business logic shared by the camera recognizer and FastAPI."""

from datetime import datetime
from zoneinfo import ZoneInfo

from config import TIMEZONE
from database import get_connection


def mark_attendance(user_id: int, now: datetime | None = None) -> bool:
    """Mark attendance once per user per local calendar day.

    Returns True if a new record was inserted, or False if the user was already
    marked for that day. A LookupError is raised for a non-existent user.
    Database uniqueness, not an in-memory set, is the final protection against
    duplicate marks from simultaneous API/camera calls.
    """
    if not isinstance(user_id, int) or isinstance(user_id, bool) or user_id <= 0:
        raise ValueError("user_id must be a positive integer")

    timezone = ZoneInfo(TIMEZONE)
    current = now or datetime.now(timezone)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone)
    else:
        current = current.astimezone(timezone)

    timestamp = current.strftime("%Y-%m-%d %H:%M:%S")
    attendance_date = current.strftime("%Y-%m-%d")

    connection = get_connection()
    try:
        user = connection.execute(
            "SELECT 1 FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if user is None:
            raise LookupError("User not found")

        cursor = connection.execute(
            """
            INSERT INTO attendance (user_id, timestamp, attendance_date)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id, attendance_date) DO NOTHING
            """,
            (user_id, timestamp, attendance_date),
        )
        connection.commit()
        return cursor.rowcount == 1
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
