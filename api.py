"""FastAPI endpoints for FaceID users and attendance records.

Important: POST /attendance should be called only by a trusted recognition
workflow after the face has been verified. A user_id by itself is not proof of
identity; do not expose this write endpoint to an untrusted network without
adding authentication/authorization.
"""

import logging
import sqlite3
from contextlib import asynccontextmanager
from datetime import date, datetime
from zoneinfo import ZoneInfo
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, PositiveInt

from attendance import mark_attendance
from config import TIMEZONE
from database import (
    get_all_attendance,
    get_user,
    get_users,
    init_db,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="FaceID API",
    description="API for registered users and daily attendance records.",
    version="1.1.0",
    lifespan=lifespan,
)


class UserResponse(BaseModel):
    id: int
    name: str
    created_at: str


class AttendanceRequest(BaseModel):
    user_id: PositiveInt


class AttendanceCreateResponse(BaseModel):
    created: bool
    user_id: int
    name: str
    message: str
    attendance_date: str


@app.get("/")
def root():
    return {"message": "FaceID API is running", "docs": "/docs"}


@app.get("/health")
def health():
    # Startup initializes the schema; a lightweight read verifies the DB is usable.
    from database import get_connection

    connection = get_connection()
    try:
        connection.execute("SELECT 1").fetchone()
    except sqlite3.Error as exc:
        logger.exception("FaceID database health check failed")
        raise HTTPException(status_code=503, detail="Database unavailable") from exc
    finally:
        connection.close()
    return {"status": "ok", "database": "ok"}


@app.get("/users", response_model=List[UserResponse])
def users(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """List users without returning sensitive face embeddings."""
    return get_users(limit=limit, offset=offset)


@app.get("/users/{user_id}", response_model=UserResponse)
def get_single_user(user_id: int):
    user = get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.get("/attendance")
def attendance(
    user_id: Optional[PositiveInt] = None,
    start_date: Optional[date] = Query(default=None, description="Inclusive start date: YYYY-MM-DD"),
    end_date: Optional[date] = Query(default=None, description="Inclusive end date: YYYY-MM-DD"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """List attendance records, optionally filtered by user and date range."""
    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=422,
            detail="start_date must be earlier than or equal to end_date",
        )

    return get_all_attendance(
        limit=limit,
        offset=offset,
        user_id=user_id,
        start_date=start_date.isoformat() if start_date else None,
        end_date=end_date.isoformat() if end_date else None,
    )


@app.post("/attendance", response_model=AttendanceCreateResponse)
def create_attendance(request: AttendanceRequest):
    """Create a daily attendance record after successful face recognition.

    Duplicate requests are idempotent: they return created=False and do not
    create a second row for the same user/date.
    """
    user = get_user(request.user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        created = mark_attendance(request.user_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="User not found") from exc
    except (ValueError, sqlite3.Error) as exc:
        logger.exception("Could not create attendance record")
        raise HTTPException(status_code=500, detail="Could not create attendance record") from exc

    today = datetime.now(ZoneInfo(TIMEZONE)).date().isoformat()
    records = get_all_attendance(
        user_id=request.user_id, start_date=today, end_date=today, limit=1
    )
    attendance_date = records[0]["attendance_date"] if records else today

    return {
        "created": created,
        "user_id": user["id"],
        "name": user["name"],
        "message": "Attendance recorded" if created else "Already marked for today",
        "attendance_date": attendance_date,
    }
