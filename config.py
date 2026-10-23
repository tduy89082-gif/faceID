"""Project configuration.

Values can be overridden with environment variables, so deployments do not need
source-code edits for a camera URL or database location.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

PHONE_CAMERA_URL = os.getenv(
    "FACEID_CAMERA_URL",
    "http://192.168.1.11:8080/video",
)
DATABASE_PATH = os.getenv(
    "FACEID_DATABASE_PATH",
    str(BASE_DIR / "faceid.db"),
)
TIMEZONE = os.getenv("FACEID_TIMEZONE", "Asia/Ho_Chi_Minh")

FACE_MODEL = os.getenv("FACEID_FACE_MODEL", "buffalo_l")
DETECTION_SIZE = (320, 320)
FACE_THRESHOLD = 0.50
REGISTRATION_SAMPLES = 20
