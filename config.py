"""Runtime configuration for the local Swanetra application.

Secrets are read from environment variables and the local .env file.
Copy .env.example to .env and fill in the values before starting the application.
"""

import os

from dotenv import load_dotenv


load_dotenv()


API_URL = os.getenv("SWANETRA_API_URL", "http://127.0.0.1:1234/v1/chat/completions")
AI_MODEL = os.getenv(
	"SWANETRA_AI_MODEL", "hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF"
)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

YOLO_MODEL_PATH = os.getenv("SWANETRA_YOLO_MODEL", "yolov8n.pt")
YOLO_CONFIDENCE = float(os.getenv("SWANETRA_YOLO_CONFIDENCE", "0.40"))
CAMERA_INDEXES = tuple(
	int(value.strip())
	for value in os.getenv("SWANETRA_CAMERA_INDEXES", "0,1").split(",")
	if value.strip().isdigit()
)
CAMERA_WIDTH = int(os.getenv("SWANETRA_CAMERA_WIDTH", "1280"))
CAMERA_HEIGHT = int(os.getenv("SWANETRA_CAMERA_HEIGHT", "720"))

# Distance uses a calibrated focal length and an assumed object width.
KNOWN_OBJECT_WIDTH_CM = float(os.getenv("SWANETRA_KNOWN_WIDTH_CM", "14"))
FOCAL_LENGTH = float(os.getenv("SWANETRA_FOCAL_LENGTH", "600"))

SPEECH_REPEAT_SECONDS = float(os.getenv("SWANETRA_SPEECH_REPEAT_SECONDS", "12"))
ALERT_COOLDOWN_SECONDS = float(os.getenv("SWANETRA_ALERT_COOLDOWN_SECONDS", "60"))

# COCO classes with an immediate safety implication. Add project-specific
# classes here rather than scattering danger rules through the camera loop.
DANGER_CLASSES = frozenset(
	value.strip().lower()
	for value in os.getenv(
		"SWANETRA_DANGER_CLASSES",
		"car,truck,bus,motorcycle,bicycle",
	).split(",")
	if value.strip()
)

# Backwards-compatible names used by older code.
bot_token = TELEGRAM_BOT_TOKEN
chat_id = TELEGRAM_CHAT_ID
known_width = KNOWN_OBJECT_WIDTH_CM
focal_length = FOCAL_LENGTH
