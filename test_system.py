"""Non-destructive local diagnostics for Swanetra."""

import importlib.util
import os
from pathlib import Path

import cv2

import main


def check(name, passed, detail=""):
    print(f"{'PASS' if passed else 'FAIL'} {name}: {detail}")
    return passed


def main_diagnostic():
    root = Path(__file__).resolve().parent
    results = []
    results.append(check("Python environment", True, os.sys.executable))
    for package in ("torch", "transformers", "ultralytics", "speech_recognition", "pyttsx3"):
        results.append(check(f"Package {package}", importlib.util.find_spec(package) is not None))
    results.append(check("YOLO model", Path(main.YOLO_MODEL_PATH).is_file(), main.YOLO_MODEL_PATH))
    camera = cv2.VideoCapture(0)
    camera_ok = camera.isOpened()
    camera.release()
    results.append(check("Camera availability", camera_ok, "index 0; may require macOS permission"))
    try:
        main.list_microphones()
        mic_ok = importlib.util.find_spec("pyaudio") is not None
    except Exception as error:
        mic_ok = False
        print(f"Microphone error: {error}")
    results.append(check("Microphone package", mic_ok, "grant macOS Microphone permission if unavailable"))
    try:
        import pyttsx3
        pyttsx3.init().stop()
        tts_ok = True
    except Exception as error:
        tts_ok = False
        print(f"TTS error: {error}")
    results.append(check("Text-to-speech", tts_ok))
    telegram_ok = bool(main.TELEGRAM_BOT_TOKEN and main.TELEGRAM_CHAT_ID)
    results.append(check("Telegram configuration", telegram_ok, "set both variables in .env"))
    return all(results)


if __name__ == "__main__":
    raise SystemExit(0 if main_diagnostic() else 1)