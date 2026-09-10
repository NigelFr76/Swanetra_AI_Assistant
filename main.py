import cv2
import json
import os
import queue
import shutil
import subprocess
import threading
import time
import warnings
import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import requests
import speech_recognition as sr
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor
from ultralytics import YOLO

import pyttsx3
from config import (
    AI_MODEL,
    ALERT_COOLDOWN_SECONDS,
    API_URL,
    CAMERA_HEIGHT,
    CAMERA_INDEXES,
    CAMERA_WIDTH,
    DANGER_CLASSES,
    FOCAL_LENGTH,
    KNOWN_OBJECT_WIDTH_CM,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    YOLO_CONFIDENCE,
    YOLO_MODEL_PATH,
)

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter(action='ignore', category=FutureWarning)

# Global variables
latest_frame = None
latest_detections = []
lock = threading.Lock()
running = True
engine = None
yolo_model = None
processor = None
blip_model = None
device = None
cap = None
speech_queue = queue.Queue()
alert_queue = queue.Queue()
speech_engine_ready = threading.Event()
recognizer = sr.Recognizer()
microphone_calibrated = False
last_queued_speech = ""
last_queued_speech_time = 0.0


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]
    distance_cm: Optional[float] = None

    @property
    def is_dangerous(self):
        return self.label.lower() in DANGER_CLASSES


def extract_detections(results, frame_width):
    """Convert an Ultralytics result into simple, testable detection data."""
    detections = []
    names = results.names
    for box in results.boxes:
        confidence = float(box.conf.item())
        if confidence < YOLO_CONFIDENCE:
            continue
        class_id = int(box.cls.item())
        label = str(names[class_id])
        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
        pixel_width = x2 - x1
        distance_cm = None
        if pixel_width > 0:
            distance_cm = (KNOWN_OBJECT_WIDTH_CM * FOCAL_LENGTH) / pixel_width
        detections.append(Detection(label, confidence, (x1, y1, x2, y2), distance_cm))
    return detections


def format_detection(detection):
    text = f"{detection.label} {detection.confidence:.0%}"
    if detection.distance_cm is not None:
        text += f" | ~{detection.distance_cm / 100:.1f}m"
    return text


def draw_overlay(frame, detections):
    """Draw a readable, stable overlay instead of relying on default plotting."""
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        color = (0, 0, 255) if detection.is_dangerous else (40, 210, 80)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        label = format_detection(detection).upper()
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        text_y = max(y1 - 10, text_height + 8)
        cv2.rectangle(frame, (x1, text_y - text_height - 8),
                      (x1 + text_width + 8, text_y + 4), color, -1)
        cv2.putText(frame, label, (x1 + 4, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.rectangle(frame, (0, 0), (390, 78), (18, 24, 32), -1)
    cv2.putText(frame, "SWANETRA VISION", (16, 29), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Objects: {len(detections)}   Status: Monitoring",
                (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (120, 220, 255), 1,
                cv2.LINE_AA)
    return frame

# ====================== TTS Setup ======================
def speak(text, priority=False):
    """Queue speech without blocking camera or microphone processing."""
    global last_queued_speech, last_queued_speech_time
    if not text:
        return False
    text = str(text)
    now = time.monotonic()
    if not priority and text == last_queued_speech and now - last_queued_speech_time < 15:
        return False
    if priority:
        while True:
            try:
                _, old_done = speech_queue.get_nowait()
                old_done.set()
            except queue.Empty:
                break
    done = threading.Event()
    speech_queue.put((text, done))
    last_queued_speech = text
    last_queued_speech_time = now
    return done


def speech_worker():
    global engine
    native_say = shutil.which("say")

    try:
        speech_engine_ready.set()
        log_event("TTS", "Ready")
    except Exception as error:
        log_event("TTS", f"ERROR: unavailable: {error}")
        return

    while running:
        try:
            text, done = speech_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            log_event("TTS", f"Speaking: {text}")

            if native_say:
                subprocess.run(
                    [native_say, "-v", "Samantha", text],
                    check=True
                )
            else:
                # Create a fresh Windows SAPI engine for each utterance
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                engine = None

        except Exception as error:
            log_event("TTS", f"ERROR: {error}")

        finally:
            done.set()

# ====================== LM Studio (Llama 3) ======================
def get_ai_response(prompt):
    try:
        payload = {
            "model": AI_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=15)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except (requests.RequestException, KeyError, ValueError, TypeError) as error:
        print(f"AI service unavailable: {error}", flush=True)
        return "I'm sorry, I couldn't process that."


def log_event(component, message):
    print(f"[{component}] {message}", flush=True)

def initialize_models():
    """Load local vision models only when the application is started."""
    global yolo_model, processor, blip_model, device
    print("Loading YOLO...", flush=True)
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
    except Exception as error:
        print(f"Unable to load YOLO model: {error}", flush=True)
        yolo_model = None
        return False
    yolo_model.verbose = False
    try:
        print("Loading BLIP processor and model...", flush=True)
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
    except Exception as error:
        print(f"BLIP unavailable; object detection will continue: {error}", flush=True)
        processor = None
        blip_model = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if blip_model is not None:
        blip_model.to(device)
        print(f"Vision models ready (BLIP device: {device}).", flush=True)
    else:
        print("YOLO ready; scene captioning is disabled.", flush=True)
    return True


def initialize_camera():
    global cap
    for camera_index in CAMERA_INDEXES:
        candidate = cv2.VideoCapture(camera_index)
        if candidate.isOpened():
            candidate.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            candidate.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            cap = candidate
            print(f"Camera opened at index {camera_index}.", flush=True)
            return True
        candidate.release()
    cap = None
    print("Unable to access camera. Check camera permissions in System Settings > Privacy & Security > Camera.", flush=True)
    return False

# ====================== Speech Recognition ======================
def list_microphones():
    try:
        names = sr.Microphone.list_microphone_names() or []
        for index, name in enumerate(names):
            print(f"Mic {index}: {name}")
    except Exception as e:
        print(f"Microphone enumeration error: {e}")


def recognize_speech():
    global microphone_calibrated
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.6
    device_index_env = os.getenv("MIC_DEVICE_INDEX")
    mic_kwargs = {}
    if device_index_env is not None:
        try:
            mic_kwargs["device_index"] = int(device_index_env)
        except ValueError:
            print("Invalid MIC_DEVICE_INDEX; ignoring.")
    try:
        with sr.Microphone(**mic_kwargs) as source:
            if not microphone_calibrated:
                log_event("SPEECH", "Calibrating microphone for ambient noise")
                recognizer.adjust_for_ambient_noise(source, duration=1)
                microphone_calibrated = True
            log_event("SPEECH", "Listening")
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=6)
        try:
            text = recognizer.recognize_google(audio)
            return text.lower().strip()
        except sr.UnknownValueError:
            log_event("SPEECH", "I couldn't understand that. Please try again.")
            return None
        except sr.RequestError as e:
            log_event("SPEECH", f"Recognition service error: {e}")
            return None
    except sr.WaitTimeoutError:
        log_event("SPEECH", "No speech detected before timeout")
        return None
    except AttributeError as e:
        # Often indicates missing PyAudio or permissions
        log_event("SPEECH", f"Microphone unavailable: {e}. Grant Microphone permission to Terminal/VS Code.")
        list_microphones()
        return None
    except OSError as e:
        # No default input device or permission denied
        log_event("SPEECH", f"Microphone error: {e}. Try MIC_DEVICE_INDEX or macOS Microphone permission.")
        list_microphones()
        return None
# ====================== Object and Scene Description ======================
def describe_objects_and_scene(frame):
    if yolo_model is None:
        return "Vision models are not ready yet."
    try:
        results = yolo_model(frame, verbose=False)[0]
        detections = extract_detections(results, frame.shape[1])
        descriptions = [format_detection(detection) for detection in detections]

        if processor is None or blip_model is None or device is None:
            return "I see " + ", ".join(descriptions) if descriptions else "I do not see a known object."

        print("Generating scene caption...", flush=True)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(image, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = blip_model.generate(**inputs, max_new_tokens=30)  # pyright: ignore[reportArgumentType]
        caption = processor.decode(out[0], skip_special_tokens=True)

        if descriptions:
            return "I see " + ", ".join(descriptions) + f". It looks like {caption}."
        return f"It looks like {caption}."
    except Exception as error:
        print(f"Vision processing error: {error}", flush=True)
        return "I couldn't process the camera image right now."

# ====================== Danger and Telegram Alerts ======================
last_alert_times = {}
last_speech_signature = None
last_speech_time = 0.0
candidate_detection_signature = None
candidate_detection_frames = 0
empty_detection_frames = 0


def danger_message(detections):
    dangerous = [detection for detection in detections if detection.is_dangerous]
    if not dangerous:
        return None
    return "Warning. " + ", ".join(
        f"{detection.label} detected nearby" for detection in dangerous
    ) + "."


def detection_message(detections):
    if not detections:
        return None
    danger = danger_message(detections)
    if danger:
        return danger
    labels = ", ".join(sorted({detection.label for detection in detections}))
    return f"{labels} detected ahead."


def should_announce(detections, now=None):
    """Return an announcement only for a new or meaningfully changed scene."""
    global last_speech_signature, last_speech_time
    now = time.monotonic() if now is None else now
    if not detections:
        last_speech_signature = None
        return False
    signature = tuple(sorted(
        detection.label
        for detection in detections
    ))
    if signature == last_speech_signature and now - last_speech_time < 12:
        return False
    last_speech_signature = signature
    last_speech_time = now
    return True


def stable_detection_announcement(detections):
    """Announce only a stable label change, not per-frame confidence noise."""
    global candidate_detection_signature, candidate_detection_frames
    global empty_detection_frames, last_speech_signature, last_speech_time
    signature = tuple(sorted({detection.label for detection in detections}))
    if not signature:
        empty_detection_frames += 1
        if empty_detection_frames >= 5:
            candidate_detection_signature = None
            candidate_detection_frames = 0
            last_speech_signature = None
        return None

    empty_detection_frames = 0
    if signature != candidate_detection_signature:
        candidate_detection_signature = signature
        candidate_detection_frames = 1
        return None
    candidate_detection_frames += 1
    if candidate_detection_frames < 5:
        return None

    if signature == last_speech_signature:
        return None
    last_speech_signature = signature
    last_speech_time = time.monotonic()
    return detection_message(detections)


def can_send_alert(labels, now=None):
    now = time.monotonic() if now is None else now
    key = ",".join(sorted(labels))
    previous = last_alert_times.get(key)
    if previous is not None and now - previous < ALERT_COOLDOWN_SECONDS:
        return False
    last_alert_times[key] = now
    return True


def current_detection_summary(detections=None):
    detections = latest_detections if detections is None else detections
    if not detections:
        return "None detected"
    return ", ".join(format_detection(detection) for detection in detections)


def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log_event("TELEGRAM", "ERROR: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing")
        return False
    if TELEGRAM_CHAT_ID.lower().endswith("bot"):
        log_event("TELEGRAM", "ERROR: TELEGRAM_CHAT_ID looks like a bot username; use the numeric destination chat ID")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=payload, timeout=10)
    except requests.RequestException as error:
        safe_error = str(error).replace(TELEGRAM_BOT_TOKEN, "<redacted>")
        log_event("TELEGRAM", f"ERROR: {safe_error}")
        return False
    try:
        data = response.json()
    except ValueError:
        log_event("TELEGRAM", f"ERROR: HTTP {response.status_code} with invalid JSON response")
        return False
    status_code = getattr(response, "status_code", 200)
    if status_code >= 400:
        description = data.get("description", f"HTTP {status_code}")
        log_event("TELEGRAM", f"ERROR: {description}")
        return False
    try:
        if data.get("ok") is not True:
            description = data.get("description", "Telegram rejected the message")
            log_event("TELEGRAM", f"ERROR: {description}")
            return False
        log_event("TELEGRAM", "Alert sent successfully")
        return True
    except AttributeError:
        log_event("TELEGRAM", "ERROR: Telegram returned an unexpected response")
        return False


def send_telegram_alert(message):
    return send_telegram_message(message)


def alert_worker():
    while running:
        try:
            message = alert_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        send_telegram_alert(message)


def trigger_emergency(reason, context="", detections=None):
    """Queue one emergency notification and immediately acknowledge it locally."""
    detections = latest_detections if detections is None else detections
    if not can_send_alert(["emergency"]):
        log_event("EMERGENCY", "Alert suppressed by cooldown")
        return False
    log_event("EMERGENCY", reason)
    speak("Emergency mode activated. Sending an alert.")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = (
        "🚨 SWANETRA EMERGENCY ALERT\n\n"
        f"User-triggered emergency.\n\n"
        f"Reason: {reason}\n"
        f"{context}\n"
        f"Time: {timestamp}\n\n"
        f"Current detected objects: {current_detection_summary(detections)}\n"
        "Vision status: Monitoring"
    )
    alert_queue.put(message)
    return True


def queue_danger_alerts(detections):
    dangerous = [detection for detection in detections if detection.is_dangerous]
    if not dangerous:
        return
    details = []
    for detection in dangerous:
        distance = (
            f"~{detection.distance_cm / 100:.1f}m"
            if detection.distance_cm is not None else "unavailable"
        )
        details.append(
            f"Object: {detection.label}\n"
            f"Confidence: {detection.confidence:.0%}\n"
            f"Distance: {distance}"
        )
    trigger_emergency(
        "Dangerous object detected by YOLO.",
        "\n".join(details),
        detections,
    )

# ====================== Location from IP ======================
def get_ip_location():
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        loc = data["loc"].split(",")
        return float(loc[0]), float(loc[1])
    except (requests.RequestException, KeyError, ValueError):
        return None, None

# ====================== Object Detection Loop ======================
def object_detection_loop():
    global latest_frame, latest_detections, running
    if cap is None or yolo_model is None:
        print("Vision loop unavailable: camera or YOLO model is not ready.", flush=True)
        return
    while running:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Camera frame could not be read.", flush=True)
                continue
            results = yolo_model(frame, verbose=False)[0]
            detections = extract_detections(results, frame.shape[1])
            with lock:
                latest_frame = frame.copy()
                latest_detections = detections
            announcement = stable_detection_announcement(detections)
            if announcement:
                speak(announcement)
            queue_danger_alerts(detections)
            annotated = draw_overlay(frame, detections)
            cv2.imshow("Swanetra Vision", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                running = False
                break
        except cv2.error as error:
            print(f"OpenCV preview error: {error}", flush=True)
            running = False
            break
        except Exception as error:
            print(f"Vision loop error: {error}", flush=True)
            time.sleep(0.1)

# ====================== Voice Loop ======================
def is_scene_query(command):
    normalized = command.lower().replace("'", "").replace("?", "")
    normalized = normalized.replace("whats", "what is")
    return any(phrase in normalized for phrase in [
        "what is in front", "what do you see", "what do i see",
        "describe my surroundings", "describe what you see", "what is around me",
        "mere saamne kya hai", "mere samne kya hai",
        "kya hai mere saamne", "kya hai mere samne",
        "aage kya hai", "kya dikh rha hai",
        "kya dikh rahe hai", "kya dikh rahe", "yeh kya hai"
    ])


def is_emergency_command(command):
    normalized = command.lower().replace("'", "")
    return any(phrase in normalized for phrase in [
        "i am in danger", "im in danger", "i'm in danger",
        "help me", "i need help", "emergency", "call for help",
    ])


def voice_loop():
    global running
    while running:
        try:
            command = recognize_speech()
            if not command:
                continue
            print(f"You said: {command}", flush=True)

            if is_emergency_command(command):
                trigger_emergency("User said: " + command, context="User requested help.")
                response = None

            elif is_scene_query(command):
                with lock:
                    frame = latest_frame.copy() if latest_frame is not None else None
                if frame is not None:
                    print("Processing the latest camera frame...", flush=True)
                    response = describe_objects_and_scene(frame)
                else:
                    print("Warning: no camera frame is available yet.", flush=True)
                    response = "I can't see anything right now."

            elif "exit" in command or "quit" in command:
                response = "Goodbye!"
                running = False
                break

            else:
                response = get_ai_response(command)

            if response:
                print(f"AI: {response}", flush=True)
                speak(response)
        except Exception as error:
            print(f"Voice command error: {error}", flush=True)
            speak("I could not process that request.")


def run_diagnostics():
    """Run checks without starting the camera or voice loops."""
    log_event("DIAGNOSE", f"Python: {os.sys.executable}")
    model_ok = False
    if os.path.isfile(YOLO_MODEL_PATH):
        try:
            YOLO(YOLO_MODEL_PATH)
            model_ok = True
        except Exception as error:
            log_event("DIAGNOSE", f"YOLO load error: {error}")
    log_event("DIAGNOSE", f"YOLO model: {'OK' if model_ok else 'MISSING/INVALID'} ({YOLO_MODEL_PATH})")
    camera = cv2.VideoCapture(0)
    camera_ok = camera.isOpened()
    camera.release()
    log_event("DIAGNOSE", f"Camera index 0: {'OK' if camera_ok else 'UNAVAILABLE'}")
    try:
        microphone_count = len(sr.Microphone.list_microphone_names() or [])
        mic_ok = microphone_count > 0
    except Exception as error:
        microphone_count = 0
        mic_ok = False
        log_event("DIAGNOSE", f"Microphone error: {error}")
    log_event("DIAGNOSE", f"Microphones: {'OK' if mic_ok else 'UNAVAILABLE'} ({microphone_count})")
    try:
        test_engine = pyttsx3.init()
        test_engine.stop()
        tts_ok = True
    except Exception as error:
        tts_ok = False
        log_event("DIAGNOSE", f"TTS error: {error}")
    log_event("DIAGNOSE", f"TTS: {'OK' if tts_ok else 'UNAVAILABLE'}")
    telegram_configured = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
    if TELEGRAM_CHAT_ID.lower().endswith("bot"):
        log_event("DIAGNOSE", "Telegram: chat ID looks like a bot username; use a numeric destination ID")
        telegram_configured = False
    else:
        log_event("DIAGNOSE", f"Telegram configuration: {'OK' if telegram_configured else 'MISSING'}")
    if telegram_configured:
        try:
            response = requests.get(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe",
                timeout=10,
            )
            data = response.json()
            telegram_ok = response.status_code < 400 and data.get("ok") is True
            log_event("DIAGNOSE", f"Telegram connectivity: {'OK' if telegram_ok else data.get('description', 'FAILED')}")
        except (requests.RequestException, ValueError) as error:
            safe_error = str(error).replace(TELEGRAM_BOT_TOKEN, "<redacted>")
            log_event("DIAGNOSE", f"Telegram connectivity error: {safe_error}")
            telegram_ok = False
    else:
        telegram_ok = False
    return model_ok and telegram_configured and telegram_ok

# ====================== Start Threads ======================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnose", action="store_true", help="check local dependencies and configuration")
    args = parser.parse_args()
    if args.diagnose:
        raise SystemExit(0 if run_diagnostics() else 1)
    voice_thread = None
    try:
        if not initialize_models():
            raise RuntimeError("Unable to load YOLO model.")
        initialize_camera()
        threading.Thread(target=speech_worker, daemon=True).start()
        threading.Thread(target=alert_worker, daemon=True).start()
        voice_thread = threading.Thread(target=voice_loop, daemon=True)
        voice_thread.start()
        object_detection_loop()
    except KeyboardInterrupt:
        print("Program interrupted. Shutting down...")
        running = False
    finally:
        running = False
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()