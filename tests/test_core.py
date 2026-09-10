import time
from types import SimpleNamespace

import main


def fake_box(confidence, class_id, coordinates):
    return SimpleNamespace(
        conf=SimpleNamespace(item=lambda: confidence),
        cls=SimpleNamespace(item=lambda: class_id),
        xyxy=[SimpleNamespace(tolist=lambda: list(coordinates))],
    )


def test_extract_detections_filters_low_confidence_and_estimates_distance():
    results = SimpleNamespace(
        names={0: "person", 2: "car"},
        boxes=[
            fake_box(0.9, 0, (10, 10, 110, 210)),
            fake_box(0.2, 2, (0, 0, 100, 100)),
        ],
    )

    detections = main.extract_detections(results, 640)

    assert len(detections) == 1
    assert detections[0].label == "person"
    assert detections[0].distance_cm is not None


def test_danger_classification_and_message():
    detection = main.Detection("car", 0.91, (0, 0, 100, 100), 240)

    assert detection.is_dangerous
    assert "car detected nearby" in main.danger_message([detection])
    assert "car detected nearby" in main.detection_message([detection])


def test_normal_detection_message():
    detection = main.Detection("person", 0.92, (0, 0, 100, 100))

    assert main.detection_message([detection]) == "person detected ahead."


def test_scene_query_accepts_common_apostrophe_variants():
    assert main.is_scene_query("what's in front of me")
    assert main.is_scene_query("whats in front")
    assert main.is_scene_query("what is in front")
    assert main.is_scene_query("describe my surroundings")
    assert main.is_scene_query("what is around me")


def test_emergency_command_accepts_user_phrases():
    assert main.is_emergency_command("I am in danger")
    assert main.is_emergency_command("Help me please")
    assert main.is_emergency_command("Emergency")
    assert main.is_emergency_command("Call for help")


def test_telegram_response_must_contain_ok(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": False, "description": "chat not found"}

    monkeypatch.setattr(main, "TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setattr(main, "TELEGRAM_CHAT_ID", "123")
    monkeypatch.setattr(main.requests, "post", lambda *args, **kwargs: FakeResponse())

    assert not main.send_telegram_message("test")


def test_telegram_request_errors_redact_token(monkeypatch, capsys):
    token = "secret-token"
    monkeypatch.setattr(main, "TELEGRAM_BOT_TOKEN", token)
    monkeypatch.setattr(main, "TELEGRAM_CHAT_ID", "123")

    def fail_request(*args, **kwargs):
        raise main.requests.RequestException(f"failed for {token}")

    monkeypatch.setattr(main.requests, "post", fail_request)

    assert not main.send_telegram_message("test")
    assert token not in capsys.readouterr().out


def test_trigger_emergency_queues_context(monkeypatch):
    main.last_alert_times.clear()
    main.latest_detections = []
    spoken = []
    monkeypatch.setattr(main, "speak", spoken.append)
    while not main.alert_queue.empty():
        main.alert_queue.get_nowait()

    assert main.trigger_emergency("User said: I am in danger", "User requested help.")
    assert spoken == ["Emergency mode activated. Sending an alert."]
    message = main.alert_queue.get_nowait()
    assert "I am in danger" in message
    assert "None detected" in message


def test_alert_cooldown():
    main.last_alert_times.clear()

    assert main.can_send_alert(["car"], now=100)
    assert not main.can_send_alert(["car"], now=100 + main.ALERT_COOLDOWN_SECONDS - 1)
    assert main.can_send_alert(["car"], now=100 + main.ALERT_COOLDOWN_SECONDS + 1)


def test_speech_debounce():
    main.last_speech_signature = None
    main.last_speech_time = 0
    detection = main.Detection("car", 0.91, (0, 0, 100, 100))

    assert main.should_announce([detection], now=100)
    assert not main.should_announce([detection], now=101)
    assert main.should_announce([detection], now=113)


def test_stable_detection_does_not_repeat_unchanged_labels():
    main.candidate_detection_signature = None
    main.candidate_detection_frames = 0
    main.empty_detection_frames = 0
    main.last_speech_signature = None
    detection = main.Detection("bottle", 0.9, (0, 0, 100, 100))

    announcements = [
        main.stable_detection_announcement([detection]) for _ in range(100)
    ]

    assert sum(value is not None for value in announcements) == 1


def test_telegram_is_disabled_without_credentials(monkeypatch):
    monkeypatch.setattr(main, "TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setattr(main, "TELEGRAM_CHAT_ID", "")

    assert not main.send_telegram_alert("test")
