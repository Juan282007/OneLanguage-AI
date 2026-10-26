from collections import deque

import numpy as np
import pytest

import web_service


VISIBLE_FEATURES = np.ones(252, dtype=np.float32)
NO_HANDS = np.zeros(252, dtype=np.float32)


class FakePreprocessor:
    def __init__(self, **kwargs):
        self.closed = False

    def prepare_sequence(self, frames, length):
        return np.asarray(frames)

    def close(self):
        self.closed = True


class FakeRuntime:
    labels = ["hola", "__idle__"]
    config = {"uses_wrist_trajectory": False}
    include_motion_features = False
    sequence_length = 32
    live_minimum_frames = 2
    stable_predictions = 2
    confidence_threshold = 0.7
    minimum_margin = 0.12

    def __init__(self, predictions=None):
        self.predictions = deque(predictions or [])

    def predict(self, sequence):
        return self.predictions.popleft() if self.predictions else np.array([0.95, 0.05])


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setattr(web_service, "Preprocessor", FakePreprocessor)
    monkeypatch.setattr(web_service, "select_live_action_window", lambda frames, length: frames)
    instance = web_service.RecognitionSession(FakeRuntime())
    yield instance
    instance.close()
    assert instance.extractor.closed


def test_translation_is_committed_once_until_hands_disappear(session):
    assert session.process_features(VISIBLE_FEATURES)["status"] == "analyzing"
    assert session.process_features(VISIBLE_FEATURES)["status"] == "analyzing"

    translated = session.process_features(VISIBLE_FEATURES)
    assert translated["status"] == "translated"
    assert translated["label"] == "hola"
    assert translated["is_new_translation"] is True
    assert session.process_features(VISIBLE_FEATURES)["is_new_translation"] is False

    for _ in range(web_service.MISSING_HAND_RESET_FRAMES - 1):
        assert session.process_features(NO_HANDS)["status"] == "analyzing"
    assert session.process_features(NO_HANDS)["status"] == "no_hands"
    assert session.process_features(VISIBLE_FEATURES)["status"] == "analyzing"


def test_neutral_prediction_resets_session(session):
    session.runtime.predictions.append(np.array([0.05, 0.95]))
    assert session.process_features(VISIBLE_FEATURES)["status"] == "analyzing"
    assert session.process_features(VISIBLE_FEATURES)["status"] == "idle"
    assert session.visible_hand_frames == 0
    assert session.committed_label is None


def test_low_confidence_clears_stability(session):
    session.runtime.predictions.extend(
        [np.array([0.95, 0.05]), np.array([0.55, 0.45]), np.array([0.95, 0.05])]
    )
    session.process_features(VISIBLE_FEATURES)
    assert session.process_features(VISIBLE_FEATURES)["status"] == "analyzing"
    assert session.process_features(VISIBLE_FEATURES)["status"] == "waiting"
    assert session.process_features(VISIBLE_FEATURES)["status"] == "analyzing"


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"invalid-jpeg", id="invalid-jpeg"),
        pytest.param(b"x" * (web_service.MAX_FRAME_BYTES + 1), id="too-large"),
    ],
)
def test_invalid_encoded_frames_are_rejected(session, payload):
    with pytest.raises(ValueError):
        session.process_encoded_frame(payload)
