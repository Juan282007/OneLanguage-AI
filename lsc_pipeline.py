import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


MAX_HANDS = 2
LANDMARKS_PER_HAND = 21
COORDINATES_PER_LANDMARK = 3
SEQUENCE_LENGTH = 32
FEATURE_SIZE = MAX_HANDS * LANDMARKS_PER_HAND * COORDINATES_PER_LANDMARK
PREPROCESSING_VERSION = 2
MODEL_PATH = Path("model/lsc_sequence_model.keras")
LABELS_PATH = Path("model/lsc_labels.json")
CONFIG_PATH = Path("model/lsc_config.json")


class LandmarkExtractor:
    def __init__(
        self,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        include_wrist_trajectory=True,
    ):
        self.include_wrist_trajectory = include_wrist_trajectory
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self):
        self.hands.close()

    def frame_features(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.hands.process(image)

        left = np.zeros((LANDMARKS_PER_HAND, COORDINATES_PER_LANDMARK), dtype=np.float32)
        right = np.zeros((LANDMARKS_PER_HAND, COORDINATES_PER_LANDMARK), dtype=np.float32)
        left_assigned = False
        right_assigned = False
        unassigned = []

        if not results.multi_hand_landmarks:
            return np.concatenate([left.flatten(), right.flatten()])

        handedness = results.multi_handedness or []
        for index, hand_landmarks in enumerate(results.multi_hand_landmarks[:MAX_HANDS]):
            label = None
            if index < len(handedness):
                label = handedness[index].classification[0].label

            raw_points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
            points = self._normalize_hand(raw_points)
            if self.include_wrist_trajectory:
                # Landmark 0 was always (0, 0, 0) after local normalization.
                # It now preserves the hand trajectory across the sequence.
                points[0] = self._normalize_wrist(raw_points[0])

            if label == "Left" and not left_assigned:
                left = points
                left_assigned = True
            elif label == "Right" and not right_assigned:
                right = points
                right_assigned = True
            else:
                unassigned.append(points)

        for points in unassigned:
            if not left_assigned:
                left = points
                left_assigned = True
            elif not right_assigned:
                right = points
                right_assigned = True

        return np.concatenate([left.flatten(), right.flatten()])

    @staticmethod
    def _normalize_hand(points):
        wrist = points[0].copy()
        points = points - wrist
        scale = np.max(np.linalg.norm(points[:, :2], axis=1))
        if scale > 1e-6:
            points = points / scale
        return points

    @staticmethod
    def _normalize_wrist(wrist):
        # Center x/y so horizontal mirroring only needs to invert x.
        return np.array([wrist[0] - 0.5, wrist[1] - 0.5, wrist[2]], dtype=np.float32)


def resample_sequence(sequence, target_length=SEQUENCE_LENGTH):
    sequence = np.asarray(sequence, dtype=np.float32)
    if len(sequence) == 0:
        return np.zeros((target_length, FEATURE_SIZE), dtype=np.float32)
    if len(sequence) == target_length:
        return sequence

    old_steps = np.linspace(0, 1, num=len(sequence))
    new_steps = np.linspace(0, 1, num=target_length)
    resized = np.empty((target_length, sequence.shape[1]), dtype=np.float32)
    for feature_index in range(sequence.shape[1]):
        resized[:, feature_index] = np.interp(new_steps, old_steps, sequence[:, feature_index])
    return resized


def read_video_features(video_path, extractor=None):
    owns_extractor = extractor is None
    extractor = extractor or LandmarkExtractor()
    capture = cv2.VideoCapture(str(video_path))
    frames = []

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(extractor.frame_features(frame))
    finally:
        capture.release()
        if owns_extractor:
            extractor.close()

    return np.asarray(frames, dtype=np.float32)


def _trim_to_visible_hands(sequence):
    if len(sequence) == 0:
        return sequence

    visible = np.any(np.abs(sequence) > 1e-6, axis=1)
    visible_indexes = np.flatnonzero(visible)
    if len(visible_indexes) == 0:
        return sequence
    return sequence[visible_indexes[0] : visible_indexes[-1] + 1]


def _window_starts(sequence, target_length, windows_per_video):
    if len(sequence) <= target_length:
        return [0]

    changes = np.linalg.norm(np.diff(sequence, axis=0), axis=1) / np.sqrt(sequence.shape[1])
    motion = np.concatenate([[0.0], changes])
    scores = np.convolve(motion, np.ones(target_length, dtype=np.float32), mode="valid")
    best_start = int(np.argmax(scores))
    maximum_start = len(sequence) - target_length
    stride = max(1, target_length // 6)
    offsets = [0]
    for multiplier in range(1, windows_per_video):
        direction = -1 if multiplier % 2 else 1
        offsets.append(direction * ((multiplier + 1) // 2) * stride)

    starts = []
    for offset in offsets:
        start = int(np.clip(best_start + offset, 0, maximum_start))
        if start not in starts:
            starts.append(start)
    return starts


def select_action_windows(sequence, target_length=SEQUENCE_LENGTH, windows_per_video=3):
    """Creates fixed-size high-motion windows equivalent to the live camera buffer."""
    sequence = _trim_to_visible_hands(np.asarray(sequence, dtype=np.float32))
    if len(sequence) <= target_length:
        return [resample_sequence(sequence, target_length)]
    return [sequence[start : start + target_length] for start in _window_starts(sequence, target_length, windows_per_video)]


def extract_video_sequences(video_path, extractor=None, sequence_length=SEQUENCE_LENGTH, windows_per_video=3):
    features = read_video_features(video_path, extractor)
    if len(features) == 0 or not np.any(np.abs(features) > 1e-6):
        return []
    return select_action_windows(features, sequence_length, windows_per_video)


def extract_video_sequence(video_path, extractor=None, sequence_length=SEQUENCE_LENGTH):
    """Compatibility helper that returns the most active window of a video."""
    windows = extract_video_sequences(video_path, extractor, sequence_length, windows_per_video=1)
    if not windows:
        return np.zeros((sequence_length, FEATURE_SIZE), dtype=np.float32)
    return windows[0]


def load_labels():
    with LABELS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_model_config():
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_metadata(labels, sequence_length=SEQUENCE_LENGTH, extra_metadata=None):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LABELS_PATH.open("w", encoding="utf-8") as file:
        json.dump(labels, file, ensure_ascii=False, indent=2)

    metadata = {
        "sequence_length": sequence_length,
        "feature_size": FEATURE_SIZE,
        "preprocessing_version": PREPROCESSING_VERSION,
        "uses_wrist_trajectory": True,
        "input_window": "active_32_frame_window",
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    with CONFIG_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
