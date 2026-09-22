"""WebSocket service that exposes the local LSC model to browser clients."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware


PROJECT_DIR = Path(__file__).resolve().parent
os.chdir(PROJECT_DIR)

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv(PROJECT_DIR / ".env")

from lsc_pipeline import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MINIMUM_MARGIN,
    DEFAULT_STABLE_PREDICTIONS,
    FEATURE_SIZE,
    LABELS_PATH,
    MIN_VISIBLE_FRAMES,
    MODEL_PATH,
    MOTION_FEATURE_SIZE,
    Preprocessor,
    is_no_sign_label,
    load_model_config,
    select_live_action_window,
)


MAX_FRAME_BYTES = 1_500_000
MISSING_HAND_RESET_FRAMES = 3


def _allowed_origins():
    configured = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


def _display_label(label):
    return str(label).replace("-", " ").replace("_", " ").strip()


def _prediction_margin(probabilities):
    if len(probabilities) < 2:
        return 1.0
    top_two = np.partition(probabilities, -2)[-2:]
    return float(top_two[1] - top_two[0])


class ModelRuntime:
    """Loads one compatible model package and shares it across web sessions."""

    def __init__(self):
        if not MODEL_PATH.exists() or not LABELS_PATH.exists():
            raise RuntimeError("No encontre el modelo. Coloca el paquete compatible dentro de model/.")

        self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        self.labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        self.config = load_model_config()
        self.sequence_length = int(self.config.get("sequence_length", self.model.input_shape[1]))
        self.include_motion_features = bool(self.config.get("include_motion_features", False))
        self.confidence_threshold = float(self.config.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD))
        self.minimum_margin = float(self.config.get("minimum_margin", DEFAULT_MINIMUM_MARGIN))
        self.stable_predictions = max(1, int(self.config.get("stable_predictions", DEFAULT_STABLE_PREDICTIONS)))
        expected_features = FEATURE_SIZE + (MOTION_FEATURE_SIZE if self.include_motion_features else 0)

        if int(self.model.input_shape[1]) != self.sequence_length:
            raise RuntimeError("El modelo y lsc_config.json usan longitudes de secuencia diferentes.")
        if int(self.model.input_shape[2]) != expected_features:
            raise RuntimeError("El modelo y lsc_config.json usan cantidades de caracteristicas diferentes.")
        if int(self.model.output_shape[-1]) != len(self.labels):
            raise RuntimeError("El modelo y lsc_labels.json pertenecen a versiones diferentes.")

        self._predict_lock = threading.Lock()

    def predict(self, sequence):
        with self._predict_lock:
            return self.model.predict(sequence[None, ...], verbose=0)[0]

    def metadata(self):
        return {
            "model_version": self.config.get("model_version", "local"),
            "preprocessing_version": self.config.get("preprocessing_version"),
            "sequence_length": self.sequence_length,
            "labels": [_display_label(label) for label in self.labels if not is_no_sign_label(label)],
        }


class RecognitionSession:
    """Keeps temporal landmarks isolated for one browser WebSocket connection."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.extractor = Preprocessor(
            include_wrist_trajectory=bool(runtime.config.get("uses_wrist_trajectory", False)),
            include_motion_features=runtime.include_motion_features,
        )
        self.sequence = deque(maxlen=runtime.sequence_length * 2)
        self.predictions = deque(maxlen=runtime.stable_predictions)
        self.missing_hand_frames = 0
        self.visible_hand_frames = 0
        self.committed_label = None

    def close(self):
        self.extractor.close()

    def _reset(self):
        self.sequence.clear()
        self.predictions.clear()
        self.visible_hand_frames = 0
        self.committed_label = None

    def _response(self, status, confidence=0.0, label=None, is_new_translation=False):
        return {
            "type": "prediction",
            "status": status,
            "label": label,
            "text": _display_label(label) if label else None,
            "confidence": round(float(confidence), 4),
            "is_stable": status == "translated",
            "is_new_translation": is_new_translation,
        }

    def process_encoded_frame(self, payload):
        if not payload or len(payload) > MAX_FRAME_BYTES:
            raise ValueError("El frame recibido no tiene un tamano valido.")

        frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("No pude decodificar el frame de la camara.")
        return self.process_frame(frame)

    def process_frame(self, frame):
        features = self.extractor.frame_features(frame)
        if not np.any(np.abs(features) > 1e-5):
            self.missing_hand_frames += 1
            if self.missing_hand_frames >= MISSING_HAND_RESET_FRAMES:
                self._reset()
            return self._response("no_hands")

        self.missing_hand_frames = 0
        self.visible_hand_frames += 1
        self.sequence.append(features)
        if len(self.sequence) < self.runtime.sequence_length or self.visible_hand_frames < MIN_VISIBLE_FRAMES:
            return self._response("analyzing")

        active_window = select_live_action_window(list(self.sequence), self.runtime.sequence_length)
        sequence = self.extractor.prepare_sequence(active_window, self.runtime.sequence_length)
        probabilities = self.runtime.predict(sequence)
        label_index = int(np.argmax(probabilities))
        raw_label = self.runtime.labels[label_index]
        confidence = float(probabilities[label_index])
        margin = _prediction_margin(probabilities)

        if is_no_sign_label(raw_label):
            self._reset()
            return self._response("idle", confidence)
        if confidence < self.runtime.confidence_threshold or margin < self.runtime.minimum_margin:
            self.predictions.clear()
            return self._response("waiting", confidence)

        self.predictions.append(raw_label)
        is_stable = len(self.predictions) == self.runtime.stable_predictions and len(set(self.predictions)) == 1
        if not is_stable:
            return self._response("analyzing", confidence, raw_label)

        is_new_translation = raw_label != self.committed_label
        self.committed_label = raw_label
        return self._response("translated", confidence, raw_label, is_new_translation)


runtime = ModelRuntime()
allowed_origins = _allowed_origins()
app = FastAPI(title="OneLanguage LSC Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", **runtime.metadata()}


@app.get("/model-info")
def model_info():
    return runtime.metadata()


@app.websocket("/ws/recognize")
async def recognize(websocket: WebSocket):
    origin = websocket.headers.get("origin")
    if origin and "*" not in allowed_origins and origin not in allowed_origins:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    session = RecognitionSession(runtime)
    await websocket.send_json({"type": "ready", **runtime.metadata()})
    try:
        while True:
            payload = await websocket.receive_bytes()
            try:
                prediction = await asyncio.to_thread(session.process_encoded_frame, payload)
            except ValueError as error:
                prediction = {"type": "error", "message": str(error)}
            await websocket.send_json(prediction)
    except WebSocketDisconnect:
        pass
    finally:
        session.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "web_service:app",
        host=os.getenv("AI_HOST", "0.0.0.0"),
        port=int(os.getenv("AI_PORT", "8000")),
        reload=os.getenv("AI_RELOAD", "false").casefold() == "true",
    )
