import argparse
import json
import threading
from abc import ABC, abstractmethod
from collections import deque
from queue import Empty, Queue
from typing import Dict, Sequence

import cv2
import joblib
import numpy as np
import tensorflow as tf

from lsc_pipeline import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MINIMUM_MARGIN,
    DEFAULT_STABLE_PREDICTIONS,
    FEATURE_SIZE,
    LABELS_PATH,
    MIN_VISIBLE_FRAMES,
    MODEL_PATH,
    MOTION_FEATURE_SIZE,
    SEQUENCE_LENGTH,
    Preprocessor,
    is_no_sign_label,
    load_model_config,
    open_video_capture,
    select_live_action_window,
)


class BaseSignRecognizer(ABC):
    """Stable contract for recognizers that consume temporal landmark sequences."""

    @abstractmethod
    def preprocess(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        """Converts raw landmark frames into one model-ready sequence."""

    @abstractmethod
    def predict(self, sequence: np.ndarray) -> Dict[str, float]:
        """Returns the probability of every known label for a prepared sequence."""

    @abstractmethod
    def postprocess(self, prediction: Dict[str, float]):
        """Applies confidence, ambiguity and temporal stability policies."""


class Speaker:
    """Runs pyttsx3 in one dedicated thread, which is required by Windows voices."""

    def __init__(self):
        self.enabled = False
        self.available = False
        self.error = None
        self.voice_name = None
        self._messages = Queue(maxsize=1)
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="lsc-speaker", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=3)

    def _select_spanish_voice(self, engine):
        for voice in engine.getProperty("voices"):
            languages = getattr(voice, "languages", []) or []
            description = " ".join(
                [str(getattr(voice, "id", "")), str(getattr(voice, "name", ""))]
                + [str(language) for language in languages]
            ).casefold()
            if any(token in description for token in ("spanish", "espanol", "es-", "es_")):
                engine.setProperty("voice", voice.id)
                return str(getattr(voice, "name", "Voz en espanol"))
        return None

    def _run(self):
        engine = None
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", 170)
            engine.setProperty("volume", 1.0)
            self.voice_name = self._select_spanish_voice(engine)
            self.available = True
        except Exception as error:
            self.error = str(error)
        finally:
            self._ready.set()

        if engine is None:
            return

        while True:
            text = self._messages.get()
            if text is None:
                return
            try:
                engine.stop()
                engine.say(text)
                engine.runAndWait()
            except Exception as error:
                self.error = str(error)
                self.available = False

    def say(self, text):
        if not self.enabled or not self.available or not text:
            return False

        try:
            while True:
                self._messages.get_nowait()
        except Empty:
            pass

        try:
            self._messages.put_nowait(text)
            return True
        except Exception:
            return False

    def close(self):
        if not self._thread.is_alive():
            return

        try:
            while True:
                self._messages.get_nowait()
        except Empty:
            pass

        try:
            self._messages.put_nowait(None)
        except Exception:
            return
        self._thread.join(timeout=0.5)


class SignRecognizer(BaseSignRecognizer):
    MISSING_HAND_RESET_FRAMES = 3

    def __init__(
        self,
        source=0,
        speak=True,
        confidence_threshold=None,
        width=1280,
        height=720,
        fullscreen=False,
        auto_speak=True,
        minimum_margin=None,
        stable_predictions=None,
    ):
        self.source = int(source) if str(source).isdigit() else source
        self.is_camera = isinstance(self.source, int)
        self.window_name = "Traductor LSC"
        self.window_width = max(640, int(width))
        self.window_height = max(480, int(height))
        self.fullscreen = fullscreen
        self.auto_speak = auto_speak
        self.speaker = Speaker()
        self.speaker.enabled = bool(speak and self.speaker.available)
        self.sequence = deque(maxlen=SEQUENCE_LENGTH * 2)
        self.committed_label = None
        self.missing_hand_frames = 0
        self.visible_hand_frames = 0

        if MODEL_PATH.exists() and LABELS_PATH.exists():
            self.mode = "sequence"
            self.model = tf.keras.models.load_model(MODEL_PATH)
            self.labels = self._load_sequence_labels()
            self.model_config = self._load_sequence_config()
            self.sequence_length = int(self.model_config.get("sequence_length", SEQUENCE_LENGTH))
            uses_wrist_trajectory = bool(self.model_config.get("uses_wrist_trajectory", False))
            self.include_motion_features = bool(self.model_config.get("include_motion_features", False))
            self.extractor = Preprocessor(
                include_wrist_trajectory=uses_wrist_trajectory,
                include_motion_features=self.include_motion_features,
            )
            self.sequence = deque(maxlen=self.sequence_length * 2)
            self._validate_model_contract()
            print("Cargando modelo temporal entrenado con videos...")
            if not uses_wrist_trajectory:
                print("Modelo anterior detectado. Reentrena para usar trayectoria de munecas y ventanas activas.")
        else:
            self.mode = "legacy"
            print("Cargando modelo anterior. Entrena con train_from_videos.py para mejorar movimientos.")
            self.model = tf.keras.models.load_model("model/sign_language_recognition.keras", compile=False)
            self.extractor = Preprocessor(include_wrist_trajectory=False)
            self.scaler = joblib.load("model/scaler.pkl")
            self.label_encoder = joblib.load("model/label_encoder.pkl")
            with open("model/feature_order.json", encoding="utf-8") as file:
                self.feature_order = json.load(file)

        configured_threshold = self.model_config.get("confidence_threshold") if self.mode == "sequence" else None
        configured_margin = self.model_config.get("minimum_margin") if self.mode == "sequence" else None
        configured_stability = self.model_config.get("stable_predictions") if self.mode == "sequence" else None
        self.confidence_threshold = float(
            configured_threshold if confidence_threshold is None and configured_threshold is not None
            else DEFAULT_CONFIDENCE_THRESHOLD if confidence_threshold is None
            else confidence_threshold
        )
        self.minimum_margin = max(
            0.0,
            float(
                configured_margin if minimum_margin is None and configured_margin is not None
                else DEFAULT_MINIMUM_MARGIN if minimum_margin is None
                else minimum_margin
            ),
        )
        self.stable_predictions = max(
            1,
            int(
                configured_stability if stable_predictions is None and configured_stability is not None
                else DEFAULT_STABLE_PREDICTIONS if stable_predictions is None
                else stable_predictions
            ),
        )
        self.predictions = deque(maxlen=self.stable_predictions)

        self.cap = open_video_capture(self.source)
        if self.is_camera:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.window_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.window_height)
        if not self.cap.isOpened():
            self.extractor.close()
            self.speaker.close()
            raise RuntimeError(f"No pude abrir la fuente de video: {self.source}")

        if self.speaker.available:
            voice = f" ({self.speaker.voice_name})" if self.speaker.voice_name else ""
            print(f"Voz lista{voice}. ESPACIO repite la traduccion.")
        else:
            print(f"Voz no disponible: {self.speaker.error or 'pyttsx3 no pudo iniciarse'}")

    def _load_sequence_labels(self):
        with LABELS_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _load_sequence_config():
        return load_model_config()

    def _validate_model_contract(self):
        input_shape = self.model.input_shape
        expected_sequence_length = int(input_shape[1])
        expected_feature_size = int(input_shape[2])
        configured_feature_size = int(self.model_config.get("feature_size", expected_feature_size))
        preprocessing_feature_size = FEATURE_SIZE + (MOTION_FEATURE_SIZE if self.include_motion_features else 0)

        if self.sequence_length != expected_sequence_length:
            raise RuntimeError(
                "El modelo y lsc_config.json usan longitudes de secuencia diferentes. "
                "Restaura archivos de la misma version o reentrena el modelo."
            )
        if configured_feature_size != expected_feature_size or preprocessing_feature_size != expected_feature_size:
            raise RuntimeError(
                "El modelo y el preprocesamiento usan cantidades de features diferentes. "
                "Restaura lsc_config.json o reentrena para generar un modelo compatible."
            )
        if int(self.model.output_shape[-1]) != len(self.labels):
            raise RuntimeError(
                "El numero de salidas del modelo no coincide con lsc_labels.json. "
                "Restaura archivos de la misma version o reentrena el modelo."
            )

    def get_legacy_landmarks(self, frame):
        features = self.extractor.frame_features(frame)[:15].tolist()
        while len(features) < len(self.feature_order):
            features.append(0)
        return np.array(features[: len(self.feature_order)])

    @staticmethod
    def _hands_visible(features):
        return bool(np.any(np.abs(features) > 1e-5))

    @staticmethod
    def _is_rest_label(label):
        return is_no_sign_label(label)

    @staticmethod
    def _display_label(label):
        """Converts dataset-safe folder labels into text suitable for people."""
        return str(label).replace("-", " ").replace("_", " ").strip()

    @staticmethod
    def _prediction_margin(prediction):
        if len(prediction) < 2:
            return 1.0
        top_two = np.partition(prediction, -2)[-2:]
        return float(top_two[1] - top_two[0])

    def _reset_recognition(self):
        self.sequence.clear()
        self.predictions.clear()
        self.committed_label = None
        self.visible_hand_frames = 0

    def preprocess(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        if self.mode == "legacy":
            return np.asarray(frames, dtype=np.float32)
        return self.extractor.prepare_sequence(frames, self.sequence_length)

    def predict(self, sequence: np.ndarray) -> Dict[str, float]:
        if self.mode == "legacy":
            frame = np.asarray(sequence)
            features = self.get_legacy_landmarks(frame)
            X = self.scaler.transform([features])
            probabilities = self.model.predict(X, verbose=0)[0]
            labels = [str(label) for label in self.label_encoder.classes_]
        else:
            prepared = np.asarray(sequence, dtype=np.float32)
            if prepared.shape != (self.sequence_length, self.model.input_shape[2]):
                raise ValueError(
                    f"Expected sequence shape ({self.sequence_length}, {self.model.input_shape[2]}), "
                    f"received {prepared.shape}."
                )
            probabilities = self.model.predict(prepared[None, ...], verbose=0)[0]
            labels = self.labels

        return {label: float(probability) for label, probability in zip(labels, probabilities)}

    def _classify_legacy(self, frame):
        prediction = self.predict(frame)
        raw_label = max(prediction, key=prediction.get)
        label = self._display_label(raw_label)
        confidence = prediction[raw_label]
        if confidence < self.confidence_threshold or self._prediction_margin(np.fromiter(prediction.values(), dtype=np.float32)) < self.minimum_margin:
            return f"Detectando: {label}", confidence, False, False

        is_new_translation = raw_label != self.committed_label
        self.committed_label = raw_label
        return label, confidence, is_new_translation, True

    def postprocess(self, prediction: Dict[str, float]):
        raw_label = max(prediction, key=prediction.get)
        confidence = prediction[raw_label]
        margin = self._prediction_margin(np.fromiter(prediction.values(), dtype=np.float32))
        label = self._display_label(raw_label)

        if self._is_rest_label(raw_label):
            self._reset_recognition()
            return "Esperando sena...", confidence, False, False

        if confidence < self.confidence_threshold or margin < self.minimum_margin:
            self.predictions.clear()
            return "Esperando sena...", confidence, False, False

        self.predictions.append((raw_label, confidence))
        recent_labels = [recent_label for recent_label, _ in self.predictions]
        is_stable = len(recent_labels) == self.stable_predictions and len(set(recent_labels)) == 1
        if not is_stable:
            return f"Detectando: {label}", confidence, False, False

        stable_confidence = float(np.mean([value for _, value in self.predictions]))
        is_new_translation = raw_label != self.committed_label
        self.committed_label = raw_label
        return label, stable_confidence, is_new_translation, True

    def classify_frame(self, frame):
        if self.mode == "legacy":
            return self._classify_legacy(frame)

        features = self.extractor.frame_features(frame)
        if not self._hands_visible(features):
            self.missing_hand_frames += 1
            if self.missing_hand_frames >= self.MISSING_HAND_RESET_FRAMES:
                self._reset_recognition()
            return "Muestra las manos", 0.0, False, False

        self.missing_hand_frames = 0
        self.visible_hand_frames += 1
        self.sequence.append(features)
        if len(self.sequence) < self.sequence_length or self.visible_hand_frames < MIN_VISIBLE_FRAMES:
            return "Analizando...", 0.0, False, False

        active_window = select_live_action_window(list(self.sequence), self.sequence_length)
        sequence = self.preprocess(active_window)
        return self.postprocess(self.predict(sequence))

    def _configure_window(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_width, self.window_height)
        self._set_fullscreen(self.fullscreen)

    def _set_fullscreen(self, enabled):
        self.fullscreen = enabled
        mode = cv2.WINDOW_FULLSCREEN if enabled else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, mode)

    def _draw_overlay(self, frame, label, confidence, is_stable):
        if confidence:
            text = f"{label} ({confidence:.0%})"
        else:
            text = label

        color = (0, 180, 0) if is_stable else (0, 180, 255)
        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if self.speaker.enabled:
            voice_status = "Voz: activa"
        elif self.speaker.available:
            voice_status = "Voz: apagada"
        else:
            voice_status = "Voz: no disponible"
        auto_status = "auto: si" if self.auto_speak else "auto: no"
        cv2.putText(frame, f"{voice_status} | {auto_status}", (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "ESC salir | ESPACIO hablar | V voz | A auto | F pantalla completa", (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    def run(self):
        self._configure_window()
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                label, confidence, is_new_translation, is_stable = self.classify_frame(frame)
                if is_new_translation and self.auto_speak:
                    self.speaker.say(label)

                self._draw_overlay(frame, label, confidence, is_stable)
                cv2.imshow(self.window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                if key in (ord("v"), ord("V")):
                    if self.speaker.available:
                        self.speaker.enabled = not self.speaker.enabled
                    else:
                        print(f"La voz no esta disponible: {self.speaker.error or 'sin motor de voz'}")
                if key in (ord("a"), ord("A")):
                    self.auto_speak = not self.auto_speak
                if key in (ord("f"), ord("F")):
                    self._set_fullscreen(not self.fullscreen)
                if key == 32 and is_stable:
                    self.speaker.say(label)
        finally:
            self.cap.release()
            self.extractor.close()
            self.speaker.close()
            cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Reconoce Lengua de Senas Colombiana desde camara o video.")
    parser.add_argument("--source", default="0", help="0 para camara o ruta a un video")
    parser.add_argument("--no-speak", action="store_true", help="Desactiva la voz")
    parser.add_argument("--no-auto-speak", action="store_true", help="No reproduce automaticamente una traduccion estable")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Confianza minima. Si se omite, usa el valor guardado con el modelo.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=None,
        help="Diferencia minima entre las dos mejores clases. Por defecto usa la configuracion del modelo.",
    )
    parser.add_argument(
        "--stable-predictions",
        type=int,
        default=None,
        help="Predicciones consecutivas para confirmar. Por defecto usa la configuracion del modelo.",
    )
    parser.add_argument("--width", type=int, default=1280, help="Ancho deseado de la camara y la ventana")
    parser.add_argument("--height", type=int, default=720, help="Alto deseado de la camara y la ventana")
    parser.add_argument("--fullscreen", action="store_true", help="Inicia la camara en pantalla completa")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = SignRecognizer(
        source=args.source,
        speak=not args.no_speak,
        confidence_threshold=args.threshold,
        width=args.width,
        height=args.height,
        fullscreen=args.fullscreen,
        auto_speak=not args.no_auto_speak,
        minimum_margin=args.margin,
        stable_predictions=args.stable_predictions,
    )
    app.run()
