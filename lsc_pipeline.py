import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


# BLOQUE: contrato comun de datos.
# Estas constantes describen exactamente la entrada de la red. Cada frame puede
# contener dos manos, cada mano tiene 21 landmarks y cada landmark posee x, y y
# z. Entrenamiento, evaluacion y servicio en vivo deben usar este mismo formato;
# modificarlo cambia el significado de los pesos ya entrenados y obliga a crear
# un modelo nuevo.
MAX_HANDS = 2
LANDMARKS_PER_HAND = 21
COORDINATES_PER_LANDMARK = 3
SEQUENCE_LENGTH = 32
FEATURE_SIZE = MAX_HANDS * LANDMARKS_PER_HAND * COORDINATES_PER_LANDMARK
MOTION_FEATURE_SIZE = FEATURE_SIZE
MIN_VISIBLE_FRAMES = 10
SMOOTHING_ALPHA = 0.65
PREPROCESSING_VERSION = 7
NO_SIGN_LABEL = "__idle__"
# BLOQUE: compatibilidad con paquetes historicos.
# `__idle__` se uso en una etapa anterior para representar ausencia de sena. Se
# retiro del modelo final porque podia confundirse con senas reales. Se conserva
# el reconocimiento de sus alias para que una version antigua pueda abrirse sin
# alterar sus etiquetas; no significa que la clase actual este activa.
NO_SIGN_ALIASES = frozenset(
    {
        NO_SIGN_LABEL,
        "background",
        "idle",
        "neutral",
        "ninguno",
        "no_sign",
        "reposo",
        "sin_sena",
        "sin_senas",
    }
)
DEFAULT_CONFIDENCE_THRESHOLD = 0.70
DEFAULT_MINIMUM_MARGIN = 0.12
DEFAULT_STABLE_PREDICTIONS = 4
DISPLAY_LABELS = {
    "bien": "Bien",
    "buenas-noches": "Buenas noches",
    "buenas-tardes": "Buenas tardes",
    "buenos-dias": "Buenos días",
    "como-estas": "Cómo estás",
    "el": "Él",
    "ella": "Ella",
    "gracias": "Gracias",
    "hola": "Hola",
    "mal": "Mal",
    "regular": "Regular",
    "tu": "Tú",
    "yo": "Yo",
}
# BLOQUE: artefactos persistidos del modelo.
# El archivo Keras contiene pesos y arquitectura; labels define que posicion de
# Softmax pertenece a cada sena; config conserva las reglas de preprocesamiento.
# Los tres archivos forman una sola unidad y no deben mezclarse entre versiones.
MODEL_PATH = Path("model/lsc_sequence_model.keras")
LABELS_PATH = Path("model/lsc_labels.json")
CONFIG_PATH = Path("model/lsc_config.json")


# BLOQUE: normalizacion de etiquetas.
# Las carpetas del dataset usan identificadores seguros para archivos, mientras
# la interfaz necesita texto legible. Estas funciones separan ambas necesidades:
# normalizan comparaciones internas sin cambiar la etiqueta que usa el modelo y
# convierten guiones o nombres especiales en texto para las personas.
def normalize_label_token(label):
    return str(label).casefold().strip().replace("-", "_").replace(" ", "_")


def is_no_sign_label(label):
    return normalize_label_token(label) in NO_SIGN_ALIASES


def canonicalize_label(label):
    """Maps supported neutral aliases to one stable model label."""
    return NO_SIGN_LABEL if is_no_sign_label(label) else str(label).strip()


def format_label(label):
    """Returns a dataset label as readable Spanish without changing its ID."""
    token = str(label).strip()
    if token in DISPLAY_LABELS:
        return DISPLAY_LABELS[token]

    text = token.replace("-", " ").replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else text


def open_video_capture(source):
    """Opens a camera or video and honors phone rotation metadata when available."""
    capture = cv2.VideoCapture(source)
    if capture.isOpened() and hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    return capture


class Preprocessor:
    """Single preprocessing contract shared by training and live recognition.

    Frames are represented as float32 hand landmarks with shape (126,). A model
    sequence always has a fixed number of frames and can optionally append
    per-landmark motion features, resulting in 252 features per frame.
    """

    def __init__(
        self,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        include_wrist_trajectory=True,
        include_motion_features=False,
    ):
        # BLOQUE: inicializacion de detectores.
        # MediaPipe Hands localiza dedos y munecas; MediaPipe Pose aporta los
        # hombros como referencia espacial cuando estan visibles. Se crean una
        # vez por sesion porque reconstruirlos para cada frame desperdiciaria
        # CPU y haria menos estable el seguimiento temporal.
        self.include_wrist_trajectory = include_wrist_trajectory
        self.include_motion_features = include_motion_features
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self):
        self.hands.close()
        self.pose.close()

    def frame_features(self, frame):
        # BLOQUE: imagen a vector numerico.
        # OpenCV entrega BGR y MediaPipe requiere RGB. Despues de detectar manos,
        # cada una se normaliza y se coloca en una posicion estable del vector.
        # La salida siempre mide 126 valores; una mano no detectada queda en cero
        # para que el modelo conozca su ausencia sin alterar la forma de entrada.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        body_reference = self._body_reference(self.pose.process(image))
        results = self.hands.process(image)

        left = np.zeros((LANDMARKS_PER_HAND, COORDINATES_PER_LANDMARK), dtype=np.float32)
        right = np.zeros((LANDMARKS_PER_HAND, COORDINATES_PER_LANDMARK), dtype=np.float32)

        if not results.multi_hand_landmarks:
            return np.concatenate([left.flatten(), right.flatten()])

        handedness = results.multi_handedness or []
        detected_hands = []
        for index, hand_landmarks in enumerate(results.multi_hand_landmarks[:MAX_HANDS]):
            label = None
            if index < len(handedness):
                label = handedness[index].classification[0].label

            raw_points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
            points = self._normalize_hand(raw_points)
            if self.include_wrist_trajectory:
                # Landmark 0 was always (0, 0, 0) after local normalization.
                # It preserves trajectory relative to the shoulders when pose
                # is visible, avoiding dependence on camera framing.
                points[0] = self._normalize_wrist(raw_points[0], body_reference)

            detected_hands.append((label, points))

        # BLOQUE: canonizacion de senas de una mano.
        # Una sena equivalente hecha con la mano derecha se refleja sobre el eje
        # horizontal y se guarda en el espacio de la mano izquierda. Asi la red
        # aprende una sola representacion para el mismo gesto y no duplica clases
        # solo por la mano dominante del usuario.
        if len(detected_hands) == 1:
            # A one-handed sign has one semantic representation regardless of
            # the hand used. A right hand is mirrored into the left-hand frame
            # and always stored in the first slot.
            label, points = detected_hands[0]
            if label == "Right":
                points = self._mirror_hand(points)
            left = points
            return np.concatenate([left.flatten(), right.flatten()])

        # BLOQUE: asignacion de senas de dos manos.
        # En gestos bimanuales no se deben fusionar las manos: la relacion entre
        # izquierda y derecha puede contener significado. Se intenta conservar la
        # lateralidad de MediaPipe y se usa una asignacion de respaldo si esta no
        # llega completa en algun frame.
        left_assigned = False
        right_assigned = False
        unassigned = []
        for label, points in detected_hands:
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

    def prepare_sequence(self, sequence, target_length=SEQUENCE_LENGTH):
        return prepare_model_sequence(
            sequence,
            target_length=target_length,
            include_motion_features=self.include_motion_features,
        )

    @staticmethod
    def _normalize_hand(points):
        # BLOQUE: invariancia de tamano y distancia.
        # Se traslada la muneca al origen y se divide por la mayor distancia de
        # un landmark. Esto conserva la forma relativa de dedos y palma aunque
        # el usuario se acerque o aleje de la camara.
        wrist = points[0].copy()
        points = points - wrist
        scale = np.max(np.linalg.norm(points[:, :2], axis=1))
        if scale > 1e-6:
            points = points / scale
        return points

    @staticmethod
    def _body_reference(pose_results):
        # BLOQUE: referencia corporal.
        # Los hombros permiten expresar la muneca respecto al centro del cuerpo.
        # Es necesario para senas cuyo significado depende de apuntar hacia el
        # firmante o hacia una posicion del espacio, no solo de la forma manual.
        if not pose_results.pose_landmarks:
            return None

        landmarks = pose_results.pose_landmarks.landmark
        left = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        right = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        shoulder_width = float(np.hypot(left.x - right.x, left.y - right.y))
        if shoulder_width <= 1e-4:
            return None
        center = np.array(
            [(left.x + right.x) / 2, (left.y + right.y) / 2, (left.z + right.z) / 2],
            dtype=np.float32,
        )
        return center, shoulder_width

    @staticmethod
    def _normalize_wrist(wrist, body_reference=None):
        # A body-relative wrist position retains the meaningful location of
        # pronouns while being robust to camera translation and distance.
        if body_reference is not None:
            center, shoulder_width = body_reference
            return ((wrist - center) / shoulder_width).astype(np.float32)
        # Fallback for close hand-only frames where shoulders are not visible.
        return np.array([wrist[0] - 0.5, wrist[1] - 0.5, wrist[2]], dtype=np.float32)

    @staticmethod
    def _mirror_hand(points):
        """Reflects a normalized one-handed sign around its vertical axis."""
        mirrored = points.copy()
        mirrored[:, 0] *= -1
        return mirrored

# Alias heredado para scripts que aun importan el nombre anterior.
LandmarkExtractor = Preprocessor


# BLOQUE: transformaciones temporales.
# Los videos no duran lo mismo ni tienen la misma cantidad de frames. Estas
# funciones recortan, remuestrean, suavizan y opcionalmente calculan velocidad
# para producir una secuencia fija. La red recibe siempre 32 pasos temporales,
# incluso si el gesto original fue mas corto o mas largo.
def resample_sequence(sequence, target_length=SEQUENCE_LENGTH):
    # Interpola cada caracteristica a una linea temporal comun. No repite frames
    # sin criterio: distribuye la informacion de todo el gesto sobre 32 pasos.
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


def smooth_landmarks(sequence, alpha=SMOOTHING_ALPHA):
    """Applies EMA smoothing independently to each visible hand.

    Missing hands remain all zeros and reset their smoothing state, preventing
    a previous hand position from leaking into frames where it is absent.
    """
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[1] != FEATURE_SIZE:
        raise ValueError(f"Expected landmark sequence with shape (frames, {FEATURE_SIZE}).")

    hands = sequence.reshape(len(sequence), MAX_HANDS, LANDMARKS_PER_HAND, COORDINATES_PER_LANDMARK)
    smoothed = np.zeros_like(hands)
    previous = [None] * MAX_HANDS

    for frame_index, frame_hands in enumerate(hands):
        for hand_index, hand in enumerate(frame_hands):
            if not np.any(np.abs(hand) > 1e-6):
                previous[hand_index] = None
                continue
            if previous[hand_index] is None:
                current = hand
            else:
                current = alpha * hand + (1.0 - alpha) * previous[hand_index]
            smoothed[frame_index, hand_index] = current
            previous[hand_index] = current

    return smoothed.reshape(len(sequence), FEATURE_SIZE)


def append_motion_features(sequence):
    """Adds frame-to-frame velocity without creating motion for missing hands."""
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[1] != FEATURE_SIZE:
        raise ValueError(f"Expected landmark sequence with shape (frames, {FEATURE_SIZE}).")

    hands = sequence.reshape(len(sequence), MAX_HANDS, LANDMARKS_PER_HAND, COORDINATES_PER_LANDMARK)
    motion = np.zeros_like(hands)
    visible = np.any(np.abs(hands) > 1e-6, axis=(2, 3))

    for frame_index in range(1, len(sequence)):
        valid_hands = visible[frame_index] & visible[frame_index - 1]
        motion[frame_index, valid_hands] = hands[frame_index, valid_hands] - hands[frame_index - 1, valid_hands]

    return np.concatenate([sequence, motion.reshape(len(sequence), MOTION_FEATURE_SIZE)], axis=1)


def prepare_model_sequence(sequence, target_length=SEQUENCE_LENGTH, include_motion_features=False):
    # Punto de entrada del contrato del modelo: primero iguala la duracion,
    # despues reduce ruido y finalmente agrega velocidad solo si la configuracion
    # del modelo fue entrenada con esas 126 caracteristicas adicionales.
    """Returns the exact fixed-shape input consumed by the TensorFlow model."""
    fixed_length = resample_sequence(sequence, target_length)
    smoothed = smooth_landmarks(fixed_length)
    if include_motion_features:
        return append_motion_features(smoothed).astype(np.float32)
    return smoothed.astype(np.float32)


def prepare_model_sequences(sequences, target_length=SEQUENCE_LENGTH, include_motion_features=False):
    """Vector-like convenience wrapper used by training and validation."""
    return np.asarray(
        [prepare_model_sequence(sequence, target_length, include_motion_features) for sequence in sequences],
        dtype=np.float32,
    )


# BLOQUE: lectura y seleccion de ventanas de accion.
# Un clip puede tener reposo antes/despues del gesto o durar mas de 32 frames.
# Estas funciones eliminan bordes sin manos y crean vistas del movimiento util.
# La seleccion se hace por video para que entrenamiento y validacion conserven
# una separacion real entre grabaciones distintas.
def read_video_features(video_path, extractor=None):
    owns_extractor = extractor is None
    extractor = extractor or LandmarkExtractor()
    capture = open_video_capture(str(video_path))
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
    # Busca zonas de mayor movimiento y genera desplazamientos cercanos para
    # que el entrenamiento vea variaciones naturales de inicio y final del gesto.
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
    """Creates full-action and high-motion views of a variable-length sign.

    The full-action view preserves long gestures whose meaning depends on the
    order of the movement. High-motion views remain useful as augmentation for
    short signs and for variations in the start/end timing.
    """
    sequence = _trim_to_visible_hands(np.asarray(sequence, dtype=np.float32))
    if len(sequence) < MIN_VISIBLE_FRAMES:
        return []
    if len(sequence) <= target_length:
        return [resample_sequence(sequence, target_length)]

    windows = [resample_sequence(sequence, target_length)]
    for start in _window_starts(sequence, target_length, windows_per_video):
        window = sequence[start : start + target_length]
        if not any(np.array_equal(window, existing) for existing in windows):
            windows.append(window)
    return windows


def select_live_action_window(sequence, target_length=SEQUENCE_LENGTH):
    # En vivo se usa toda la accion visible y se remuestrea, en vez de escoger
    # solo el tramo mas activo. Esto conserva el orden temporal del gesto cuando
    # la camara llega a pocos FPS.
    """Preserves the visible gesture chronology for real-time recognition."""
    sequence = _trim_to_visible_hands(np.asarray(sequence, dtype=np.float32))
    return resample_sequence(sequence, target_length)


def select_dense_windows(sequence, target_length=SEQUENCE_LENGTH, stride=None):
    # Esta funcion se conserva para paquetes antiguos que empleaban ejemplos
    # negativos densos. El modelo final no incluye la clase __idle__.
    """Covers a negative clip densely so camera-like intermediate windows are learned."""
    sequence = _trim_to_visible_hands(np.asarray(sequence, dtype=np.float32))
    if len(sequence) < MIN_VISIBLE_FRAMES:
        return []
    if len(sequence) <= target_length:
        return [resample_sequence(sequence, target_length)]

    stride = max(1, int(stride or target_length // 4))
    maximum_start = len(sequence) - target_length
    starts = list(range(0, maximum_start + 1, stride))
    if starts[-1] != maximum_start:
        starts.append(maximum_start)
    return [sequence[start : start + target_length] for start in starts]


def extract_video_sequences(
    video_path,
    extractor=None,
    sequence_length=SEQUENCE_LENGTH,
    windows_per_video=3,
    dense_windows=False,
):
    features = read_video_features(video_path, extractor)
    if len(features) == 0 or not np.any(np.abs(features) > 1e-6):
        return []
    if dense_windows:
        return select_dense_windows(features, sequence_length)
    return select_action_windows(features, sequence_length, windows_per_video)


def extract_video_sequence(video_path, extractor=None, sequence_length=SEQUENCE_LENGTH):
    """Compatibility helper that returns the most active window of a video."""
    windows = extract_video_sequences(video_path, extractor, sequence_length, windows_per_video=1)
    if not windows:
        return np.zeros((sequence_length, FEATURE_SIZE), dtype=np.float32)
    return windows[0]


# BLOQUE: persistencia del paquete de modelo.
# Las funciones siguientes leen o escriben los metadatos que acompañan los pesos.
# Permiten que el servicio detecte de forma explicita un modelo incompatible en
# lugar de devolver traducciones con un preprocesamiento equivocado.
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
        "input_window": "full_visible_action_resampled_to_32",
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    with CONFIG_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
