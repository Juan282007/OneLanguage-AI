import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from lsc_pipeline import MODEL_PATH, SEQUENCE_LENGTH, LandmarkExtractor, extract_video_sequences, save_metadata


VIDEO_EXTENSIONS = {".avi", ".mov", ".mp4", ".mkv", ".webm"}


def collect_videos(dataset_dir):
    dataset_dir = Path(dataset_dir)
    samples = []
    for label_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        for video_path in sorted(label_dir.rglob("*")):
            if video_path.suffix.lower() in VIDEO_EXTENSIONS:
                samples.append((video_path, label_dir.name))
    return samples


def build_model(sequence_length, feature_size, num_classes):
    regularizer = tf.keras.regularizers.l2(1e-4)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(sequence_length, feature_size)),
            tf.keras.layers.Masking(mask_value=0.0),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(48, return_sequences=True, dropout=0.15, kernel_regularizer=regularizer)
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(24, dropout=0.15, kernel_regularizer=regularizer)
            ),
            tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizer),
            tf.keras.layers.Dropout(0.45),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0007),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def augment_sequence(sequence, rng):
    augmented = sequence.copy()

    speed = rng.uniform(0.85, 1.15)
    source_steps = np.linspace(0, 1, num=len(augmented))
    target_steps = np.linspace(0, 1, num=len(augmented)) ** speed
    warped = np.empty_like(augmented)
    for feature_index in range(augmented.shape[1]):
        warped[:, feature_index] = np.interp(target_steps, source_steps, augmented[:, feature_index])
    augmented = warped

    scale = rng.uniform(0.92, 1.08)
    augmented *= scale

    noise = rng.normal(0, 0.01, size=augmented.shape).astype(np.float32)
    # Keep missing-hand coordinates at zero. Adding noise there creates fake hands.
    augmented += noise * (np.abs(augmented) > 1e-6)

    return augmented.astype(np.float32)


def mirror_sequence(sequence):
    """Mirrors landmarks and swaps the left/right hand slots."""
    sequence = np.asarray(sequence, dtype=np.float32)
    hands = sequence.reshape(sequence.shape[0], 2, 21, 3).copy()
    hands = hands[:, [1, 0], :, :]
    hands[..., 0] *= -1
    return hands.reshape(sequence.shape)


def expand_with_augmentation(X, y, augmentations, include_mirror=True):
    """Adds realistic timing/noise variants and optional opposite-hand variants."""
    rng = np.random.default_rng(42)
    expanded_X = []
    expanded_y = []

    for sample, label in zip(X, y):
        variants = [sample]
        if include_mirror:
            variants.append(mirror_sequence(sample))

        for variant in variants:
            expanded_X.append(variant)
            expanded_y.append(label)
            for _ in range(augmentations):
                expanded_X.append(augment_sequence(variant, rng))
                expanded_y.append(label)

    return np.asarray(expanded_X, dtype=np.float32), np.asarray(expanded_y, dtype=np.int64)


def flatten_video_windows(video_indexes, windows_by_video, video_labels, primary_only=False):
    X, y = [], []
    for video_index in video_indexes:
        windows = windows_by_video[video_index][:1] if primary_only else windows_by_video[video_index]
        X.extend(windows)
        y.extend([video_labels[video_index]] * len(windows))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def print_validation_report(model, X_val, y_val, labels, title):
    probabilities = model.predict(X_val, verbose=0)
    predicted = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    print(title)
    for index, label in enumerate(labels):
        mask = y_val == index
        total = int(np.sum(mask))
        correct = int(np.sum(predicted[mask] == index))
        average_confidence = float(np.mean(confidence[mask])) if total else 0.0
        wrong = predicted[mask & (predicted != index)]
        if len(wrong):
            values, counts = np.unique(wrong, return_counts=True)
            confusions = ", ".join(f"{labels[value]}={count}" for value, count in zip(values, counts))
        else:
            confusions = "ninguna"
        print(f"  {label}: {correct}/{total} ({correct / total:.0%}), confianza={average_confidence:.0%}, errores: {confusions}")


def main():
    parser = argparse.ArgumentParser(
        description="Entrena un modelo de Lengua de Señas Colombiana usando videos organizados por carpeta."
    )
    parser.add_argument("--data", default="videos", help="Carpeta con estructura videos/<senal>/*.mp4")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument(
        "--augmentations",
        type=int,
        default=3,
        help="Variaciones artificiales por video para mejorar datasets pequeños.",
    )
    parser.add_argument(
        "--windows-per-video",
        type=int,
        default=3,
        help="Ventanas activas de 32 frames extraidas por cada video para igualar la camara.",
    )
    parser.add_argument(
        "--no-mirror-augmentation",
        action="store_true",
        help="No crea variantes reflejadas para la mano contraria.",
    )
    args = parser.parse_args()
    args.windows_per_video = max(1, args.windows_per_video)

    samples = collect_videos(args.data)
    if not samples:
        raise SystemExit("No encontré videos. Usa una estructura como videos/hola/video1.mp4")

    labels = sorted({label for _, label in samples})
    if len(samples) < 2:
        raise SystemExit(
            "Encontré solo 1 video. Necesitas al menos 2 videos para entrenar, "
            "y para un modelo útil recomiendo 5 a 10 videos por cada seña."
        )
    if len(labels) < 2:
        print(
            "Aviso: solo hay una seña en el dataset. El modelo puede entrenar, "
            "pero no aprenderá a diferenciarla de otras señas hasta que agregues más carpetas."
        )

    label_to_index = {label: index for index, label in enumerate(labels)}

    extractor = LandmarkExtractor()
    windows_by_video, video_labels = [], []
    try:
        for index, (video_path, label) in enumerate(samples, start=1):
            print(f"[{index}/{len(samples)}] Extrayendo: {video_path}")
            windows = extract_video_sequences(
                video_path,
                extractor,
                args.sequence_length,
                args.windows_per_video,
            )
            if not windows:
                print(f"  Aviso: no detecte manos en {video_path}; se omite del entrenamiento.")
                continue
            windows_by_video.append(windows)
            video_labels.append(label_to_index[label])
    finally:
        extractor.close()

    video_labels = np.asarray(video_labels, dtype=np.int64)
    if len(video_labels) < 2:
        raise SystemExit("No hubo suficientes videos con manos detectadas para entrenar.")
    raw_counts = np.bincount(video_labels, minlength=len(labels))
    missing_labels = [label for index, label in enumerate(labels) if raw_counts[index] == 0]
    if missing_labels:
        raise SystemExit("No detecte manos en ningun video de: " + ", ".join(missing_labels))
    print("Videos por sena: " + ", ".join(f"{label}={raw_counts[index]}" for index, label in enumerate(labels)))
    if raw_counts.max() > raw_counts.min() * 1.5:
        print("Aviso: hay un desbalance entre senas. Graba mas videos de las clases con menos ejemplos.")
    if not any(label.casefold().strip() in {"nada", "reposo", "sin_sena", "sin senas"} for label in labels):
        print("Consejo: agrega una carpeta 'nada' o 'reposo' para que la camara no fuerce una traduccion cuando no hay sena.")

    # Split videos before creating windows or augmentations. This prevents windows from
    # the same recording leaking into validation.
    can_validate = len(labels) > 1 and len(video_labels) >= len(labels) * 3 and np.all(raw_counts >= 3)
    video_indexes = np.arange(len(video_labels))
    if can_validate:
        train_indexes, validation_indexes = train_test_split(
            video_indexes,
            stratify=video_labels,
            test_size=0.25,
            random_state=42,
        )
        X_train_raw, y_train_raw = flatten_video_windows(train_indexes, windows_by_video, video_labels)
        X_val, y_val = flatten_video_windows(
            validation_indexes,
            windows_by_video,
            video_labels,
            primary_only=True,
        )
        X_train, y_train = expand_with_augmentation(
            X_train_raw,
            y_train_raw,
            args.augmentations,
            include_mirror=not args.no_mirror_augmentation,
        )
        validation_data = (X_val, y_val)
        monitor = "val_loss"
        monitor_mode = "min"
        print(
            f"Entrenamiento: {len(train_indexes)} videos, {len(X_train_raw)} ventanas activas, "
            f"{len(X_train)} secuencias con aumentos. Validacion: {len(X_val)} videos sin aumentos."
        )
    else:
        print("Aviso: hay pocos videos; entrenaré sin conjunto de validación.")
        X_train_raw, y_train_raw = flatten_video_windows(video_indexes, windows_by_video, video_labels)
        X_train, y_train = expand_with_augmentation(
            X_train_raw,
            y_train_raw,
            args.augmentations,
            include_mirror=not args.no_mirror_augmentation,
        )
        validation_data = None
        monitor = "loss"
        monitor_mode = "min"
        print(f"Dataset final: {len(X_train)} secuencias ({args.augmentations} aumentos por video).")

    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    class_weight = {int(label): float(weight) for label, weight in zip(np.unique(y_train), class_weights)}

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model = build_model(args.sequence_length, X_train.shape[2], len(labels))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor=monitor, mode=monitor_mode),
        tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, mode=monitor_mode, factor=0.5, patience=5, min_lr=1e-5),
        tf.keras.callbacks.ModelCheckpoint(str(MODEL_PATH), save_best_only=True, monitor=monitor, mode=monitor_mode),
    ]

    fit_kwargs = {
        "x": X_train,
        "y": y_train,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        # Text-only progress avoids Unicode progress-bar failures in Windows consoles using cp1252.
        "verbose": 2,
        "class_weight": class_weight,
        "callbacks": callbacks,
    }
    if validation_data:
        fit_kwargs["validation_data"] = validation_data

    model.fit(**fit_kwargs)

    if validation_data:
        print_validation_report(model, X_val, y_val, labels, "Validacion por sena:")
        if not args.no_mirror_augmentation:
            X_val_mirrored = np.asarray([mirror_sequence(sample) for sample in X_val], dtype=np.float32)
            mirrored_metrics = model.evaluate(X_val_mirrored, y_val, verbose=0, return_dict=True)
            print(f"Validacion espejo: accuracy={mirrored_metrics['accuracy']:.1%}, loss={mirrored_metrics['loss']:.4f}")

    model.save(MODEL_PATH)
    save_metadata(
        labels,
        args.sequence_length,
        extra_metadata={"windows_per_video": args.windows_per_video},
    )
    print(f"Modelo guardado en {MODEL_PATH}")
    print(f"Etiquetas guardadas: {', '.join(labels)}")


if __name__ == "__main__":
    main()
