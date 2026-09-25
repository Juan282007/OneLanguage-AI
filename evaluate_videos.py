"""Audits a trained LSC model against videos grouped by label.

This is a diagnostic tool. Use separate recordings for an unbiased evaluation.
"""

import argparse
import json
from collections import defaultdict

import numpy as np
import tensorflow as tf

from lsc_pipeline import (
    LABELS_PATH,
    MODEL_PATH,
    LandmarkExtractor,
    extract_video_sequences,
    is_no_sign_label,
    load_model_config,
    prepare_model_sequences,
)
from train_from_videos import collect_videos


def main():
    parser = argparse.ArgumentParser(description="Evalua el modelo LSC por video y muestra confusiones.")
    parser.add_argument("--data", default="videos", help="Carpeta videos/<etiqueta>/*")
    parser.add_argument("--labels", help="Etiquetas separadas por coma para limitar la auditoria.")
    args = parser.parse_args()

    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    config = load_model_config()
    sequence_length = int(config["sequence_length"])
    include_motion = bool(config.get("include_motion_features", False))
    selected_labels = {label.strip() for label in args.labels.split(",")} if args.labels else None
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    samples = [sample for sample in collect_videos(args.data) if not selected_labels or sample[1] in selected_labels]
    if not samples:
        raise SystemExit("No encontre videos para las etiquetas solicitadas.")

    extractor = LandmarkExtractor(include_motion_features=include_motion)
    totals = defaultdict(int)
    correct = defaultdict(int)
    confusions = defaultdict(lambda: defaultdict(int))
    try:
        for video_path, expected_label in samples:
            windows = extract_video_sequences(
                video_path,
                extractor,
                sequence_length,
                windows_per_video=3,
                dense_windows=is_no_sign_label(expected_label),
            )
            if not windows:
                print(f"SIN MANOS  {video_path}")
                continue

            prepared = prepare_model_sequences(windows, sequence_length, include_motion_features=include_motion)
            probabilities = model.predict(prepared, verbose=0).mean(axis=0)
            predicted_label = labels[int(np.argmax(probabilities))]
            confidence = float(np.max(probabilities))
            totals[expected_label] += 1
            correct[expected_label] += int(predicted_label == expected_label)
            confusions[expected_label][predicted_label] += 1
            marker = "OK" if predicted_label == expected_label else "ERROR"
            print(f"{marker:5} {video_path.name}: {expected_label} -> {predicted_label} ({confidence:.0%})")
    finally:
        extractor.close()

    print("\nResumen por sena:")
    for label in sorted(totals):
        errors = [f"{predicted}={count}" for predicted, count in confusions[label].items() if predicted != label]
        error_text = ", ".join(errors) if errors else "ninguna"
        print(f"  {label}: {correct[label]}/{totals[label]} correctos; confusiones: {error_text}")


if __name__ == "__main__":
    main()
