"""Reports per-label model performance against the locally labelled videos."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf

from lsc_pipeline import (
    LABELS_PATH,
    MODEL_PATH,
    Preprocessor,
    extract_video_sequences,
    load_model_config,
    prepare_model_sequences,
)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def main():
    parser = argparse.ArgumentParser(description="Evalua el modelo LSC con videos etiquetados.")
    parser.add_argument("--data", default="videos", help="Carpeta videos/<etiqueta>/*.mp4")
    parser.add_argument("--target", help="Muestra de que clases provienen las predicciones de esta etiqueta.")
    args = parser.parse_args()

    data_dir = Path(args.data)
    samples = sorted(path for path in data_dir.glob("*/*") if path.suffix.lower() in VIDEO_EXTENSIONS)
    if not samples:
        raise SystemExit(f"No encontre videos en {data_dir}.")

    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    config = load_model_config()
    sequence_length = int(config["sequence_length"])
    include_motion_features = bool(config.get("include_motion_features", False))
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    extractor = Preprocessor(
        include_wrist_trajectory=bool(config.get("uses_wrist_trajectory", False)),
        include_motion_features=include_motion_features,
    )

    results = []
    try:
        for index, path in enumerate(samples, start=1):
            windows = extract_video_sequences(path, extractor, sequence_length, windows_per_video=1)
            if not windows:
                results.append((path.parent.name, "<sin_manos>", 0.0))
                continue
            sequences = prepare_model_sequences(
                windows,
                sequence_length,
                include_motion_features=include_motion_features,
            )
            probabilities = model.predict(sequences, verbose=0).mean(axis=0)
            predicted = labels[int(np.argmax(probabilities))]
            results.append((path.parent.name, predicted, float(np.max(probabilities))))
            print(f"[{index}/{len(samples)}] {path.parent.name} -> {predicted}")
    finally:
        extractor.close()

    grouped = defaultdict(list)
    for actual, predicted, confidence in results:
        grouped[actual].append((predicted, confidence))

    print("\nResultado por clase")
    for actual in sorted(grouped):
        rows = grouped[actual]
        correct = sum(predicted == actual for predicted, _ in rows)
        average_confidence = float(np.mean([confidence for _, confidence in rows]))
        predictions = ", ".join(
            f"{label}={count}" for label, count in Counter(predicted for predicted, _ in rows).most_common()
        )
        print(f"  {actual}: {correct}/{len(rows)} ({correct / len(rows):.0%}), "
              f"confianza media={average_confidence:.0%}; {predictions}")

    if args.target:
        print(f"\nVideos clasificados como '{args.target}'")
        for actual in sorted(grouped):
            count = sum(predicted == args.target for predicted, _ in grouped[actual])
            if count:
                print(f"  {actual}: {count}")


if __name__ == "__main__":
    main()
