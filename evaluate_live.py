"""Replay held-out videos at camera-like frame rates through the live recognizer."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2

from lsc_pipeline import Preprocessor, is_no_sign_label, open_video_capture
from web_service import RecognitionSession, runtime


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def read_camera_features(path, extractor, target_fps, jpeg_width, jpeg_quality):
    capture = open_video_capture(str(path))
    frames = []
    try:
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        step = max(1, round(source_fps / target_fps)) if source_fps > 0 else 1
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % step == 0:
                if jpeg_width:
                    height, width = frame.shape[:2]
                    target_width = min(width, jpeg_width)
                    if target_width != width:
                        frame = cv2.resize(frame, (target_width, round(height * target_width / width)))
                    encoded_ok, encoded = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                    )
                    if not encoded_ok:
                        raise RuntimeError(f"No pude codificar {path}")
                    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                frames.append(extractor.frame_features(frame))
            index += 1
    finally:
        capture.release()
    return frames


def main():
    parser = argparse.ArgumentParser(description="Evalua reconocimiento temporal simulando una camara lenta.")
    parser.add_argument("--data", default="evaluation_videos")
    parser.add_argument("--fps", type=float, default=5.0, help="Fotogramas enviados por segundo (aproximado).")
    parser.add_argument("--jpeg-width", type=int, default=1280, help="Ancho maximo del JPEG web; 0 desactiva JPEG.")
    parser.add_argument("--jpeg-quality", type=int, default=68, help="Calidad JPEG de 0 a 100.")
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps debe ser mayor que cero")
    if args.jpeg_width < 0 or not 0 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-width debe ser >= 0 y --jpeg-quality debe estar entre 0 y 100")

    samples = sorted(path for path in Path(args.data).glob("*/*") if path.suffix.lower() in VIDEO_EXTENSIONS)
    if not samples:
        parser.error(f"No encontre videos etiquetados en {args.data}")

    extractor = Preprocessor(
        include_wrist_trajectory=bool(runtime.config.get("uses_wrist_trajectory", False)),
        include_motion_features=runtime.include_motion_features,
    )
    results = defaultdict(list)
    try:
        for index, path in enumerate(samples, 1):
            features = read_camera_features(path, extractor, args.fps, args.jpeg_width, args.jpeg_quality)
            session = RecognitionSession(runtime)
            translations = []
            try:
                for frame_features in features:
                    prediction = session.process_features(frame_features)
                    if prediction["is_new_translation"]:
                        translations.append(prediction["label"])
            finally:
                session.close()

            expected = [] if is_no_sign_label(path.parent.name) else [path.parent.name]
            correct = translations == expected
            results[path.parent.name].append(correct)
            if not correct:
                print(f"[{index}/{len(samples)}] {path.parent.name}/{path.name} -> {', '.join(translations) or '<sin traduccion>'}", flush=True)
    finally:
        extractor.close()

    print(f"\nReproduccion aproximada a {args.fps:g} fps, JPEG {args.jpeg_width}px calidad {args.jpeg_quality} (incluye espera y confirmacion):")
    for label, decisions in sorted(results.items()):
        print(f"  {label}: {sum(decisions)}/{len(decisions)}")
    all_decisions = [decision for decisions in results.values() for decision in decisions]
    print(f"Total: {sum(all_decisions)}/{len(all_decisions)}")


if __name__ == "__main__":
    main()
