"""Create the versioned model archive used by the inference service."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


FILES = ("lsc_sequence_model.keras", "lsc_labels.json", "lsc_config.json")
VERSION = "v1.0.0"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("model"))
    parser.add_argument("--output-dir", type=Path, default=Path("dist"))
    args = parser.parse_args()

    sources = [args.model_dir / name for name in FILES]
    if any(not path.is_file() for path in sources):
        parser.error("Model package requires the .keras, labels and config files together")

    labels = json.loads(sources[1].read_text(encoding="utf-8"))
    config = json.loads(sources[2].read_text(encoding="utf-8"))
    if not isinstance(labels, list) or not labels or len(set(labels)) != len(labels):
        parser.error("Labels must be a nonempty list without duplicates")
    if (config.get("sequence_length"), config.get("feature_size"), config.get("preprocessing_version")) != (32, 252, 7):
        parser.error("Unexpected model input dimensions")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / f"onelanguage-lsc-model-{VERSION}.zip"
    with ZipFile(archive, "w") as bundle:
        for source in sources:
            entry = ZipInfo(f"model/{source.name}", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            bundle.writestr(entry, source.read_bytes())

    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    print(f"{archive}  SHA-256 {digest}")


if __name__ == "__main__":
    main()
