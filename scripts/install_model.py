"""Download and verify the pinned One Language inference model."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO


VERSION = "v1.0.0"
URL = (
    "https://github.com/Juan282007/OneLanguage-AI/releases/download/"
    f"lsc-{VERSION}/onelanguage-lsc-model-{VERSION}.zip"
)
SHA256 = "bf4830cac585779c7ae03fb25ad545dcf325ad480259ebc1f87e1b9c60a06074"
FILES = ("lsc_sequence_model.keras", "lsc_labels.json", "lsc_config.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use a downloaded archive instead of GitHub")
    parser.add_argument("--model-dir", type=Path, default=Path("model"))
    args = parser.parse_args()

    if len(SHA256) != 64:
        parser.error("The model release checksum has not been set")
    if args.archive:
        data = args.archive.read_bytes()
    else:
        with urlopen(URL, timeout=30) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        parser.error("Model archive checksum mismatch")

    with ZipFile(BytesIO(data)) as bundle:
        expected = {f"model/{name}" for name in FILES}
        if set(bundle.namelist()) != expected:
            parser.error("Model archive must contain exactly the three compatible artifacts")
        labels = json.loads(bundle.read("model/lsc_labels.json"))
        config = json.loads(bundle.read("model/lsc_config.json"))
        if not isinstance(labels, list) or not labels or len(set(labels)) != len(labels):
            parser.error("Invalid model labels")
        if (config.get("sequence_length"), config.get("feature_size"), config.get("preprocessing_version")) != (32, 252, 7):
            parser.error("Invalid model input dimensions")
        args.model_dir.mkdir(parents=True, exist_ok=True)
        for name in FILES:
            (args.model_dir / name).write_bytes(bundle.read(f"model/{name}"))
    print(f"Installed {VERSION} in {args.model_dir}")


if __name__ == "__main__":
    main()
