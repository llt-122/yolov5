from __future__ import annotations

import argparse
import csv
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "datasets" / "coco1000" / "manifest.csv"
DEFAULT_OUTPUT = ROOT / "datasets" / "coco1000" / "images" / "val"


def download(row, output_dir, retries):
    destination = output_dir / row["file_name"]
    if destination.exists() and destination.stat().st_size:
        return destination

    temporary = destination.with_suffix(".part")
    for attempt in range(retries):
        try:
            request = urllib.request.Request(row["source_url"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            temporary.replace(destination)
            return destination
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)


def main():
    parser = argparse.ArgumentParser(description="Download the fixed COCO1000 image subset from its manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=6)
    args = parser.parse_args()

    with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No images found in {args.manifest}")

    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        paths = list(pool.map(lambda row: download(row, args.output, args.retries), rows))

    errors = []
    for path in paths:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
    if errors:
        raise RuntimeError("Image verification failed:\n" + "\n".join(errors[:20]))
    print(f"Downloaded and verified {len(paths)} images in {args.output}")


if __name__ == "__main__":
    main()
