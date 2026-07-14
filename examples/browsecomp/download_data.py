"""Download the BrowseComp-Plus train/test data used by this example.

The bc_train / bc_test parquet files come from the Context-Folding / FoldAgent
release (https://arxiv.org/abs/2510.11967). This script fetches the archive
from its public Google Drive mirror and extracts the parquet files, then you
run prepare_data.py to convert them to slime jsonl.

Usage:
    pip install gdown
    python download_data.py --out ./data_raw
    python prepare_data.py --input ./data_raw/bc_train.parquet --output data/bc_train.jsonl
    python prepare_data.py --input ./data_raw/bc_test.parquet  --output data/bc_test.jsonl

If the Google Drive link is unavailable, obtain bc_train.parquet / bc_test.parquet
from the FoldAgent repository's data/ directory directly.
"""

import argparse
import os
import subprocess
import sys

# FoldAgent BrowseComp data archive (Google Drive file id from the FoldAgent README).
GDRIVE_FILE_ID = "1aX5xXAN5R-gLKd8A0AY-troxXJRawyAM"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./data_raw", help="directory to download+extract into")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    try:
        import gdown  # noqa: F401
    except ImportError:
        sys.exit("gdown is required: pip install gdown")

    archive = os.path.join(args.out, "browsecomp_data.zip")
    print(f"Downloading BrowseComp data (gdrive id={GDRIVE_FILE_ID}) -> {archive}")
    # gdown >= 6 dropped the --id flag; a bare file id works on all versions.
    subprocess.run(
        [sys.executable, "-m", "gdown", GDRIVE_FILE_ID, "-O", archive],
        check=True,
    )
    print("Extracting ...")
    # The Drive archive is a zip (despite earlier docs calling it a tarball).
    subprocess.run(["unzip", "-o", archive, "-d", args.out], check=True)
    print(f"Done. Look for bc_train.parquet / bc_test.parquet under {args.out}")


if __name__ == "__main__":
    main()
