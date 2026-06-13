from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from singerlens.features import extract_dataset_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    df = extract_dataset_features(args.metadata, args.output)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()

