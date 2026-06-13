from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from singerlens.train import run_ablation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    results = run_ablation(args.features, args.output_dir)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

