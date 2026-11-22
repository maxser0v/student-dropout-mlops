"""Import an externally downloaded PAN archive into a local corpus manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plagiarism.data import assign_splits, load_pan, save_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="PAN 2013 archive or extracted corpus directory")
    parser.add_argument("--output", type=Path, default=ROOT / "plagiarism_data" / "pan13.json")
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--training-archive-only", action="store_true",
                        help="Use only the official PAN training ZIP inside the downloaded release")
    parser.add_argument("--test-fraction", type=float, default=0.0,
                        help="Fraction of document groups to reserve as a custom test split")
    parser.add_argument("--seed", type=int, default=903)
    args = parser.parse_args()
    if args.training_archive_only != (args.test_fraction > 0):
        parser.error("--training-archive-only requires a positive --test-fraction and vice versa")
    source = load_pan(args.input, encoding=args.encoding, training_archive_only=args.training_archive_only)
    corpus = assign_splits(source, validation_fraction=args.validation_fraction,
                           test_fraction=args.test_fraction, seed=args.seed)
    save_corpus(corpus, args.output)
    print(f"saved {len(corpus.documents)} documents, {len(corpus.annotations)} aligned spans, {len(corpus.pairs)} judged pairs to {args.output}")


if __name__ == "__main__":
    main()
