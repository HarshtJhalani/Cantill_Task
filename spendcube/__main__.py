"""python -m spendcube --config configs/company_A.yaml --out output/yourname__A.csv"""
import argparse
import logging
import warnings

from .pipeline import run


def main():
    ap = argparse.ArgumentParser(description="Spend cube: ingest, clean, count once, de-duplicate, classify.")
    ap.add_argument("--config", required=True, help="company config YAML")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--data-dir", default=None, help="folder holding the company's files (overrides data_dir in the config)")
    ap.add_argument("--no-labels", action="store_true", help="ignore any labels file (zero-label mode)")
    ap.add_argument("--holdout", type=float, default=0.0, help="evaluation only: hide this share of labels")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("sentence_transformers", "httpx", "urllib3", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    run(a.config, a.out, a.data_dir, use_labels=not a.no_labels, holdout=a.holdout)


if __name__ == "__main__":
    main()
