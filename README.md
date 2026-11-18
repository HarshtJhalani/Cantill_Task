# spendcube

Turns a company's raw purchase data and its category tree into a spend cube. It combines the
company's systems and counts each purchase once, cleans amounts, tax, currency and dates, decides
what counts as spend (with a reason code for every exclusion), de-duplicates suppliers and items,
and classifies every line into the company's own tree with a confidence.

A new company needs a new YAML config, not new code. The tree is read from a file at run time, so
no category names live in the code.

## Setup

Python 3.10+ (developed on 3.14).

```bash
pip install -r requirements.txt
```

The first run downloads the local embedding model (`intfloat/multilingual-e5-base`, about 1.1 GB)
from Hugging Face and caches it. After that the tool runs offline.

## The one command

```bash
python -m spendcube --config configs/company_A.yaml --out output/yourname__A.csv
```

| Flag | Meaning |
|---|---|
| `--config` | company config (YAML) |
| `--out` | output CSV, one row per input record |
| `--data-dir` | folder holding that company's files, overriding `data_dir` in the config (use this for A2/B2/C) |
| `--no-labels` | ignore the labels file, so it classifies from the tree alone |
| `--holdout 0.2` | evaluation only: hide 20% of labels and write their ids next to the output |

Blind-test runs, with the new files in their own folders:

```bash
python -m spendcube --config configs/company_A.yaml --data-dir path/to/A2 --out submission/yourname__A2.csv
```

```bash
python -m spendcube --config configs/company_B.yaml --data-dir path/to/B2 --out submission/yourname__B2.csv
```

```bash
python -m spendcube --config submission/company_C.yaml --out submission/yourname__C.csv
```

For A2 and B2, if the new folder does not contain the training labels, copy `labels_A.csv` /
`seed_labels.csv` into it, or point `labels.file` in the config at an absolute path. The labels
come from the build data and train the classifier.

Every run also writes `<out>_diagnostics.csv`, which has the exclusion rule that fired, the record
a duplicate points to, the FX source, date flags and the classification method for each line.

Score an output against a labels file:

```bash
python -m spendcube.evaluate --config configs/company_A.yaml --pred output/x.csv --only output/x.csv.holdout_ids
```

## Optional LLM step

If `ANTHROPIC_API_KEY` is set and `pip install anthropic` has been run, the tool asks Claude to
label the highest-spend unique line texts (up to `classification.llm.max_unique_texts`). Those
labels then train the local text model, so the remaining lines inherit them. Without a key it uses
the local path only (embeddings plus self-training). Cost and settings are in APPROACH.md, and
results are cached in `cache/`.

## Writing a config for a new company

Copy `configs/template_company.yaml`. Name the files, map each canonical field (amount, date,
supplier, description, status, ...) to a column, say whether amounts are gross, give the reporting
currency and optional FX table, and point at the tree. Optionally add `taxonomy.hints` (keywords
per leaf) and `extra_exclusion_rules`. `configs/defaults.yaml` holds the shared rules (status
words, non-procurement patterns, thresholds), and company configs override it.

## Layout

```
spendcube/
  config.py      config loading (company YAML over defaults.yaml)
  ingest.py      read sources, parse amounts/dates/VAT, convert currency
  matching.py    cross-system matching and repeated-export detection
  exclusions.py  status, non-procurement rules, vendor propagation
  entities.py    supplier and item de-duplication (union-find)
  taxonomy.py    tree file -> leaves and descriptors
  classify.py    supervised / zero-shot / self-training / supplier smoothing
  llm.py         optional Claude labelling
  pipeline.py    end to end
  evaluate.py    scoring against labels
configs/         defaults + one YAML per company
output/          A and B outputs at the build deadline
submission/      blind-test outputs and the C config
```

The task data is confidential and is never committed (see `.gitignore`).
