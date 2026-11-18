# APPROACH

## Pipeline (one command: `python -m spendcube --config <company>.yaml --out <file>.csv`)

1. **Ingest by config.** Each source file is mapped to one canonical schema (amount, date,
   supplier name/id, description, material, PO, invoice, status, GL text, context). Company
   YAML is merged over `configs/defaults.yaml`. A new company needs only a new YAML.
2. **Clean.** Amounts are parsed in any format (`1,234.56`, `1.234,56`, `(12.30)`). VAT comes
   off gross amounts (`15`, `15%` and `0.15` are all 15%; blank means the configured default).
   Currency is converted with the given rate, sanity-checked against the monthly table (nearest
   month when one is missing). Dates are parsed through a format list, Excel serials and a
   lenient fallback; missing, unparseable and out-of-range dates are flagged, never dropped.
3. **Suppliers.** Union-find over vendor ids and cleaned core names: legal forms stripped,
   spaces and punctuation removed, OCR-folded, fuzzy ratio ≥ 92.
4. **Count once.** Secondary-system lines are linked to primary lines through two candidate
   blocks: same PO, and same net amount ±1% (a sorted-array search). Each candidate is scored
   on text and supplier similarity, PO item and amount difference. Lines exported twice inside
   one system are keyed on invoice + supplier + amount + quantity.
5. **Exclusions.** Status words → `CANCELLED` / `NOT_APPROVED`. Regex rules on description and
   GL text → `NON_PROCUREMENT` (intercompany, taxes, finance, payroll). Intercompany rules
   propagate to a vendor when most of its lines hit them. Vendor-level exclusions are learned
   from labels when labels exist. Matches → `DUPLICATE`.
6. **Items.** Union-find linking material numbers and OCR-folded normalised descriptions; OCR
   text is fuzzy-joined to clean text. Linked duplicates inherit the primary's item, supplier
   and category.
7. **Classify** to the deepest leaf of the tree file (no category names in code):
   - *With labels* (A, B): TF-IDF word + char n-grams → logistic regression, blended with the
     zero-shot prior (15% for clean labels, 35% for B's noisy seed). Clean labels are used
     as-is on the lines they cover.
   - *Zero labels* (C): (a) `intfloat/multilingual-e5-base` embeddings of the line vs. a
     descriptor per leaf (leaf + ancestors, acronyms expanded from ancestor initials, optional
     config **hints**); (b) TF-IDF match to the same leaf documents (word-level when hints
     exist, char-level otherwise); (c) self-training: a text model fitted on the 30% most
     confident lines, *description only* (training on GL codes made one code a rule for
     thousands of lines); (d) supplier smoothing toward each supplier's category mix.
   - Confidence = the final blended probability of the chosen leaf.

## Measured accuracy

| Setting | L1 (line / spend) | L3 (line / spend) |
|---|---|---|
| A, trained on 80% of labels, scored on the hidden 20% | 99.8% / 99.8% | 98.4% / 98.9% |
| **A, all labels hidden, tree only (no hints)** | 53% / 70% | 26% / 36% |
| B, trained on 70% of seed labels, scored on the hidden 30% | 95.6% / 98.4% | 93.0% / 98.2% |
| B, all labels hidden, tree + config hints | 91% / 98.7% | 71% / 84% |

A without labels is hard: its tree splits mining equipment by service type ("CM Full Overhaul"
vs. "CM Repair Items") and its texts are SAP abbreviations. Hints in the config are the main
lever for an unseen tree. On B they moved spend-weighted L1 from 40% to 99%, so I will write
hints for C at the blind test (config only, no code change).

Other checks: A suppliers match the client grouping exactly outside the client's "no vendor"
catch-all bucket (69 of 69 groups). A items: homogeneity 0.98, completeness 0.95. B
include/exclude: 98.6% agreement with held-out seed labels. A count-once: 25,316 of 28,520 tool
lines matched to ERP, 2,712 tool-only lines included as spend.

## Runtime

On a 16-core laptop CPU: A (65,549 records) about 6 min; B (41,028) about 3 min. The first run
also downloads the embedding model (about 1.1 GB). 25,000 lines is well inside the 60-minute
limit.

## AI and LLM use

- **Inside the tool:** local models only by default. `intfloat/multilingual-e5-base` from
  Hugging Face via `sentence-transformers` is free, runs on CPU and needs no key.
- **Optional:** `spendcube/llm.py` labels the highest-spend unique texts with Claude
  (`claude-opus-5-5`, effort `low`, structured output, batches of 60, cached on disk) when
  `ANTHROPIC_API_KEY` is set and `anthropic` is installed. The local text model then spreads
  those labels to the rest. **The submitted outputs did not use it.** Estimated cost: about
  1,500 input tokens for the tree prompt (cached) + 40 tokens per line in, about 15 tokens per
  line out. At $4 / $20 per million tokens that is about $0.50 per 1,000 unique lines.
- **Building it:** I used Claude Code (Anthropic) as a coding assistant to write and debug the
  code and these documents. The analysis scripts ran locally, and the assistant read samples
  of their output while exploring the data. The tool itself sends no data anywhere unless the
  optional LLM step is switched on.
