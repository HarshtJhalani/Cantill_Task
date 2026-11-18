# DECISIONS

What I found in the data and what I decided. Shared rules live in `configs/defaults.yaml` and
company-specific choices in `configs/company_*.yaml`, so every decision below can be changed
without touching code.

## Company A (two systems, ZAR)

### What the data looks like
- **ERP** (37,029 goods-receipt lines): `value_zar` is net and already in ZAR, even on the 16
  lines whose `document_currency` is EUR, CHF or MZN. Two-thirds of lines have no posting date,
  vendor id, UoM or site code. There is no status column and no duplicate rows. Some lines are
  legitimately repeated: the same PO line received twice, with different receipt documents.
- **Invoice tool** (28,520 lines): amounts are gross and use mixed formats (`5,266.63`,
  `4240.22`). The VAT rate is written as `15`, `0.15`, `15%`, `0` or blank. Dates come in four
  formats (`2022-01-29`, `19/01/2024`, `18-Sep-21`, blank in 65% of lines). PO references appear
  as `4510…`, `PO#4510…`, `PO 4510…`, `4510…/10.0`, and 18% are missing. Line text is OCR'd
  (`P0WDR;STNEDST`, `8ALLAST`, `VEH1CLE`). Supplier names come in 702 spellings of about 190 real
  suppliers (`OAK ENT MINING`, `YA RAX`, `Kelmont (Pty) Ltd`).

### Decisions
| Topic | Decision |
|---|---|
| VAT | Net = gross / (1 + rate). `15`, `15%` and `0.15` all mean 15%. Blank means 15%, as the brief says (I checked: 5,020 blank-rate lines match an ERP value at /1.15 and only 63 match at /1). `0` is zero-rated, so net = gross (1,242 such lines match ERP exactly). |
| PO reference | Take the 10-digit PO number with a regex. Take the PO item from the `/10.0` suffix when present. |
| Count once | A tool line is the same purchase as an ERP line when **(a)** the PO matches, net amounts agree within 1%, the PO item does not contradict and text or supplier agrees, **or (b)** the PO matches, the text is near-identical (≥85) and the amount is within 25% (OCR'd quantities), **or (c)** there is no usable PO (missing, or not in the ERP), but the amount agrees within 1% and both text (≥70) and supplier (≥70) agree. The best-scoring candidate wins. Text is compared after OCR folding (0→O, 5→S, 1→I, 8→B). Result: 25,316 of 28,520 tool lines are `DUPLICATE` of an ERP line. Spot checks of the low-similarity matches showed only case or OCR differences. |
| Matching many-to-one | Several tool lines may point to one ERP line, because an invoice exported twice is the same purchase. A tool line never points to more than one ERP line. |
| Exported twice | Inside the tool export, the same invoice number (digits only), supplier, gross amount and quantity means a repeat export. The first copy counts and later copies are `DUPLICATE`. 2,359 such lines; most also match ERP. |
| Status | `Paid` and `Approved` count. `Cancelled` → `CANCELLED`. `On Hold` and `Rejected` → `NOT_APPROVED`. An unknown status word → `NOT_APPROVED` (only approved or paid counts). |
| Precedence | `CANCELLED` > `NOT_APPROVED` > `NON_PROCUREMENT` > `DUPLICATE`. A cancelled tool line that also matches an ERP receipt is reported as `CANCELLED`: the status is the more specific fact, and the ERP line counts the spend either way. |
| Tool-only spend | 2,712 tool lines have no ERP counterpart and are approved or paid. They are included as spend, which is the "spend that exists in only one system". |
| Supplier key | Union-find over (vendor id) and (cleaned core name). The cleaned core name is ASCII-folded, upper-cased, punctuation removed, legal forms (`PTY`, `LTD`, `CC`, `SA`, `GMBH`…) dropped, spaces removed and OCR-folded, then fuzzy-joined at ratio ≥ 92 within the same first two letters. On the ERP lines that have a real client key, this matches the client's grouping exactly (69 of 69 groups, homogeneity and completeness 1.0). |
| Client's catch-all supplier key | In `labels_A.csv`, every ERP line with no `vendor_id` and a name never seen with a vendor id shares one supplier_key (`SG-f85711`, 20,718 lines, 122 different names). I read this as an "unmapped vendor" bucket in the client's master data, not one real supplier, so I do **not** copy it. Those lines get their real supplier. This lowers agreement on that bucket by construction, but it is the right spend cube. |
| Item key | Union-find linking material number ↔ normalised (OCR-folded) description. The client's keys behave this way: one text under several material numbers is one item. Tool text joins ERP text at fuzzy ratio ≥ 93. A tool line matched to an ERP line takes that line's item, supplier and category. Blank or generic text with no material ("No Object", "Parts") becomes item = supplier + context + text, so all blanks don't merge into one item. |
| Dates | Parsed with the configured formats in order, then Excel serials, then a lenient parse. Missing, unparseable and out-of-range dates are flagged in diagnostics and never cause an exclusion. Day-first is assumed for `dd/mm/yyyy` (South African data). |
| Zero / 0.01 amounts | Kept as included lines with their (tiny) spend. They are real receipts, often service-entry placeholders. |

## Company B (8 ERPs, 25 currencies, USD)

### What the data looks like
- No status column. No supplier names, only vendor ids (used as the supplier key).
- `currency_rate` is filled on about half the lines and is sometimes wrong: some `EUR` and `CHF`
  lines carry about 18.8, the MXN rate, and many `USD` lines carry a rate of about 17.5.
- `invoice_date` ranges from 2018 to 2027 while `posting_date` is all 2023–2024.
- 17% of descriptions are blank. Others are truncated (`Microsoft Server Cloud Enrollm -`) or
  codes (`0124X70.089,24OZ,NL - 12OUT,ID&OD-SEE SPEC`), in many languages.
- The seed labels mark only `NON_PROCUREMENT` exclusions (450 of 3,000), driven by intercompany
  GL accounts (`IC06`, `IC Payables`, `I/C Payable`…), tax lines (`State & Local Taxes`,
  `Sales Tax Expense`, `GST Input Tax`) and a few intercompany vendors.

### Decisions
| Topic | Decision |
|---|---|
| USD lines | Taken as USD whatever `currency_rate` says (the rule in the brief). |
| Given rate | Used when present, unless it is more than 3× away from the table rate for that currency and month. Then the table rate is used and the line is flagged `given_rate_implausible->table`. |
| Missing rate | `fx_rates_usd.csv` for the invoice month. If that month is missing for the currency, the nearest month is used. If the invoice date is missing or outside 2023-01..2025-06, the posting date month is used instead and the line is flagged. |
| Intercompany | GL account text matching `IC<digits>`, `I/C`, `intercompany` → `NON_PROCUREMENT`. A vendor with ≥50% of its lines on such accounts (and ≥3 lines) is an intercompany vendor, so all its lines are excluded. |
| Taxes | Lines whose description starts with a tax phrase (`Sales Tax`, `State & Local Taxes`, `GST Input Tax`, `VAT`…) → `NON_PROCUREMENT`. Tax consulting, advisory or software is not matched. |
| Label-learned vendors | A vendor whose seed-labelled lines are ≥90% excluded (≥2 labelled lines) has its other lines excluded with the same reason. |
| Duplicates | Only lines identical in system, invoice, invoice line, vendor, amount, currency, description, material, quantity, GL and date are treated as one line keyed twice. |
| Reversals and credits | Negative lines and reversal pairs are kept as included lines, so they net out in spend. They are real accounting entries, not duplicates. |
| Seed labels | "Useful, not perfect": training labels with weight. A seed-labelled line keeps a 50% pull to its label but can be overruled by a confident model. |
| Accuracy (30% of seed labels held out) | Include/exclude agreement 98.6%. Category L1 95.6%, L3 93.0% line-weighted (98.2% spend-weighted). |

## Company C and the unseen tree

- The tool reads the tree file at run time. Level columns are detected by their trailing number,
  so 3 or 4 levels both work, and lines get the deepest leaf. A tree may repeat a name across
  levels (`Roof Bolts > Roof Bolts`); output keeps it exactly as written.
- Zero-label classification blends (1) multilingual sentence embeddings of the line against a
  descriptor per leaf (the leaf name with its ancestors; acronyms like `CM`, `RB`, `SC` expanded
  from the ancestor names) and (2) a lexical TF-IDF match against the leaf names plus optional
  **keyword hints** in the config. Then (3) a self-trained text model on confident lines, using
  only the description (see below), and (4) supplier smoothing.
- **Keyword hints are config, not code.** At the blind test I write a `taxonomy.hints` block for
  C's tree (a few words per leaf, from general knowledge of the industry, after looking at C's
  descriptions). On B, hints written this way lift zero-label accuracy from L1 40% → 98% and
  L3 27% → 85% by spend.
- **Self-training on the description only.** An early version trained the pseudo-label model on
  description + GL code + supplier. On B it turned a shared GL code (`PEUS`) into a rule and sent
  Microsoft licences to *Pallets*. Training on the description alone fixed it.

## Things I chose not to do
- No category names or company-specific words in code. Everything company-specific is in the
  YAML (column mapping, hints, extra rules).
- No exclusion just because a date is impossible or a description is blank. The spend is still
  real, so it is flagged in diagnostics instead.
