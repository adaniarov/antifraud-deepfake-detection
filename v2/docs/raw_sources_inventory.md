# Raw and prepared data sources (v2 Core)

Inventory of **Core-relevant** human-side inputs: where raw data is explored, how it is prepared, and what lands in `v2/data/interim/gathered/`. For the normative dataset definition see [`dataset_design_final.md`](dataset_design_final.md) and [`dataset_contract.md`](dataset_contract.md).

Prepared row counts below are **`wc -l` on JSONL** under `v2/data/interim/gathered/` at documentation time (one line = one record).

---

## 1. Core v2 — prepared human lines (`gathered/`)

| `scenario_family` | Human source (conceptual) | Preparation notebook | Output file | Language / notes | Rows |
|-------------------|---------------------------|----------------------|-------------|------------------|------|
| `phishing_email` | Nazario Phishing | [`nazario_prepare.ipynb`](../notebooks/03_dataset_creation/nazario_prepare.ipynb) | `nazario_prepared.jsonl` | `english_like_ascii_v1`; outlier cap ≤5000 tokens; `length_bin` from [`config.py`](../config.py) | 5201 |
| `advance_fee_scam_email` | Nigerian Fraud (419-style) | [`nigerian_fraud_prepare.ipynb`](../notebooks/03_dataset_creation/nigerian_fraud_prepare.ipynb) | `nigerian_fraud_prepared.jsonl` | same email policy as Nazario | 3234 |
| `fraud_sms_deceptive` | SmishTank | [`smishtank_prepare.ipynb`](../notebooks/03_dataset_creation/smishtank_prepare.ipynb) | `smishtank_prepared.jsonl` | `langdetect_v1`; `length_bin` from `config.py` (SMS thresholds) | 483 |
| `fraud_sms_deceptive` | Mendeley SMS Phishing (smishing subset) | [`mendeley_smishing_prepare.ipynb`](../notebooks/03_dataset_creation/mendeley_smishing_prepare.ipynb) | `mendeley_smishing_prepared.jsonl` | `langdetect_v1`; Core subtype policy aligned with SmishTank (see §2) | 369 |
| `legitimate_email` | Enron ham | [`14_enron_ham_prepare.ipynb`](../notebooks/03_dataset_creation/14_enron_ham_prepare.ipynb) | `enron_ham_prepared.jsonl` | `english_like_ascii_v1` | 15088 |
| `legitimate_email` | SpamAssassin ham | [`15_spamassassin_ham_prepare.ipynb`](../notebooks/03_dataset_creation/15_spamassassin_ham_prepare.ipynb) | `spamassassin_ham_prepared.jsonl` | `english_like_ascii_v1` | 3753 |
| `legitimate_sms` | SMS ham | [`16_sms_ham_prepare.ipynb`](../notebooks/03_dataset_creation/16_sms_ham_prepare.ipynb) | `sms_ham_prepared.jsonl` | `langdetect_v1` | 4061 |
| `financial_qa` | HC3 Finance (human + chatgpt answers) | [`hc3_finance_prepare.ipynb`](../notebooks/03_dataset_creation/hc3_finance_prepare.ipynb) | `financial_qa_prepared.jsonl` | `langdetect_v1`; paired via `question_id`; human `label=0`, chatgpt `label=1` | 7842 |

**SMS fraud combined:** 483 + 369 = **852** rows (below the §7.2 target range 1 200–1 800 — data availability / policy, not a pipeline bug; see [`project_status.md`](project_status.md)).

Channel-level `length_bin` token thresholds are defined in [`config.py`](../config.py) (SMS 20/60, email 100/400, QA 75/250), consistent with [`dataset_design_final.md`](dataset_design_final.md).

---

## 2. Mendeley SMS Phishing dataset (Core smishing branch)

**Dataset:** *SMS Phishing Dataset for Machine Learning and Pattern Recognition* (Mendeley Data). In Core it supplies part of **`fraud_sms_deceptive`**, together with SmishTank, under a **single** subtype policy ([`dataset_design_final.md`](dataset_design_final.md) §5.7–§5.8).

**Preparation:** [`mendeley_smishing_prepare.ipynb`](../notebooks/03_dataset_creation/mendeley_smishing_prepare.ipynb) — text extraction, URL masking (two-pass), language filter, deduplication, `sms_fraud_subtype` / Core inclusion flags, `length_bin` from `config.py`.

**April 2026 policy fix:** `financial_or_crypto_lure` and `loan_or_credit_lure` were **removed** from `CORE_SUBTYPES` in this notebook so Mendeley stays aligned with the SmishTank Core policy (see [`CHANGELOG.md`](CHANGELOG.md)). Effect: 388 → 369 rows.

**Annotated pilot / QC distribution** (counts of rows by `scenario_family` label in [`v2/outputs/tables/mendeley_smishing_annotation_summary.csv`](../outputs/tables/mendeley_smishing_annotation_summary.csv)):

| scenario_family (annotation axis) | count |
|-----------------------------------|------:|
| account_alert | 10 |
| delivery_fee_or_service_issue | 6 |
| financial_or_crypto_lure | 2 |
| generic_deceptive_sms | 27 |
| loan_or_credit_lure | 3 |
| prize_or_contest_scam | 51 |
| unclear_other | 1 |

(This table describes the annotation summary artifact, not the full prepared corpus row counts.)

---

## 3. Diagnostic ham audits (not `gathered/`)

Stratified LLM-assisted audits for prompt design; outputs in `v2/data/interim/annotated/`. Sizes from [`project_status.md`](project_status.md):

| Source | File | Rows | Notebooks |
|--------|------|-----:|-----------|
| Enron ham | `enron_ham_annotated.jsonl` | 420 | `06_enron_ham_annotation.ipynb` |
| SpamAssassin ham | `spamassassin_ham_annotated.jsonl` | 320 | `07_spamassassin_ham_annotation.ipynb` |
| SMS ham | `sms_ham_annotated.jsonl` | 320 | `08_sms_ham_annotation.ipynb` |

---

## 4. Legacy exploration (`data/raw/collected/`)

The following facts come from **early inventory** in [`v2/notebooks/01_data_sources/01_explore_sources.ipynb`](../notebooks/01_data_sources/01_explore_sources.ipynb) over repository-root **`data/raw/collected/`** (six JSONL files: Nazario, SMS collection, Enron, SpamAssassin, HC3 Finance, HC3 Open QA). They are useful for **raw-scale and era** context; they are **not** the v2 Core assembly counts above. Mendeley and Nigerian Fraud corpora were added on the v2 track after this exploration.

### 4.1 Record counts (raw exploration)

| Source | File | N raw | N unique | Dup% | Median words | has_url% |
|--------|------|------:|---------:|-----:|-------------:|---------:|
| Nazario Phishing | nazario.jsonl | 8 472 | 5 820 | 31% | 136 | 48% |
| SMS spam | sms_spam.jsonl | 747 | 642 | 14% | 25 | 3% |
| SMS ham | sms_spam.jsonl | 4 827 | 4 518 | 6% | 11 | 0% |
| Enron spam (all) | enron.jsonl | 17 171 | — | — | 128 | 0% |
| Enron spam (419 filter) | enron.jsonl | 872 | 731 | 16% | 513 | 0% |
| Enron ham | enron.jsonl | 16 545 | 15 794 | 5% | 169 | 0% |
| SpamAssassin spam | spamassassin.jsonl | 3 293 | 1 630 | 51% | 154 | 54% |
| SpamAssassin ham | spamassassin.jsonl | 3 900 | 3 861 | 1% | 137 | 86% |
| HC3 Finance human | hc3_finance.jsonl | 3 933 | 3 933 | 0% | 129 | 7% |
| HC3 Finance chatgpt | hc3_finance.jsonl | 4 503 | 4 462 | 1% | 205 | 7% |
| HC3 Open QA human | hc3_open_qa.jsonl | 1 187 | 1 171 | 1% | 27 | — |
| HC3 Open QA chatgpt | hc3_open_qa.jsonl | 3 546 | 3 518 | 1% | 106 | — |

### 4.2 Nazario — per-record year (Date header)

| Year | Records |
|------|--------:|
| 1999–2004 | 132 |
| 2005 | 1 541 |
| 2006 | 1 875 |
| 2007 | 1 303 |
| 2008–2014 | 0 |
| 2015 | 305 |
| 2016 | 496 |
| 2017 | 321 |
| 2018 | 287 |
| 2019 | 241 |
| 2020 | 157 |
| 2021 | 101 |
| 2022 | 248 |
| 2023 | 362 |
| 2024 | 400 |
| 2025 | 478 |
| no date | 225 |

Pre-2010 legacy archives: **5 009** records; 2015–2025 archives: **3 463**; **8 247 / 8 472** rows with parseable year.

### 4.3 Source-level periods, word-count percentiles, fields, 419 keyword filter

- **Periods (approx.):** SMS Spam Collection 2006–2012; Enron ~2000–2002; SpamAssassin 2003–2005; HC3 human 2008–2022; HC3 chatgpt 2022–2023.
- **Word percentiles (exploration):** Nazario p25/p50/p75/p95 ≈ 63 / 136 / 235 / 366; Enron 419 filter ≈ 365 / 513 / 661 / 1 683; SMS spam ~10 / 25 / ~40 / ~60; SMS ham ~5 / 11 / ~20 / ~40.
- **Typical fields:** email JSONL — `text`, `subject`, `from`, `date`, `source` / `archive` / `subset`, URL flags; SMS — `text`, `label`, URL flags; HC3 — `text`, `role`, `question`, `question_id`, URL flags. Nazario: subject present 8 414 / 8 472.
- **419 keyword filter on Enron spam:** 872 / 17 171; on SpamAssassin spam: 277 / 3 293 (same keyword list as in the exploration notebook).

### 4.4 Obsolete planning table (superseded)

Older notes mapped sources to legacy `generate.py` tasks (T1–T6) and content types such as `smishing` / `open_qa` as core candidates. **Current Core** uses only the `scenario_family` set in [`dataset_contract.md`](dataset_contract.md) (e.g. `fraud_sms_deceptive`, not a separate legacy `smishing` type; HC3 Open QA is **out of Core** per design). Do not use §4.4 for implementation.

### 4.5 Temporal gap (human era vs LLM generation)

| Human source | Era | Gap to 2024–2025 LLM text |
|--------------|-----|---------------------------|
| Enron | ~2000–2002 | large |
| SpamAssassin | 2003–2005 | large |
| Nazario legacy | 2005–2007 | large |
| SMS collection | 2006–2012 | medium–large |
| Nazario 2015–2025 | 2015–2025 | smaller |
| HC3 Finance human | 2008–2022 | medium |
| HC3 Finance chatgpt | 2022–2023 | small |

---

## 5. Invariants (pointer)

Generator policy, label semantics, URL masking symmetry, and Claude holdout rules are stated in [`dataset_contract.md`](dataset_contract.md) and [`v2/AGENTS.md`](../AGENTS.md). For dated pipeline changes see [`CHANGELOG.md`](CHANGELOG.md).
