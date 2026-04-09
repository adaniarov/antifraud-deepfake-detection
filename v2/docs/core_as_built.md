# Core v2 — as-built (implementation record)

This document describes **what was actually built** on the v2 track: notebooks, interim files, and tooling. Use it together with the **normative specification** [`dataset_design_final.md`](dataset_design_final.md) and the compact contract [`dataset_contract.md`](dataset_contract.md). When they differ, the spec defines intent; this file and [`raw_sources_inventory.md`](raw_sources_inventory.md) record practice. Dated changes: [`CHANGELOG.md`](CHANGELOG.md).

---

## 1. Spec vs as-built

| Layer | Document |
|-------|----------|
| Target design, checklists, §7.2 counts | [`dataset_design_final.md`](dataset_design_final.md) |
| Non-negotiable invariants (labels, generators, symmetry) | [`dataset_contract.md`](dataset_contract.md) |
| Prepared row counts, raw exploration facts | [`raw_sources_inventory.md`](raw_sources_inventory.md) |
| Current sprint summary | [`project_status.md`](project_status.md) |

---

## 2. Human-side prepared data (`v2/data/interim/gathered/`)

All fraud + `financial_qa` + legitimate email/SMS listed below go through cleaning, URL masking, language filter, deduplication, and `length_bin` assignment.

**`length_bin`:** [`v2/config.py`](../config.py) — SMS short/medium/long token thresholds 20 / 60; email 100 / 400; QA 75 / 250.

**Language filters (as implemented in notebooks):**

- SMS + QA: `langdetect_v1`
- Email (Nazario, Nigerian, Enron ham, SpamAssassin ham): `english_like_ascii_v1`

### 2.1 Fraud and QA

| Source | Output file | Notebook | Notes |
|--------|-------------|----------|-------|
| Nazario | `nazario_prepared.jsonl` | [`nazario_prepare.ipynb`](../notebooks/03_dataset_creation/nazario_prepare.ipynb) | `scenario_family`: `phishing_email`; outlier cap ≤5000 tokens |
| Nigerian 419 | `nigerian_fraud_prepared.jsonl` | [`nigerian_fraud_prepare.ipynb`](../notebooks/03_dataset_creation/nigerian_fraud_prepare.ipynb) | `advance_fee_scam_email`; same outlier cap |
| SmishTank | `smishtank_prepared.jsonl` | [`smishtank_prepare.ipynb`](../notebooks/03_dataset_creation/smishtank_prepare.ipynb) | `fraud_sms_deceptive`; `length_bin` from config |
| Mendeley smishing | `mendeley_smishing_prepared.jsonl` | [`mendeley_smishing_prepare.ipynb`](../notebooks/03_dataset_creation/mendeley_smishing_prepare.ipynb) | Same family; Core subtype list aligned with SmishTank; two-pass URL mask |
| HC3 Finance | `financial_qa_prepared.jsonl` | [`hc3_finance_prepare.ipynb`](../notebooks/03_dataset_creation/hc3_finance_prepare.ipynb) | Human + chatgpt rows; `question_id` pairing; `financial_qa` |

### 2.2 Legitimate email / SMS

| Source | Output file | Notebook |
|--------|-------------|----------|
| Enron ham | `enron_ham_prepared.jsonl` | [`14_enron_ham_prepare.ipynb`](../notebooks/03_dataset_creation/14_enron_ham_prepare.ipynb) |
| SpamAssassin ham | `spamassassin_ham_prepared.jsonl` | [`15_spamassassin_ham_prepare.ipynb`](../notebooks/03_dataset_creation/15_spamassassin_ham_prepare.ipynb) |
| SMS ham | `sms_ham_prepared.jsonl` | [`16_sms_ham_prepare.ipynb`](../notebooks/03_dataset_creation/16_sms_ham_prepare.ipynb) |

**Row counts (machine count, `wc -l` on `gathered/*.jsonl`):** see table in [`raw_sources_inventory.md`](raw_sources_inventory.md) §1.

**Deviation from §7.2:** combined SMS fraud (SmishTank + Mendeley) = **852** rows vs target **1 200–1 800** — documented as data availability / policy constraint in [`project_status.md`](project_status.md).

---

## 3. Diagnostic ham audits (`v2/data/interim/annotated/`)

Not part of `gathered/`; used to characterize legitimate registers for prompts.

| Output | Rows | Notebook | Model (per project_status) |
|--------|-----:|----------|----------------------------|
| `enron_ham_annotated.jsonl` | 420 | `06_enron_ham_annotation.ipynb` | `openai/gpt-4o-mini` via OpenRouter |
| `spamassassin_ham_annotated.jsonl` | 320 | `07_spamassassin_ham_annotation.ipynb` | same |
| `sms_ham_annotated.jsonl` | 320 | `08_sms_ham_annotation.ipynb` | same |

Stratification: `subset` / `archive` × word-count bins as in those notebooks.

---

## 4. LLM-side mass generation (Dataset 2)

| Component | Path |
|-----------|------|
| Library | [`v2/src/llm_mass_generation.py`](../src/llm_mass_generation.py) |
| Seen — OpenAI | [`11_mass_generation_openai.ipynb`](../notebooks/03_dataset_creation/11_mass_generation_openai.ipynb) |
| Seen — Mistral | [`12_mass_generation_mistral.ipynb`](../notebooks/03_dataset_creation/12_mass_generation_mistral.ipynb) |
| Holdout — Claude | [`13_mass_generation_claude_openrouter.ipynb`](../notebooks/03_dataset_creation/13_mass_generation_claude_openrouter.ipynb) |

**Outputs:** `v2/data/interim/llm-generated/core_llm_<lane>_<model_slug>.jsonl` where `<lane>` is `seen` or `holdout`.

**Behavior (summary):**

- Single logical grid of tasks `(family, subtype, target_bin, idx)`; OpenAI and Mistral partition **seen** slots without overlap using `crc32` quantiles; share controlled by env **`OPENAI_SEEN_SHARE`** (see notebooks / [`project_status.md`](project_status.md)).
- Claude runs the **full** grid for holdout with `split="test"`; must not enter train/val ([`dataset_contract.md`](dataset_contract.md)).
- Append-only JSONL, resume by `generation_job_id`; QC and near-dedup in module; concurrency via **`LLM_GEN_MAX_WORKERS`**.

**Prompt + validation contract:** [`llm_prompt_families_contract.md`](llm_prompt_families_contract.md). JSON specs: `v2/data/prompts/*.json`.

---

## 5. Assembled Core v2 (frozen splits)

| Step | Notebook | Output (under `v2/data/interim/assembled/`) |
|------|----------|-----------------------------------------------|
| Human table | [`17_dataset1_human_assembly.ipynb`](../notebooks/03_dataset_creation/17_dataset1_human_assembly.ipynb) | `dataset1_human.jsonl` |
| LLM table | [`18_dataset2_llm_assembly.ipynb`](../notebooks/03_dataset_creation/18_dataset2_llm_assembly.ipynb) | `dataset2_llm.jsonl` |
| Core + splits + manifest | [`19_core_train_val_test_assembly.ipynb`](../notebooks/03_dataset_creation/19_core_train_val_test_assembly.ipynb) | `core_v2.jsonl`, `core_train.jsonl`, `core_val.jsonl`, `core_test.jsonl` (= test_full), `core_test_full.jsonl`, `core_test_non_claude.jsonl`, `core_test_claude_only.jsonl`, `core_manifest.json`, `core_split_diagnostics.json` |

**Semantics:** `assembly_policy` = `core_v2_seen_balanced_plus_claude_holdout` (see manifest). Main pool **human ↔ seen_llm** balance for train / val / `test_seen`; Claude holdout rows only in **test_claude_only**. Field **`core_eval_slice`**: `train` \| `val` \| `test_seen` \| `test_claude_holdout`.

**Docs:** [`core_dataset_description.md`](core_dataset_description.md), [`core_split_diagnostics.md`](core_split_diagnostics.md); crosstabs: `v2/outputs/tables/core_crosstab_*.csv`.

**Experimental reporting:** report metrics separately for **val**, **test_non_claude**, **test_claude_only**; **test_full** only as a supplementary aggregate ([`project_status.md`](project_status.md), [`thesis_constraints.md`](thesis_constraints.md)).

### 5.1 Feature extraction & diagnostic EDA

| Notebook | Artifacts |
|----------|-----------|
| [`01_core_feature_extraction_and_eda.ipynb`](../notebooks/04_features/01_core_feature_extraction_and_eda.ipynb) | `v2/data/interim/features/` (dense + **68 legacy HC** `hc_*` in `core_dense_features.parquet`, logic in [`v2/src/core_legacy_hc_features.py`](../src/core_legacy_hc_features.py); TF-IDF `core_tfidf_*.npz` + vectorizer pickles; optional LM cache after `uv sync --extra lm_scoring`); `v2/outputs/tables/features/`, `v2/outputs/figures/features/` |

### 5.2 Classical ML baselines (Core)

| Notebook | Artifacts / doc |
|----------|-----------------|
| [`01_core_classical_baselines.ipynb`](../notebooks/05_classical_ml/01_core_classical_baselines.ipynb) | `v2/outputs/tables/classical_ml/`, `v2/outputs/figures/classical_ml/`; narrative summary [`baseline_summary.md`](baseline_summary.md) |

## 6. Not yet as-built

- **Neural / transformer baselines** and systematic hyperparameter search on frozen Core (классические sparse/dense baseline — см. §5.2 и [`baseline_summary.md`](baseline_summary.md)).
- Optional: systematic robustness / stress-tests **outside** Core.

Normative checklists: [`dataset_design_final.md`](dataset_design_final.md) §8 / §10; queue: [`next_tasks.md`](next_tasks.md).
