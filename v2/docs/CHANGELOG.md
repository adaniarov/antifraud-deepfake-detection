# Changelog — v2 / Core track

Significant changes to data pipeline, documentation, generator policy, and dataset semantics. **Normative spec** remains [`dataset_design_final.md`](dataset_design_final.md) unless an entry explicitly says the spec was amended.

## 2026-04-08

- **Core v2 frozen assembly:** Notebooks `17_dataset1_human_assembly.ipynb`, `18_dataset2_llm_assembly.ipynb`, `19_core_train_val_test_assembly.ipynb` produce `v2/data/interim/assembled/core_*.jsonl` and `core_manifest.json`. Policy: **main comparable** train/val/test_seen balanced **human vs seen_llm**; **Claude holdout** additional test-only slice (`test_claude_only`); `assembly_policy` = `core_v2_seen_balanced_plus_claude_holdout`. User-facing summary: [`core_dataset_description.md`](core_dataset_description.md); split tables: [`core_split_diagnostics.md`](core_split_diagnostics.md), `core_split_diagnostics.json`, `v2/outputs/tables/core_crosstab_*.csv`.
- **Docs — experiment / thesis protocol:** [`project_status.md`](project_status.md), [`project_overview.md`](project_overview.md), [`next_tasks.md`](next_tasks.md), [`thesis_constraints.md`](thesis_constraints.md), [`core_as_built.md`](core_as_built.md) updated: mandatory separate reporting for **val**, **test_non_claude**, **test_claude_only**; **test_full** only as supplementary aggregate; VKR must state main-pool balance, Claude as holdout stress-test, multi-genre benchmark, temporal bias, generator limits, SMS fraud data-limited.
- **Documentation layout:** Intermediate design drafts moved to [`archive/`](archive/README.md). Removed `opesource_description.md`; replaced with [`raw_sources_inventory.md`](raw_sources_inventory.md) (Core prepared sources + Mendeley subsection + legacy exploration appendix). Added [`core_as_built.md`](core_as_built.md), this file, [`README.md`](README.md) (docs index), and [`llm_prompt_families_contract.md`](llm_prompt_families_contract.md) (prompt generation contract; canonical copy under `docs/`).
- **Prompt contract location:** Full text moved from `v2/data/prompts/README.md` to `v2/docs/llm_prompt_families_contract.md`; prompts folder README is a short pointer.

## 2026-04 (human `gathered/` audit and fixes)

- **`smishtank_prepare.ipynb`:** Removed hardcoded `length_bin='short'` for all rows; 180/483 rows (40–197 tokens) should be `medium`. Now uses `compute_length_bin` from [`config.py`](../config.py).
- **`mendeley_smishing_prepare.ipynb`:** Removed `financial_or_crypto_lure` and `loan_or_credit_lure` from `CORE_SUBTYPES` to align Mendeley with SmishTank Core policy ([`dataset_design_final.md`](dataset_design_final.md) §5.7–§5.8); **−19** rows (388→369). Added second `mask_url` regex pass (unmasked URL assertion).
- **`nazario_prepare.ipynb`:** Outlier cap **≤5000** tokens (excluded ~308k-token parsing artifact).
- **`nigerian_fraud_prepare.ipynb`:** Outlier cap **≤5000** tokens; `length_bin` from `config.py` (removed local `length_bin_email()`).
- **`v2/config.py`:** Introduced as single source of truth for channel-level `length_bin` token thresholds (SMS 20/60, email 100/400, QA 75/250). Analysis notebook: [`00_length_bin_analysis.ipynb`](../notebooks/03_dataset_creation/00_length_bin_analysis.ipynb).

## 2026-04 (prompts and LLM mass generation)

- **Prompt families:** Five JSON families finalized under `v2/data/prompts/` (`phishing_email`, `advance_fee_scam_email`, `fraud_sms_deceptive`, `legitimate_email`, `legitimate_sms`). `financial_qa` uses HC3 chatgpt side only (no new prompt family). See [`llm_prompt_families_contract.md`](llm_prompt_families_contract.md).
- **Mass generation:** Shared module [`llm_mass_generation.py`](../src/llm_mass_generation.py) and notebooks `11_mass_generation_openai.ipynb`, `12_mass_generation_mistral.ipynb`, `13_mass_generation_claude_openrouter.ipynb`. Outputs: `v2/data/interim/llm-generated/core_llm_<lane>_<model_slug>.jsonl`, append-only, resume via `generation_job_id`. Seen lane: OpenAI vs Mistral slot split via **`OPENAI_SEEN_SHARE`** (default in notebooks noted in [`project_status.md`](project_status.md)); parallelism via **`LLM_GEN_MAX_WORKERS`**. Claude: full grid, **`split="test"`** only (holdout).

## Maintenance

- After each material change to data policy, tooling, or docs, add a dated entry here (newest first).
- Also update [`project_status.md`](project_status.md) when milestones shift.
