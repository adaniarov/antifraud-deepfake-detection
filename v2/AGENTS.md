# AGENTS.md
# Project agent contract for Cursor — **v2 / Core track**

## Project
Bachelor thesis / ML research project:
"Detection of Textual Deepfake Attacks in Anti-Fraud Systems"

This subdirectory (`v2/`) follows the **Core dataset** specification: cleaner factorial design, matched scenario families, no monolithic broad-spam fraud classes. The authoritative design is `v2/docs/dataset_design_final.md`.

## Mission
Build, evaluate, and document robust detectors for human-written vs LLM-generated **anti-fraud-domain** texts, on a Core dataset that is **smaller but methodologically cleaner** than legacy v1.

## Research scope
Binary classification:
- `label = 0` → human-written
- `label = 1` → LLM-generated

Domain: anti-fraud-relevant communications represented in Core (email, SMS, and a bilateral QA control slice), with explicit axes:
- `fraudness` ∈ `{fraud, legitimate}`
- `channel` ∈ `{email, sms, qa}`
- `scenario_family` (matched human ↔ LLM where generation applies)

Core **scenario families** (see design doc for sources and exclusions):
- Fraud: `phishing_email`, `advance_fee_scam_email`, `fraud_sms_deceptive`
- Legitimate: `legitimate_email`, `legitimate_sms`, `financial_qa` (HC3 Finance paired; no new LLM prompt family)

## Current research status (v2)
- Literature review and thesis framing exist at repo root; **Core dataset** is specified in `dataset_design_final.md`.
- **Frozen Core v2** splits and manifest: `v2/data/interim/assembled/core_manifest.json`, `core_*.jsonl`; assembly notebook `v2/notebooks/03_dataset_creation/19_core_train_val_test_assembly.ipynb`. Summary: `v2/docs/core_dataset_description.md`. **Experiments:** report metrics separately for **val**, **test_non_claude**, **test_claude_only**; **test_full** only as a supplementary aggregate — see `v2/docs/project_status.md` and `v2/docs/thesis_constraints.md`.
- **Final assembled Core** for v2 is **not** the same as legacy `data/final/` — do not mix v1 numbers with Core v2.

Always read `v2/docs/project_status.md` before proposing next steps for work inside `v2/`.

## Source-of-truth files (v2)
Prefer these over assumptions when working in `v2/`:
- `v2/docs/README.md` — index of all v2 docs and how they relate
- `v2/docs/dataset_design_final.md` — **primary** design and checklist
- `v2/docs/dataset_contract.md` — compact invariants (must stay consistent with the design doc)
- `v2/docs/core_as_built.md` — implemented pipeline vs spec
- `v2/docs/raw_sources_inventory.md` — prepared sources, Mendeley, legacy exploration facts
- `v2/docs/CHANGELOG.md` — dated significant changes
- `v2/docs/llm_prompt_families_contract.md` — LLM generation contract (Dataset 2)
- `v2/docs/project_status.md`
- `v2/docs/project_overview.md`
- `v2/docs/thesis_constraints.md`
- `v2/docs/next_tasks.md`

Superseded design drafts (historical only): `v2/docs/archive/`.

For **legacy v1** numbers and baselines only (not automatically valid for Core):
- repo root `outputs/tables/classical_results_no_ppl.csv`, `data/final/dataset_stats.json`, `docs/dataset_contract.md`

## Hard constraints
- Never invent metric values or sample sizes for v2; if not in a committed artifact, say **TBD**.
- Never state a result unless it comes from a real file.
- Never suggest placing temperature inside prompt text; `temperature` is an **API parameter**.
- URL masking, deduplication, and normalization must be **symmetric** across human and LLM sides.
- **Claude-family generator stays test-only**; no Claude-generated rows in train/val.
- Distinguish confirmed findings vs hypotheses; cite dataset limitations when interpreting.

## Evaluation protocol (intended for Core, once built)
When discussing quality, prefer:
1. Validation performance
2. Test full
3. Test **Claude** partition (held-out generator)
4. Test **non-Claude** partition (seen generators)
5. Breakdowns by `scenario_family`, `channel`, `fraudness`, and key subtype fields where populated

## Coding behavior
- Inspect `v2/` layout before coding; keep paths explicit.
- Prefer minimal diffs; preserve reproducibility.
- Do not silently change label semantics or scenario-family definitions relative to `dataset_design_final.md`.
- Add comments only where they aid understanding.

## Writing behavior
When drafting thesis text:
- formal academic Russian by default; code and prompts in English
- avoid hype; note temporal bias, channel imbalance, and source-artifact risks when relevant
- do not overclaim generalization beyond Core scope
