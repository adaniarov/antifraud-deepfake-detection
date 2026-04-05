# AGENTS.md
# Project agent contract for Cursor

## Project
Bachelor thesis / ML research project:
"Detection of Textual Deepfake Attacks in Anti-Fraud Systems"

## Mission
Build, evaluate, and document robust detectors for human-written vs LLM-generated anti-fraud texts.

## Research scope
The task is binary classification:
- label = 0 -> human-written
- label = 1 -> LLM-generated

Domain is restricted to anti-fraud text:
- phishing email
- smishing SMS
- social engineering support message
- 419 / advance-fee scam letter
- bank / financial notification
- financial review / feedback

## Current research status
The project already has:
- completed literature review chapter
- dataset design and assembled splits
- classical ML baseline implemented
- transformer fine-tuning planned
- ensembles planned
- robustness evaluation planned

Always read `docs/project_status.md` before proposing next steps.

## Source-of-truth files
Always prefer these files over assumptions:
- `docs/project_status.md`
- `docs/project_overview.md`
- `docs/dataset_contract.md`
- `docs/thesis_constraints.md`
- `outputs/tables/classical_results_no_ppl.csv`
- `data/final/dataset_stats.json`

## Hard constraints
- Never invent metric values.
- Never state a result unless it comes from a real file.
- Never suggest placing temperature inside prompt text.
- Temperature must be treated as an API parameter.
- URL masking must be symmetric across all groups.
- Held-out Claude generator must stay test-only.
- Distinguish clearly between confirmed findings and hypotheses.
- Any scientific interpretation must mention dataset limitations if relevant.

## Dataset design
Factorial structure:
- A: Human + Fraudulent
- B: Human + Legitimate
- C: LLM + Fraudulent
- D: LLM + Legitimate

Content types:
- T1: phishing email
- T2: smishing SMS
- T3: social engineering support message
- T4: 419 scam
- T5: bank/financial notification
- T6: financial review

## Evaluation protocol
Whenever discussing final quality, prefer:
1. Validation performance
2. Test Full
3. Test Claude partition
4. Test Non-Claude partition
5. Per-content-type results

## Coding behavior
- Inspect repository structure before coding.
- Prefer minimal diffs.
- Preserve reproducibility.
- Do not silently change dataset semantics.
- Add comments only where they help understanding.
- When editing scripts, keep paths/configs explicit and deterministic.
- After edits, suggest or run a minimal validation step if available.

## Writing behavior
When drafting thesis text:
- use formal academic Russian by default
- keep code and prompts in English
- avoid hype
- mention risks of temporal bias and genre imbalance when relevant
- do not overclaim generalization
