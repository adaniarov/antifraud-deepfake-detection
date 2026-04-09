# v2 documentation index

Documentation for the **Core** track under `v2/`. Legacy v1 lives at the repository root (`docs/`, `data/final/`).

## Source-of-truth stack

| Order | Document | Role |
|------:|----------|------|
| 1 | [`dataset_design_final.md`](dataset_design_final.md) | Normative Core specification and checklists. |
| 2 | [`dataset_contract.md`](dataset_contract.md) | Short invariants (must match the design doc). |
| 3 | [`core_as_built.md`](core_as_built.md) | What was actually implemented (notebooks, paths, tooling). |
| 4 | [`raw_sources_inventory.md`](raw_sources_inventory.md) | Prepared `gathered/` inventory, Mendeley notes, legacy raw exploration facts. |
| 5 | [`CHANGELOG.md`](CHANGELOG.md) | Dated log of significant changes. |

## Working docs

| File | Role |
|------|------|
| [`project_status.md`](project_status.md) | Current stage, tables, risks; update after milestones. |
| [`next_tasks.md`](next_tasks.md) | Prioritized queue vs §8 / §10 of the design doc. |
| [`project_overview.md`](project_overview.md) | High-level goals and relation to legacy v1. |
| [`thesis_constraints.md`](thesis_constraints.md) | Writing rules for the thesis text (Core); VKR bullets on final dataset + metric reporting protocol. |
| [`core_dataset_description.md`](core_dataset_description.md) | Human-readable summary of frozen Core v2 (splits, balance policy, test files). |
| [`llm_prompt_families_contract.md`](llm_prompt_families_contract.md) | LLM generation format, masking, length, QC (Dataset 2). |

## Archive

[`archive/README.md`](archive/README.md) — superseded design drafts (`dataset_design_v0`–`v2`); use only for history.

## Agent / contributor note

Cursor agents working in `v2/` should read [`../AGENTS.md`](../AGENTS.md) and [`project_status.md`](project_status.md) before non-trivial changes.
