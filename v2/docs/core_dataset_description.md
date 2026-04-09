# Core v2 — описание итогового датасета

Сгенерировано из `19_core_train_val_test_assembly.ipynb` (UTC: 2026-04-08T23:04:58.210993+00:00).

## Источник истины и политика сборки

Спецификация: `v2/docs/dataset_design_final.md`, `v2/docs/dataset_contract.md`. Сборка **не** меняет список `scenario_family`, источники, preprocessing, парную логику `financial_qa` и правило **Claude только в test**.

### Что исправлено относительно предыдущей сборки

- Раньше family-level баланс фактически считался как human ↔ (seen_llm + Claude holdout); после изъятия Claude в `test_claude_only` основные сплиты (train/val/test_seen) становились **human-heavy**.
- Теперь **train, val, test_seen** балансируются как **human ↔ seen_llm** с бюджетом `main_seen_budget_per_side = min(target_cap, n_human_raw, n_seen_llm_raw)`. Claude **не** участвует в этом бюджете.
- **Claude holdout** — отдельный add-on: `claude_holdout_budget = min(round(main_seen_budget * CLAUDE_HOLDOUT_REL_FRAC), n_claude_raw)` (дефолт **0.25**), только `core_eval_slice=test_claude_holdout`.

## Два смысла баланса на уровне семьи

- **Main comparable balance:** human ↔ seen_llm в train/val/test_seen (и в сумме main pool).
- **Дополнительный срез:** Claude holdout добавляет LLM-строки только в `test_claude_only`, без участия в бюджете main.

- `assembly_policy`: **core_v2_seen_balanced_plus_claude_holdout** (см. `core_manifest.json`).

## Цель

Стратифицированный downsampling и **симметричный** сплит 75/15/10 для main pool (human + seen_llm по каждой семье); holdout Claude только в test. `financial_qa` — только полные пары (bilateral seen-only control).

## Downsampling

1. **Human (main):** до `main_seen_budget_per_side` со стратами `length_bin` + при необходимости `source_family` (legitimate / fraud SMS), `time_band` (phishing / advance fee).
2. **Seen LLM (main):** до того же бюджета; квота OpenAI vs Mistral **пропорционально сырому объёму**, внутри — стратификация по `length_bin` (см. `manifest.downsampling.seen_llm_split_quota_rule`).
3. **Claude holdout:** отдельно до `claude_holdout_budget`, страта `length_bin`; параметр `CLAUDE_HOLDOUT_REL_FRAC=0.25`.
4. **`financial_qa`:** только полные пары `question_id`; выбор до `TARGET_PAIRS_FQ` со стратификацией пар по `length_bin` human-стороны. Claude для этой семьи не используется.

- Исходные объёмы: dataset1 **36120** строк, dataset2 **21977** строк.
- После отбора: **20135** строк.

## Метод сплита

- Seed `42`; доли **main pool**: 75% / 15% / 10% для **каждой** non-FQ семьи симметрично на human и seen_llm (одинаковые целочисленные квоты train/val/test).
- Страты human: `human_strata_cols`; seen_llm: `length_bin` + `generator_lane` (+ `origin_model` при вариативности).
- Все строки Claude → `split=test`, `core_eval_slice=test_claude_holdout` (не участвуют в train/val/test_seen).

## Интерпретация test

- **`test_non_claude`** (`core_test_non_claude.jsonl`, `test_seen`) — **основной сопоставимый** test (human vs seen LLM).
- **`test_claude_only`** — **unseen-generator / holdout** stress-test.
- **`test_claude_binary`** (`core_test_claude_binary.jsonl`) — **бинарный** robustness-test: `label=1` = Claude holdout, `label=0` = human companions из **резерва** dataset1 (строки, **не** попавшие в train/val/test_seen). Согласование страт: `channel`, `fraudness`, `length_bin`, при необходимости более грубые тиры (см. `v2/src/core_test_claude_binary.py`, `core_manifest.json` → `test_claude_binary`).
- **`test_full`** — объединение test_seen и Claude holdout; **операционный** полный test, может быть LLM-heavy из-за Claude; не подменяет интерпретацию основных сплитов.

### Ограничение `test_claude_binary`

Для **`fraud_sms_deceptive`** весь human-пул уже израсходован в main comparable pool → **резерв пуст**, пары Claude↔human для этой семьи **не формируются** (см. таблицу availability в `core_split_diagnostics.md`, блок `test_claude_binary`). Остальные семьи с Claude holdout покрыты полностью (по числу Claude-строк).

## Эксперименты и текст ВКР (жёсткий протокол)

Главные метрики показывать **отдельно** для **val**, **test_non_claude**, **`test_claude_binary`** (бинарный перенос на Claude), и **`test_claude_only`** (one-class сигнал). **`test_full`** использовать только как **дополнительный** агрегированный срез; **не** делать его единственной главной цифрой качества.

В ВКР явно зафиксировать: main comparable pool **balanced human vs seen_llm**; Claude — **отдельный holdout stress-test**; Core — **multi-genre** бенчмарк; есть **temporal bias** и ограничения покрытия генераторов; срез **SMS fraud** **data-limited**. Подробнее: `v2/docs/project_status.md`, `v2/docs/thesis_constraints.md`.

## Объёмы по split

| область | count |
|---------|------:|
| train | 13554 |
| val | 2706 |
| test_full | 3875 |
| test_non_claude (test_seen) | 1816 |
| test_claude_only | 2059 |
| **total** | **20135** |

- Строк с `generator_lane == holdout_claude`: **2059**.

## Распределение по `channel` × `split`

| | train | val | test | **Σ** |
|---|:---|:---|:---|---:|
| `email` | 9900 | 1980 | 2970 | **14850** |
| `qa` | 1198 | 238 | 164 | **1600** |
| `sms` | 2456 | 488 | 741 | **3685** |

## Распределение по `scenario_family` × `split`

| | train | val | test | **Σ** |
|---|:---|:---|:---|---:|
| `advance_fee_scam_email` | 2400 | 480 | 720 | **3600** |
| `financial_qa` | 1198 | 238 | 164 | **1600** |
| `fraud_sms_deceptive` | 1278 | 254 | 385 | **1917** |
| `legitimate_email` | 3000 | 600 | 900 | **4500** |
| `legitimate_sms` | 1178 | 234 | 356 | **1768** |
| `phishing_email` | 4500 | 900 | 1350 | **6750** |

## Генерация по `scenario_family` (модель / lane)

Для **human** (`label=0`) — `(human)`. Для **LLM** — `generator_lane`, иначе `llm:` + `origin_model` (часто для `financial_qa`).

| | (human) | holdout_claude | `llm:openai/chatgpt` | seen_mistral | seen_openai | **Σ** |
|---|:---|:---|:---|:---|:---|---:|
| `advance_fee_scam_email` | 1600 | 400 | 0 | 626 | 974 | **3600** |
| `financial_qa` | 800 | 0 | 800 | 0 | 0 | **1600** |
| `fraud_sms_deceptive` | 852 | 213 | 0 | 349 | 503 | **1917** |
| `legitimate_email` | 2000 | 500 | 0 | 818 | 1182 | **4500** |
| `legitimate_sms` | 786 | 196 | 0 | 310 | 476 | **1768** |
| `phishing_email` | 3000 | 750 | 0 | 1202 | 1798 | **6750** |

## Множество `test`

| | `human (label=0)` | `llm (label=1)` | **Σ** |
|---|:---|:---|---:|
| `(human)` | 908 | 0 | **908** |
| `holdout_claude` | 0 | 2059 | **2059** |
| `llm:openai/chatgpt` | 0 | 82 | **82** |
| `seen_mistral` | 0 | 333 | **333** |
| `seen_openai` | 0 | 493 | **493** |

- В **test** (**3875** строк): human **908**, LLM **2967**; из LLM в test Claude holdout **2059**, прочие LLM **908**.
- Среди **всех** LLM в Core: Claude holdout **2059 / 11097 ≈ 18.6%**.

## Проверки

- **required_keys**: PASS — `{"ok": true, "bad_rows": 0}`
- **claude_only_test**: PASS — `{"ok": true, "violations": 0}`
- **financial_qa_complete_pairs**: PASS — `{"ok": true, "pair_violations": 0}`
- **balance_human_seenllm_whole_per_family**: PASS — `{"ok": true, "mismatches": []}`
- **balance_human_seenllm_per_family_main_slices**: PASS — `{"ok": true, "tolerance_rule": "0 except nh+nl<=8 -> 1", "mismatches": []}`
- **balance_human_seenllm_per_family_test_seen**: PASS — `{"ok": true, "tolerance_rule": "0 except nh+nl<=8 -> 1", "mismatches": []}`
- **no_empty_text**: PASS — `{"ok": true, "empty_rows": 0}`
- **no_duplicate_text_global**: PASS — `{"ok": true, "duplicate_rows": 0}`
- **financial_qa_question_id_pair_shape**: PASS — `{"ok": true, "bad_question_groups": 0}`
- **eval_slices_present**: PASS — `{"ok": true, "slices": {"train": 13554, "val": 2706, "test_claude_holdout": 2059, "test_seen": 1816}}`
- **audit_channel_split_label**: PASS — `{"ok": true, "warnings": []}`
- **origin_model_distribution_check**: PASS — `{"ok": true, "warnings": []}`
- **source_family_distribution_check**: PASS — `{"ok": true, "warnings": []}`

## Сопоставление с §7.2 (human vs все LLM в Core)

| scenario_family | human | all_llm | seen_llm (excl. Claude) | §7.2 human | §7.2 llm |
|----------------|------:|--------:|-------------------------|------------|----------|
| `phishing_email` | 3000 | 3750 | 3000 | 2500–3000 | 2500–3000 |
| `advance_fee_scam_email` | 1600 | 2000 | 1600 | 1200–1800 | 1200–1800 |
| `fraud_sms_deceptive` | 852 | 1065 | 852 | 1200–1800 | 1200–1800 |
| `legitimate_email` | 2000 | 2500 | 2000 | 2500–3000 | 2500–3000 |
| `legitimate_sms` | 786 | 982 | 786 | 800–1200 | 800–1200 |
| `financial_qa` | 800 | 800 | 800 | 800–1200 | 800–1200 |

## Ограничения

- Human `fraud_sms_deceptive` ≤ ~852 строк: ниже середины §7.2 (доступность данных).
- `legitimate_email`: при цели 3000/сторона на main human↔seen_llm факт может быть **ниже**, если узкое место — **seen**-LLM пул в dataset2 (см. `downsampling.report.per_family.legitimate_email`).
- `legitimate_sms`: объём ограничен доступностью human/seen_llm.
- Выводы не распространять на произвольный unseen generator beyond заявленных seen + Claude holdout.

## Артефакты

- `v2/data/interim/assembled/core_v2.jsonl`, `core_train.jsonl`, `core_val.jsonl`, `core_test.jsonl` (= test_full), `core_test_full.jsonl`, `core_test_non_claude.jsonl`, `core_test_claude_only.jsonl`, `core_test_claude_binary.jsonl`
- `v2/data/interim/assembled/core_manifest.json`, `core_split_diagnostics.json`
- `v2/docs/core_split_diagnostics.md`
- `v2/outputs/tables/core_crosstab_*.csv`
