# v2/docs/dataset_contract.md

Компактный контракт **Core-датасета** v2. При противоречии с деталями приоритет у `v2/docs/dataset_design_final.md`.

## Task
Бинарная классификация:
- human-written → `label 0`, `label_str human`
- LLM-generated → `label 1`, `label_str llm`

## Factorial axes (Core)
Каждая запись несёт как минимум:
- `fraudness` ∈ `{fraud, legitimate}`
- `channel` ∈ `{email, sms, qa}`
- `scenario_family` — одно из финальных семейств Core (см. ниже)

Трактовка: модель учится на `human vs llm`; оси нужны для стратификации, диагностики и снижения путаницы домена/жанра.

## Scenario families in Core

### Fraud
- `phishing_email` — human anchor: Nazario
- `advance_fee_scam_email` — human anchor: Nigerian Fraud (**не** смешивать с phishing)
- `fraud_sms_deceptive` — human anchors: SmishTank + Mendeley smishing subset по **единой** policy подтипов

### Legitimate
- `legitimate_email` — Enron ham + SpamAssassin ham
- `legitimate_sms` — SMS ham
- `financial_qa` — HC3 Finance human; LLM сторона — готовый HC3 Finance chatgpt (**без** новой prompt family)

## Explicitly out of Core
Enron/SpamAssassin/SMS **spam** как классы Core, T3 social engineering, T5 bank notification, T6 financial review, HC3 Open QA, broad generic spam без чистого fraud mapping — по §3.4 `dataset_design_final.md`. Возможны отдельные вспомогательные эксперименты, но не как состав Core без изменения спецификации.

## Human sources (Core)
- Nazario Phishing
- Nigerian Fraud (419-style corpus)
- SmishTank
- Mendeley SMS Phishing dataset — только согласованный smishing subset
- Enron ham, SpamAssassin ham
- SMS ham
- HC3 Finance (human answers)

## LLM sources (generator policy)
**Train / validation (seen generators):**
- один OpenAI-family generator
- один open-weight / non-OpenAI instruction-tuned generator (например, Mistral-family)

**Test only (held-out):**
- Claude-family

**Жёсткие инварианты:**
- Ни одна Claude-generated запись не попадает в train или val.
- `temperature` задаётся как **API parameter**, не как текст внутри prompt.
- URL masking и прочий preprocessing — **симметрично** для human и LLM.
- Не переопределять `label` / `scenario_family` молча; изменения только через обновление спецификации.

## Split sizes
Конкретные числа train / val / test для Core фиксируются **после сборки** и выносятся в manifest / `dataset_stats` (ссылка будет в `v2/docs/project_status.md`).

Плановые **целевые** объёмы по срезам — таблица §7.2 `dataset_design_final.md` (ориентиры, не финальный отчёт).

## Schema highlights
Обязательные поля перечислены в §4.1 `dataset_design_final.md` (`text`, метки, `fraudness`, `channel`, `scenario_family`, `source_family`, `dataset_source`, `time_band`, `length_bin`, `origin_model`, `split`, и др.).

## Derived evaluation slice `test_claude_binary`

Отдельный JSONL (не входит в union `core_v2` row count): бинарная оценка **human (резерв dataset1, disjoint от train/val/test_seen) vs Claude holdout**. Использовать как основной **robustness**-тест при полноценных ROC/PR; `test_claude_only` остаётся **one-class** диагностикой. Сборка и проверки: `v2/src/core_test_claude_binary.py`, `core_manifest.json` → `test_claude_binary`.

## Known limitations
- temporal bias (`time_band`)
- дисбаланс каналов и семейств при приоритете чистоты
- ограниченное число генераторов; интерпретация без партиции Claude неконтролируема
- для части семей human-резерв может быть исчерпан main pool → неполное покрытие `test_claude_binary` (фиксируется в manifest/diagnostics)

## Use rule
Если код или черновик расходятся с этим контрактом или с `dataset_design_final.md`, побеждает спецификация в `dataset_design_final.md`, пока пользователь явно не изменит дизайн.
