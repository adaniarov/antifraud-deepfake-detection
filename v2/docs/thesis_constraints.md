# v2/docs/thesis_constraints.md

Ограничения на текст диссертации и научные формулировки для трека **Core (v2)**. Общие принципы совпадают с корневым `docs/thesis_constraints.md`; ниже — уточнения под новую постановку.

## Writing style
- Основной язык изложения: русский
- Код, промпты, конфиги: английский
- Тон: формальный, академический, сдержанный

## What not to do
- Не выдумывать метрики, размеры выборок и итоги экспериментов
- Не выдавать результаты legacy-сборки (корневой `data/final/`) за результаты **Core v2** без явного указания, что это разные датасеты
- Не преувеличивать обобщение на весь антифрод-домен
- Не игнорировать ограничения датасета

## Required discussion points when relevant
- смещение по эпохе источников (`time_band`)
- дисбаланс каналов и `scenario_family`
- оценка на held-out generator (Claude-family, test-only) и партициях
- различие **phishing_email** vs **advance_fee_scam_email** в интерпретации fraud-email
- единая SMS policy на двух источниках fraud SMS
- ограничения robustness и переноса на внешние жанры

## Итоговый датасет Core v2 — обязательные формулировки (ВКР)
При описании данных и результатов **нужно** явно зафиксировать:
- **Main comparable pool** в train / val / test_seen **сбалансирован** по **human vs seen_llm** (seen = OpenAI/Mistral; в `financial_qa` — HC3 ChatGPT как seen-LLM); это не то же самое, что суммарный LLM-ряд с учётом Claude.
- **Claude holdout** — **отдельный** test-only stress-test (`test_claude_only` / `core_eval_slice=test_claude_holdout`), не смешивать с интерпретацией основного сопоставимого теста без явного разделения.
- Core — **multi-genre / multi-scenario** бенчмарк (несколько `scenario_family`, каналы email / sms / qa).
- Указать **temporal bias** и **ограниченное покрытие генераторов**; не преувеличивать обобщение за пределы заявленных seen + holdout + HC3 в control slice.
- Срез **SMS fraud** (`fraud_sms_deceptive`) **ограничен объёмом данных** (ниже целевого диапазона §7.2 — см. `project_status.md`, manifest).

## Протокол отчёта метрик (эксперименты на Core)
- Главные таблицы/цифры — **раздельно** для **val**, **test_non_claude**, **test_claude_only** (файлы и поля — `v2/docs/core_dataset_description.md`, `core_manifest.json`).
- **test_full** допустим как **дополнительный** агрегированный тест; **недопустимо** представлять его как **единственную** главную метрику качества.

## Dataset and task anchor (v2)
При описании постановки задачи и данных ссылаться на:
- `v2/docs/dataset_design_final.md` как первичное описание Core
- `v2/docs/dataset_contract.md` для инвариантов
- `v2/docs/core_as_built.md` и `v2/docs/raw_sources_inventory.md` для фактической линии данных и отклонений от плана (глава про датасет / воспроизводимость)

## Thesis structure anchor
1. Introduction
2. Related work
3. Dataset and task setup (**Core**, оси, исключения)
4. Methodology
5. Experiments and results
6. Prototype / implementation
7. Conclusion
