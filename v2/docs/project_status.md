# v2/docs/project_status.md

## Current stage
**Core v2 — замороженная сборка (Apr-2026):** итоговые сплиты и manifest собраны ноутбуком `v2/notebooks/03_dataset_creation/19_core_train_val_test_assembly.ipynb` из `dataset1_human.jsonl` + `dataset2_llm.jsonl`. **Источник чисел по строкам:** `v2/data/interim/assembled/core_manifest.json` (поле `row_counts`, `created_utc`). Краткое описание датасета: `v2/docs/core_dataset_description.md`; диагностика сплита: `v2/docs/core_split_diagnostics.md`, `core_split_diagnostics.json`, `v2/outputs/tables/core_crosstab_*.csv`.

**Политика сборки (зафиксирована в manifest):** `assembly_policy` = `core_v2_seen_balanced_plus_claude_holdout`; **main comparable pool** (train / val / `test_seen`) балансируется как **human ↔ seen_llm** (OpenAI/Mistral; в `financial_qa` — HC3 ChatGPT как seen); **Claude holdout** — отдельный add-on только в `test_claude_only` (`core_eval_slice=test_claude_holdout`), не участвует в бюджете main.

**LLM-side:** инфраструктура массовой генерации (`v2/src/llm_mass_generation.py`, ноутбуки 11/12/13) и **5 prompt families** в `v2/data/prompts/` (Apr-2026).

**Модели / эксперименты:** отчётные метрики обучения на Core пока не зафиксированы; **следующий шаг** — baselines по протоколу метрик ниже.

### Обязательный протокол метрик в экспериментах (Core)
Чтобы интерпретация оставалась корректной, **главные** метрики (и таблицы в ВКР) показывать **раздельно** для:
- **val** (`core_val.jsonl`);
- **test_non_claude** (`core_test_non_claude.jsonl`, `core_eval_slice=test_seen`) — основной сопоставимый тест **human vs seen_llm**;
- **`test_claude_binary`** (`core_test_claude_binary.jsonl`) — **бинарный** robustness-тест **human reserve vs Claude holdout** (полноценные ROC/PR/F1 при наличии обоих классов);
- **test_claude_only** (`core_test_claude_only.jsonl`) — **one-class** stress-сигнал по holdout-генератору (без отрицательного класса; не заменяет `test_claude_binary`).

**test_full** (`core_test_full.jsonl` / `core_test.jsonl` = union test_seen ∪ test_claude_only) — только **дополнительный агрегированный** срез; **нельзя** делать его единственной главной цифрой качества.

### Что явно зафиксировать в тексте ВКР (итоговый датасет)
- Main comparable pool по семьям сценариев **сбалансирован** по **human vs seen_llm** в train / val / test_seen; суммарно train+val+test_seen близки к 50/50 human–seen_llm.
- **Claude** — отдельный **holdout stress-test**, не смешивать с интерпретацией «основного» теста без явного разделения.
- Core остаётся **multi-genre / multi-scenario** бенчмарком (несколько `scenario_family`, каналы email/sms/qa).
- Есть **temporal bias** (`time_band`) и **ограниченное покрытие генераторов** (seen OpenAI/Mistral + holdout Claude + HC3 в `financial_qa`); обобщение за пределы заявленных генераторов не преувеличивать.
- Срез **SMS fraud** (`fraud_sms_deceptive`) **ограничен объёмом данных** (SmishTank + Mendeley; суммарно ниже целевого диапазона §7.2 — см. таблицы выше и `core_manifest.json` → `downsampling.report`).

## Completed (v2 track)
- **Сборка Core v2 (train/val/test + eval slices):** ноутбуки `17_dataset1_human_assembly.ipynb`, `18_dataset2_llm_assembly.ipynb`, `19_core_train_val_test_assembly.ipynb` → `v2/data/interim/assembled/core_v2.jsonl`, `core_train.jsonl`, `core_val.jsonl`, `core_test*.jsonl`, `core_manifest.json`; баланс main pool **human vs seen_llm**; Claude только в `test_claude_only`; описание и диагностика — `core_dataset_description.md`, `core_split_diagnostics.md` / `.json`, crosstab CSV.
- **Производный срез `test_claude_binary`:** скрипт/ноутбук `v2/src/core_test_claude_binary.py`, `20_core_test_claude_binary.ipynb` → `core_test_claude_binary.jsonl`, обновление `core_manifest.json`, crosstab `core_crosstab_test_claude_binary__*.csv`, авто-блок в `core_split_diagnostics.md`. Main pool не изменяется.
- Переопределение границ Core: отказ от монолитных broad-spam классов как fraud; явная онтология `fraudness` × `channel` × `scenario_family`.
- Зафиксирован состав human fraud: Nazario (`phishing_email`), Nigerian Fraud (`advance_fee_scam_email`), SmishTank + фильтрованный Mendeley smishing (`fraud_sms_deceptive`).
- Зафиксирован состав human legitimate: Enron ham + SpamAssassin ham (`legitimate_email`), SMS ham (`legitimate_sms`), HC3 Finance human (`financial_qa` control slice).
- Политика генераторов: seen (train/val) — OpenAI-family + open-weight/non-OpenAI instruction-tuned; **holdout — Claude-family, только test**.
- Документированы обязательные поля записи, preprocessing rules, target counts (плановые диапазоны), checklist до baseline.

### Human-side preprocessing (Dataset 1) — завершено
Все источники fraud и legitimate прошли очистку, URL masking, language filter, дедупликацию и сохранены в `v2/data/interim/gathered/`.

`length_bin` вычисляется через `v2/config.py` (channel-level пороги: sms 20/60, email 100/400, qa 75/250).

| Источник | Файл | Строк | Примечание |
| --- | --- | --- | --- |
| SmishTank | `smishtank_prepared.jsonl` | 483 | `langdetect_v1`; после исправления `length_bin` Apr-2026 |
| Mendeley smishing | `mendeley_smishing_prepared.jsonl` | 369 | `langdetect_v1`; −19 строк после §5.8 policy fix Apr-2026 |
| Nazario phishing | `nazario_prepared.jsonl` | ~5 213 | `english_like_ascii_v1`; outlier cap ≤5000 tok Apr-2026 |
| Nigerian 419 fraud | `nigerian_fraud_prepared.jsonl` | ~3 230 | `english_like_ascii_v1`; outlier cap ≤5000 tok Apr-2026 |
| HC3 Finance (human + chatgpt) | `financial_qa_prepared.jsonl` | ~7 800 | `langdetect_v1`; human (label=0) + chatgpt (label=1); paired via `question_id` |

Точные числа строк по `wc -l` на `gathered/*.jsonl` и разбивка по источникам: [`raw_sources_inventory.md`](raw_sources_inventory.md) §1.

**Known limitation:** SMS fraud combined 483+369 = 852 строки < target 1 200–1 800 (проблема доступности данных, не ошибка pipeline).

Ноутбуки используют:
- SMS: `langdetect_v1`
- Email: `english_like_ascii_v1`
- QA: `langdetect_v1`

### Диагностические аудиты ham — завершено
Все три аудита выполнены и сохранены в `v2/data/interim/annotated/` (модель: `openai/gpt-4o-mini` через OpenRouter, стратифицированная выборка по `subset/archive × wc_bin`):

| Источник | Файл | Строк | Страта |
| --- | --- | --- | --- |
| Enron ham | `enron_ham_annotated.jsonl` | 420 | `subset × wc_bin` |
| SpamAssassin ham | `spamassassin_ham_annotated.jsonl` | 320 | `archive × wc_bin` |
| SMS ham | `sms_ham_annotated.jsonl` | 320 | `wc_bin` |

Ноутбуки (06, 07, 08) обновлены: расширенная документация категорий, детализированные system prompts с определениями, analysis cell для диагностического summary, `MAX_NEW_THIS_RUN` = full sample (420/320/320).

### Аудит и исправления gathered/ — апрель 2026

В ходе ревизии выявлены и исправлены следующие проблемы:

| # | Ноутбук | Проблема | Исправление |
|---|---------|----------|-------------|
| 1 | `smishtank_prepare.ipynb` | `length_bin = 'short'` hardcoded для всех строк; 180/483 (37%) имели 40–197 токенов → должны быть `medium` | Заменено на `compute_length_bin(t, 'sms')` из `config.py` |
| 2 | `mendeley_smishing_prepare.ipynb` | `financial_or_crypto_lure` и `loan_or_credit_lure` включены в `CORE_SUBTYPES` — нарушение единой policy §5.7/§5.8 с smishtank | Удалены из `CORE_SUBTYPES`; −19 строк (388→369) |
| 2 | `mendeley_smishing_prepare.ipynb` | `mask_url` не имела второго прохода; assertion "unmasked URLs" падала | Добавлен второй re.sub pass |
| 3 | `nazario_prepare.ipynb` | Запись с ~308k токенами (parsing artifact) попадала в output | Добавлен outlier cap ≤5000 токенов |
| 4 | `nigerian_fraud_prepare.ipynb` | Локальная `length_bin_email()` вместо `config.py`; outlier до ~26k токенов (p90=758) | Добавлен outlier cap ≤5000 токенов; `length_bin` из `config.py` |

Создан **`v2/config.py`** — единый источник истины для `length_bin` порогов (channel-level). Все preparation notebooks обновлены для импорта из него. Анализ порогов — `v2/notebooks/03_dataset_creation/00_length_bin_analysis.ipynb`.

### Prompt design — завершено (Apr-2026)

5 prompt families созданы в `v2/data/prompts/`. Каждый файл содержит `system_prompt`, `user_template` с `{subtype}` и `{length_bin}`, `few_shot_anchors` из реальных аннотированных данных, `subtype_weights`, `length_bin_word_guide`.

| Family | Channel | Fraudness | Length bins | Subtypes |
|--------|---------|-----------|-------------|---------|
| `phishing_email` | email | fraud | short/medium/long | 9 (Nazario-based) |
| `advance_fee_scam_email` | email | fraud | medium/long | 4 (Nigerian 419-based) |
| `fraud_sms_deceptive` | sms | fraud | short+medium ([`llm_prompt_families_contract.md`](llm_prompt_families_contract.md)) | 3 Core (SmishTank+Mendeley) |
| `legitimate_email` | email | legitimate | short/medium/long | 5 подтипов в JSON (см. `legitimate_email.json`) |
| `legitimate_sms` | sms | legitimate | short | 2 (SMS ham-based) |

`financial_qa` не требует отдельного prompt family — HC3 Finance chatgpt side используется напрямую (§6.1).

### Массовая генерация LLM (реализовано)
- **Seen (train/val пул):** OpenAI и Mistral делят **одну** логическую сетку задач `(family, subtype, target_bin, idx)` без пересечения: доля слотов под OpenAI задаётся **`openai_seen_share`** (в ноутбуках 11/12 по умолчанию **0.6** — 60/40 GPT/Mistral; переопределение через **`OPENAI_SEEN_SHARE`**). Остаток сетки уходит под Mistral (квантиль на 10 000 корзин по `crc32`).
- **Ускорение:** в `v2/src/llm_mass_generation.py` поддерживается **`max_workers`** (в ноутбуках из env **`LLM_GEN_MAX_WORKERS`**, по умолчанию 8): параллельные HTTP-вызовы, запись JSONL и near-dedup под lock.
- **Holdout (test):** Claude проходит **полную** ту же сетку независимо (отдельные `generation_job_id`), `split="test"` — см. §6.9.
- **Размер сетки (логических ячеек)** при фиксированном `SAMPLES_PER_SUBTYPE = K`:
  \[
  N = K \times \sum_{\text{family}} |\texttt{subtypes}(\text{family})|
  \]
  (сумма по пяти JSON в `v2/data/prompts/`). Ожидаемое число **успешных** строк: QC/дедуп могут уменьшить фактический счёт.
  - **OpenAI (seen):** ≈ `openai_seen_share × N` слотов.
  - **Mistral (seen):** ≈ `(1 - openai_seen_share) × N` слотов.
  - **Claude:** `N` слотов (полная сетка).
  - **Сумма по трём файлам (верхняя оценка при полном успехе):** ≈ `N + N = 2N` (не три равные трети).

## Not done yet (v2)
- Полный прогон новых ноутбуков HC60 (`01_core_handcrafted_features_v2`, `02_core_feature_analysis_and_shap`, `01_core_hc60_baselines`) и фиксация отчётных метрик в тексте ВКР после `uv run`/Jupyter.
- **Перезапуск** `smishtank_prepare.ipynb`, `mendeley_smishing_prepare.ipynb`, `nazario_prepare.ipynb`, `nigerian_fraud_prepare.ipynb` — если нужно обновить gathered-файлы после правок ноутбуков (текущий Core собран из уже существующих interim-артефактов).
- При **пересборке** LLM или human после смены **`OPENAI_SEEN_SHARE`** или сырья — заново прогнать 17/18/19 и зафиксировать новый `core_manifest.json`.
- Classical / transformer **baselines** на замороженном Core с протоколом метрик (**val**, **test_non_claude**, **`test_claude_binary`**, **test_claude_only**; **test_full** — дополнительно).
- Расширенная robustness-визуализация / error analysis (частично в `02_core_feature_analysis_and_shap` / `01_core_hc60_baselines`; при необходимости — отдельный прогон).

## Thesis plan alignment (ориентир)

### Методы (гл. 3)
| Блок | Статус (v2) |
| --- | --- |
| Описание Core, осей и ограничений | Спецификация готова (`dataset_design_final.md`) |
| Признаки: stylometric / linguistic / TF-IDF | Планируется перенос/адаптация после сборки данных |
| Perplexity-фичи | По необходимости; не смешивать с «опубликованными» v1 числами |
| Классические ML | После заморозки Core |
| Трансформеры (fine-tune / LoRA) | После заморозки Core |

### Эксперименты (гл. 4)
| Блок | Статус (v2) |
| --- | --- |
| Среда, железо, метрики | Зафиксировать при первом полном прогоне на Core |
| Протокол сплитов и held-out generator | Реализовано: `core_manifest.json`, поле `core_eval_slice`; метрики — val, test_non_claude, **test_claude_binary**, test_claude_only отдельно; test_full — только доп. агрегат |
| CV / интерпретируемость | Решить протокол после появления baseline на Core |
| Robustness | После основных baseline |

## Main risks (Core-specific)
- Качество фильтрации SmishTank / Mendeley и единая policy подтипов SMS fraud.
- Временной сдвиг (`time_band`) между источниками human-текстов.
- Дисбаланс по `channel` и `scenario_family` при жёстком требовании чистоты срезов.
- Ограниченное число генераторов; **неверная** интерпретация, если подменить **test_full** основным тестом вместо **test_non_claude** + **test_claude_only**.

## Source of truth for v2 numbers
- **Замороженный Core:** `v2/data/interim/assembled/core_manifest.json` (и производные `core_*.jsonl`).
- Legacy: **не** использовать корневой `outputs/tables/classical_results_no_ppl.csv` как отчёт о Core v2.

## Maintenance rule
Обновлять этот файл после: заморозки human-side, завершения генерации, сборки сплитов, каждого валидированного эксперимента на Core. Значимые изменения политики и инструментов дублировать датой в [`CHANGELOG.md`](CHANGELOG.md); детали реализации — в [`core_as_built.md`](core_as_built.md).
