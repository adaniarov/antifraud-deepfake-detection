# v2/docs/next_tasks.md

Очередь задач согласована с чеклистом §8 и §10 в `v2/docs/dataset_design_final.md`.

## Priority queue

### P0 — Подготовка `gathered/` (prepare-ноутбуки)
**✅ Выполнено** (артефакты на диске в `v2/data/interim/gathered/`). Ожидаемые эффекты исправлений отражены в файлах: например Mendeley **369** строк (после §5.8), SmishTank **483**, плюс `nazario_prepared.jsonl`, `nigerian_fraud_prepared.jsonl`, `financial_qa_prepared.jsonl`, `enron_ham_prepared.jsonl`, `spamassassin_ham_prepared.jsonl`, `sms_ham_prepared.jsonl`.

- **Регресс:** после правок любого prepare-ноутбука — перезапустить его и при необходимости цепочку **17 → 18 → 19**.

### P0 — LLM-side (Dataset 2) и согласование с §7.2
> **Инфраструктура (Apr-2026):** `v2/src/llm_mass_generation.py` + `11_*` / `12_*` / `13_*` → `v2/data/interim/llm-generated/`.

**Частично закрыто.**

- ✅ Сырой пул в `llm-generated/` (seen OpenAI/Mistral + holdout Claude); сборка **`dataset2_llm.jsonl`** — `18_dataset2_llm_assembly.ipynb` (поля §4.1 по контракту сборки).
- ✅ **Claude holdout** в Core не смешивается с train/val/test_seen — политика `19_core_train_val_test_assembly.ipynb` / `core_manifest.json`.
- ⏳ **Открыто при желании «догнать §7.2»:** дополнительные прогоны 11/12/13 и повтор 18 → 19. Текущий замороженный Core уже может быть **ниже верхних границ §7.2** по ряду семей из‑за `min(target_cap, n_human_raw, n_seen_llm_raw)` — факты в `core_manifest.json` → `downsampling.report`.
- **Напоминание на будущие прогоны:** env **`OPENAI_SEEN_SHARE`** (0.0–1.0) одинаково в 11 и 12; по умолчанию в ноутбуках часто `0.5`; **не менять долю** посреди уже начатой пары seen-файлов без понимания перераспределения слотов.

### P0 — Сборка Dataset 1 (human-side)
**✅ Выполнено:** `17_dataset1_human_assembly.ipynb` → `v2/data/interim/assembled/dataset1_human.jsonl` (fraud + ham + human-сторона `financial_qa` из `gathered/`). Enron и SpamAssassin ham входят через **`enron_ham_prepared.jsonl`**, **`spamassassin_ham_prepared.jsonl`** в `gathered/` (ноутбуки 14/15). Дедуп и схема §4.1 — по логике ноутбука 17; сводные числа — `core_manifest.json` / исходные counts в 17.

- **Known limitation:** SMS fraud combined 483+369 = **852** < целевой диапазон 1 200–1 800 §7.2 (data availability).

### P0 — Сборка train/val/test и manifest
- ✅ Выполнено: `19_core_train_val_test_assembly.ipynb` → `v2/data/interim/assembled/core_manifest.json`, `core_*.jsonl`, `core_eval_slice`, протокол **human vs seen_llm** для train/val/test_seen; Claude — `test_claude_only`.
- При изменении сырья или политики сборки — перезапуск 17 → 18 → 19 и обновление manifest.
- **Формат отчёта экспериментов:** обязательно отдельно **val**, **test_non_claude**, **test_claude_only**; **test_full** — только дополнительно (см. `project_status.md`).

### P1 — Baselines на Core
- Адаптировать/подключить classical pipeline (features + LR/RF/XGBoost или эквивалент) к v2 manifest.
- Подготовить transformer baseline (full fine-tune и/или LoRA) на тех же сплитах.
- Зафиксировать окружение (версии библиотек, железо) для главы 4.

### P2 — Углубление
- Кросс-валидация или явное обоснование single-split для classical; интерпретируемость (SHAP / permutation / коэффициенты на подмножестве).
- План robustness (парафразы, rewrite) и stress-test источников **вне** Core, если понадобится отдельная оценка.

## Completed tasks (архив)
- ✅ Подготовка всех 4 fraud-источников (`smishtank`, `mendeley_smishing`, `nazario`, `nigerian_fraud`) с `langdetect_v1` — `v2/data/interim/gathered/` (актуальные строки и эффекты фиксов — см. P0 «Подготовка gathered» выше)
- ✅ Legitimate ham в `gathered/`: `enron_ham_prepared.jsonl`, `spamassassin_ham_prepared.jsonl`, `sms_ham_prepared.jsonl` (ноутбуки `14_*`, `15_*`, `16_*`)
- ✅ Диагностические аудиты ham: Enron (420), SpamAssassin (320), SMS (320) — `v2/data/interim/annotated/`
- ✅ Ноутбуки 06/07/08: расширенные system prompts с определениями категорий, `MAX_NEW_THIS_RUN` = full sample, analysis cell
- ✅ Анализ длин токенов по всем источникам; channel-level пороги `length_bin` зафиксированы в `v2/config.py` — `v2/notebooks/03_dataset_creation/00_length_bin_analysis.ipynb`
- ✅ Исправление `smishtank_prepare.ipynb`: hardcoded `length_bin='short'` → вычисляемый из `config.py`
- ✅ Исправление `mendeley_smishing_prepare.ipynb`: §5.8 policy (−19 строк), `mask_url` второй pass, `length_bin` из `config.py`
- ✅ Исправление `nazario_prepare.ipynb`: outlier cap ≤5000 токенов, `length_bin` из `config.py`
- ✅ Исправление `nigerian_fraud_prepare.ipynb`: outlier cap ≤5000 токенов, `length_bin` из `config.py`
- ✅ `hc3_finance_prepare.ipynb` создан; выводит `financial_qa_prepared.jsonl` (human + chatgpt, paired via `question_id`)
- ✅ **Prompt design финализирован** — 5 prompt families в `v2/data/prompts/` (Apr-2026):
  - `phishing_email.json` — 9 subtypes, few-shot из Nazario annotated, email/short+medium+long
  - `advance_fee_scam_email.json` — 4 subtypes, few-shot из Nigerian 419 annotated, email/medium+long
  - `fraud_sms_deceptive.json` — 3 Core subtypes, few-shot из SmishTank+Mendeley annotated, sms/short+medium ([`llm_prompt_families_contract.md`](llm_prompt_families_contract.md))
  - `legitimate_email.json` — 5 subtypes в JSON, few-shot из Enron ham annotated, email/short+medium+long
  - `legitimate_sms.json` — 2 subtypes, few-shot из SMS ham annotated, sms/short
  - [`llm_prompt_families_contract.md`](llm_prompt_families_contract.md) — формат, схема, правила генерации, word-count guide (генераторный split — в дизайне §6 и `core_as_built.md`)
- ✅ **Массовая генерация LLM:** `v2/src/llm_mass_generation.py` (QC, `length_bin` из `config.py`, resume, tqdm), ноутбуки `11_mass_generation_openai.ipynb`, `12_mass_generation_mistral.ipynb`, `13_mass_generation_claude_openrouter.ipynb`, вывод в `v2/data/interim/llm-generated/`; настраиваемая доля OpenAI в seen через **`OPENAI_SEEN_SHARE`**
- ✅ **Сборка Dataset 1 / 2 и Core:** `17_dataset1_human_assembly.ipynb`, `18_dataset2_llm_assembly.ipynb`, `19_core_train_val_test_assembly.ipynb` → assembled JSONL + замороженный Core с `assembly_policy` `core_v2_seen_balanced_plus_claude_holdout`

## How Cursor should choose the next task
Предпочитать задачи, которые:
1. Двигают чеклист §8 к «Definition of done» §10.
2. Снижают риск semantic mismatch между human и LLM ветками.
3. Улучшают воспроизводимость (явные пути, версии, manifest).
4. Не меняют семантику меток и семейств без явного обновления `dataset_design_final.md`.
