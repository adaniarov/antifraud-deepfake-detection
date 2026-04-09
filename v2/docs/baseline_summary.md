# Core v2 — сводка по признакам (EDA) и классическим baseline

Документ опирается на **зафиксированные артефакты** в репозитории: таблицы и графики из `v2/notebooks/04_features/01_core_feature_extraction_and_eda.ipynb` и результаты прогона `v2/notebooks/05_classical_ml/01_core_classical_baselines.ipynb`. Числа по объёму данных — из `v2/data/interim/assembled/core_manifest.json` (`row_counts`).

---

## 1. Данные и протокол оценки

| Срез | Файл | Строк (manifest) | Назначение |
|------|------|------------------|------------|
| train | `core_train.jsonl` | 13 554 | Обучение, fit TF-IDF и скейлеров |
| val | `core_val.jsonl` | 2 706 | Подбор/санити-чек |
| test_seen | `core_test_non_claude.jsonl` | 1 816 | Основной тест **human vs seen_llm** |
| test_claude_holdout | `core_test_claude_only.jsonl` | 2 059 | Holdout по генератору (Claude-family) |
| test_full (агрегат) | union test_seen ∪ holdout | 3 875 | Дополнительно, не вместо отчётных срезов |

**Важно (проверено по `core_dense_features.parquet`):** в `test_claude_only` все строки имеют **`label = 1` (только LLM)**. Поэтому для этого среза **ROC-AUC и average precision не определены** (нет отрицательного класса). В таблице baseline для этих строк добавлен показатель **`llm_predicted_rate`** — доля примеров, классифицированных как LLM при пороге 0.5; при всех истинных метках «LLM» он совпадает с recall для класса 1.

Источник политики сплитов: `v2/docs/project_status.md`, `v2/docs/thesis_constraints.md`.

---

## 2. Выводы из EDA признаков (`04_features`)

Ниже — **описательные факты** по файлам в `v2/outputs/tables/features/` и `v2/outputs/figures/features/`. Интерпретация отделена там, где это гипотеза.

### 2.0 Legacy HC (v1-style, 68 признаков)

В ноутбуке `04_features` добавлен блок, совместимый по смыслу с **HC** из v1 [`notebooks/02_features/03_feature_engineering.ipynb`](../../notebooks/02_features/03_feature_engineering.ipynb) (**68** признаков в v1; в репозитории v1 указано именно 68). Реализация: **`v2/src/core_legacy_hc_features.py`**, токенизация/POS/леммы через **NLTK** (без spaCy). В таблице признаки с префиксом `hc_`.

Дополнительные артефакты после прогона EDA:

- [`hc_summary_mean_std_by_label_train.csv`](../outputs/tables/features/hc_summary_mean_std_by_label_train.csv)
- [`hc_pointbiserial_train_label.csv`](../outputs/tables/features/hc_pointbiserial_train_label.csv)
- [`hc_pointbiserial_top20_train.png`](../outputs/figures/features/hc_pointbiserial_top20_train.png), [`hc_kde_top8_by_label_train.png`](../outputs/figures/features/hc_kde_top8_by_label_train.png), [`hc_correlation_top24_train.png`](../outputs/figures/features/hc_correlation_top24_train.png)

Тесты Mann–Whitney / MI / диагностическая LR в ноутбуке **автоматически включают** dense + `hc_*` (+ LM при наличии), поэтому состав колонок в `mannwhitney_train_human_vs_llm.csv` и т.п. расширяется после пересчёта.

### 2.1 Различия human vs LLM на train (Mann–Whitney, Bonferroni)

Файл: [`v2/outputs/tables/features/mannwhitney_train_human_vs_llm.csv`](../outputs/tables/features/mannwhitney_train_human_vs_llm.csv).

- По всем перечисленным признакам p-value после Bonferroni остаётся < 0.05: распределения human (0) и LLM (1) **статистически различаются**.
- Наибольшие эффекты по Cohen’s d (human − LLM): **`punct_ratio` (~0.64)**, **`char_entropy` (~0.55)**, **`digit_ratio` (~0.53)**, **`smog_index` (~0.52)**; **`flesch_reading_ease` отрицательный (~−0.41)** (LLM-сторона сдвинута в сторону другой читаемости).
- **`lm_mean_nll`** (DistilGPT-2): крайне малое p, d ~ **0.46** — LM-scoring различает классы, но **scoring-модель ≠ генераторы Core** (оговорка из ноутбука 04).

### 2.2 Mutual information с меткой (train)

Файл: [`mutual_info_train_label.csv`](../outputs/tables/features/mutual_info_train_label.csv).

- Наибольшая MI: **`flesch_kincaid_grade`**, **`smog_index`**, **`digit_ratio`**, **`space_ratio`**, **`char_entropy`**, **`punct_ratio`** — согласуется с тем, что стиль/читаемость и поверхностная лексика информативны для разделения.

### 2.3 Диагностическая логрегрессия (только val, из 04)

Файл: [`diagnostic_logreg_val_metrics.csv`](../outputs/tables/features/diagnostic_logreg_val_metrics.csv).

- ROC-AUC **0.982**, AP **0.980** на val — **только sanity-check** того же типа признаков (dense+LM в том прогоне EDA); не тот же пайплайн, что полный baseline в §3.

### 2.4 SHAP (линейная модель на dense+LM)

Файл: [`shap_linear_dense_top_features.csv`](../outputs/tables/features/shap_linear_dense_top_features.csv).

- Крупнейшие средние |SHAP|: **`tiktoken_len`**, **`word_len_ws`**, индексы читаемости, **`char_len`**, **`punct_ratio`**, затем **`lm_mean_nll`**.

### 2.5 PCA (train, dense)

Файл: [`pca_explained_variance_train.csv`](../outputs/tables/features/pca_explained_variance_train.csv); рисунки [`pca_train_by_label.png`](../outputs/figures/features/pca_train_by_label.png), [`pca_train_by_scenario.png`](../outputs/figures/features/pca_train_by_scenario.png).

- PC1 + PC2 дают **~53%** дисперсии (0.354 + 0.178) — часть вариации укладывается в низкоразмерное подпространство; визуально полезно смотреть пересечения по `scenario_family`.

### 2.6 Стабильность признаков: test_seen vs Claude holdout (label=1)

Файл: [`stability_llm_slices_and_lanes.csv`](../outputs/tables/features/stability_llm_slices_and_lanes.csv).

- **`char_len`**: медианы близки (541.5 vs 652.0), Mann–Whitney **не значим** (p ≈ 0.46) на доступных n.
- **`tiktoken_len`, `ttr`, `char_entropy`, `lm_mean_nll`**: **сильные сдвиги** между seen и Claude (малые p) — **генераторный сдвиг** отражается в признаках; это согласуется с ожидаемым **domain/generator shift** на holdout.
- По `generator_lane` (только LLM): medians показывают, что **holdout_claude** заметно выше по **`lm_mean_nll`** и **`ttr`**, чем seen_openai / seen_mistral — **эмпирический факт по этому кэшу LM**.

### 2.7 Поверхностные медианы длины (train)

- [`median_char_len_by_channel_train.csv`](../outputs/tables/features/median_char_len_by_channel_train.csv): каналы email / sms / qa сильно различаются по длине — **разрезы по `channel` обязательны** при интерпретации.
- [`median_char_len_by_scenario_family_train.csv`](../outputs/tables/features/median_char_len_by_scenario_family_train.csv): **`legitimate_email`** и **`advance_fee_scam_email`** — очень длинные human-сообщения относительно SMS-семейств; это влияет на переносимость пороговых признаков.

### 2.8 Общий вывод по EDA (синтез)

**Факт:** классические текстовые признаки и LM-NLL **стабильно различают** human и seen_llm на train; **часть признаков заметно сдвигается** между seen_llm и Claude-holdout при том же `label=1`.

**Гипотеза (осторожно):** detector, оптимизированный только под seen_llm, может **остаться полезным** на Claude по длине/стилю, но **сдвиг по `lm_mean_nll` и `ttr`** указывает на риск **занижения уверенности** или смены ошибок — это нужно смотреть по baseline на полных срезах (§3).

---

## 3. Классические baseline (`05_classical_ml`)

Ноутбук: [`v2/notebooks/05_classical_ml/01_core_classical_baselines.ipynb`](../notebooks/05_classical_ml/01_core_classical_baselines.ipynb).

**Признаки:** числовой блок = **14 dense** + **68 `hc_*`** (legacy v1-style, NLTK) + опционально **`lm_mean_nll`**, всё в `core_dense_features.parquet`; плюс TF-IDF word/char из pickle’ов (`04_features`, fit только на train).

**Модели:**

1. **`lr_dense_tfidf`** — `LogisticRegression(saga)`, dense + word + char TF-IDF, `class_weight=balanced`.
2. **`lr_dense_only`** — `LogisticRegression(lbfgs)` только на dense (+LM).
3. **`rf_dense`** — `RandomForestClassifier` на dense (+LM).
4. **`xgb_dense`** — `XGBClassifier` на dense (+LM).

**Метрики:** см. [`v2/outputs/tables/classical_ml/baseline_metrics_by_split.csv`](../outputs/tables/classical_ml/baseline_metrics_by_split.csv). Таблица §3.1 нуждается в **обновлении после перепрогона** ноутбука, если менялся набор признаков.

### 3.1 Численные результаты (основные срезы)

| Модель | val ROC-AUC | val AP | test_seen ROC-AUC | test_seen AP | test_claude_holdout `llm_predicted_rate` |
|--------|-------------|--------|-------------------|--------------|------------------------------------------|
| lr_dense_tfidf | 0.988 | 0.987 | 0.988 | 0.989 | **0.761** |
| lr_dense_only | 0.892 | 0.849 | 0.887 | 0.869 | **0.356** |
| rf_dense | 0.986 | 0.986 | 0.985 | 0.986 | **0.383** |
| xgb_dense | 0.993 | 0.993 | 0.993 | 0.993 | **0.478** |

Для **test_claude_holdout** ROC-AUC и AP в CSV **пустые (NaN)** из-за одного класса в `y` (см. §1). **`llm_predicted_rate`** интерпретировать как «долю текстов Claude-holdout, отнесённых к LLM» при пороге 0.5.

**Агрегат test_full** (две тестовые части вместе, без отдельной интерпретации как главной метрики):

| Модель | ROC-AUC | AP |
|--------|---------|-----|
| lr_dense_tfidf | 0.963 | 0.986 |
| lr_dense_only | 0.733 | 0.884 |
| rf_dense | 0.847 | 0.949 |
| xgb_dense | 0.908 | 0.970 |

### 3.2 Разрез по `scenario_family` (test_seen, `lr_dense_tfidf`)

Файл: [`baseline_by_scenario_family_test_seen.csv`](../outputs/tables/classical_ml/baseline_by_scenario_family_test_seen.csv).

- **Email fraud** (`phishing_email`, `advance_fee_scam_email`): ROC-AUC **≥ 0.998**, n 320–600.
- **SMS** (`legitimate_sms`, `fraud_sms_deceptive`): ROC-AUC **~0.89–0.93**, n 160–172 — **ниже и на меньших n**, что согласуется с меньшим объёмом и большей вариативностью коротких текстов.
- **`financial_qa`**: ROC-AUC **~0.90**, n 164.

### 3.3 Визуализации

В `v2/outputs/figures/classical_ml/` для каждой модели сохранены:

- `*_roc.png` — ROC для val / test_seen; для test_claude_holdout подпись **«ROC N/A (один класс в y)»**.
- `*_pr.png` — аналогично для PR.
- `*_confusion.png` — матрицы ошибок по трём срезам.

---

## 4. Итоговые выводы

1. **Отчётные бинарные метрики (ROC-AUC, AP)** на Core v2 в этом прогоне **высокие** на **val** и **test_seen** для моделей с TF-IDF и для **XGB/RF на dense** — при условии соблюдения протокола (отдельно val / test_seen / Claude-holdout).
2. **TF-IDF + dense** даёт существенный прирост относительно **только dense** на test_seen (ROC-AUC **~0.988 vs ~0.887** для логрегрессий) — **факт по таблице baseline**.
3. **Срез test_claude_only** в текущей сборке **не содержит human** (`label=0` отсутствует): стандартный ROC/PR **не определены**; по **`llm_predicted_rate`** видно, что **часть Claude-текстов классифицируется как human** при пороге 0.5 (например **~24%** для `lr_dense_tfidf`, **~52–62%** для моделей без TF-IDF) — **эмпирический факт**, интерпретировать как **сдвиг распределения / failure modes**, а не как «AUC на holdout».
4. EDA и breakdown по **`scenario_family`** указывают на **неоднородность**: email-сценарии проще отделяются, чем SMS и QA — при планировании отчёта и дальнейших экспериментов стоит **явно стратифицировать** метрики.

---

## 5. Файлы для воспроизведения

| Артефакт | Путь |
|----------|------|
| EDA ноутбук | `v2/notebooks/04_features/01_core_feature_extraction_and_eda.ipynb` |
| Baseline ноутбук | `v2/notebooks/05_classical_ml/01_core_classical_baselines.ipynb` |
| Генератор baseline ipynb | `v2/notebooks/05_classical_ml/_gen_baseline_nb.py` |
| Таблицы baseline | `v2/outputs/tables/classical_ml/` |
| Графики baseline | `v2/outputs/figures/classical_ml/` |
| Таблицы EDA | `v2/outputs/tables/features/` |
| Графики EDA | `v2/outputs/figures/features/` |

Дата актуальности сводки: по времени последнего успешного прогона `01_core_classical_baselines.ipynb` и содержимому перечисленных CSV.
