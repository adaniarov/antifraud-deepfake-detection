# Аналитический отчёт: инженерия признаков и диагностические модели (Core v2)

**Статус:** черновик выводов по зафиксированным артефактам. Ниже **встроены** ключевые таблицы (значения из CSV на момент генерации) и **графики** (относительные пути от этого файла: `v2/data/docs/` → `../../outputs/...`). При перепрогоне ноутбуков сверьте CSV.

**Источники пайплайна (два трека признаков):**  
**Legacy full-dim:** `v2/notebooks/04_features/01_core_feature_extraction_and_eda.ipynb`, `v2/notebooks/05_classical_ml/01_core_classical_baselines.ipynb`, `v2/src/core_legacy_hc_features.py`.  
**HC60 (только `hc60_*`, без TF-IDF / без legacy `hc_*`):** `v2/notebooks/04_features/01_core_handcrafted_features_v2.ipynb`, `v2/notebooks/04_features/02_core_feature_analysis_and_shap.ipynb`, `v2/notebooks/05_classical_ml/01_core_hc60_baselines.ipynb`, `v2/src/core_hc60_features.py`.

---

## 1. Цель и рамки

Задача анализа — описать **отличимость** human vs LLM на замороженном Core v2 с помощью классических и стилометрических признаков, оценить **устойчивость** к сдвигу генератора (seen vs Claude holdout) и дать **диагностические** метрики линейных/деревянных baseline без претензии на финальный production-контур.

**Чему этот отчёт не является:** итоговой калиброванной оценкой «качества антифрод-продукта»; обобщением за пределы заявленных источников и генераторов Core.

---

## 2. Данные и протокол оценки

Числа объёма — из `v2/data/interim/assembled/core_manifest.json` (`row_counts`):


| Срез                | Строк  | Файл / смысл                                                       |
| ------------------- | ------ | ------------------------------------------------------------------ |
| train               | 13 554 | `core_train.jsonl`                                                 |
| val                 | 2 706  | `core_val.jsonl`                                                   |
| test_seen           | 1 816  | `core_test_non_claude.jsonl` — основной тест human vs **seen_llm** |
| test_claude_holdout | 2 059  | `core_test_claude_only.jsonl` — holdout по **Claude-family** (только LLM, `label = 1`) |
| test_claude_binary  | 3 692  | `core_test_claude_binary.jsonl` — **бинарный** срез human vs Claude-holdout для полноценных ROC/PR/F1 |
| Всего               | 20 135 | `core_v2`                                                          |


Политика сплитов: `assembly_policy` = `core_v2_seen_balanced_plus_claude_holdout` (см. `v2/docs/project_status.md`).

**Ограничение для `test_claude_only`:** в этом срезе **только `label = 1` (LLM)**. Поэтому **ROC-AUC и average precision на нём не определены**. В таблице legacy-baseline используется вспомогательный показатель **`llm_predicted_rate`** — доля примеров, отнесённых к классу LLM при пороге 0.5; при всех истинных метках «LLM» он совпадает с recall для положительного класса. Это **не** замена AUC.

**Срез `test_claude_binary`:** отдельная выборка с **двумя классами** (human reserve + Claude holdout); на нём отчитываются robustness-метрики **HC60-baseline** (§6.2) и сравнение распределений признаков (§4.10). Это **не** отменяет осторожности при переносе с seen-генераторов на Claude: сдвиг распределений остаётся (см. §4.6 и §4.10).

---

## 3. Состав признаков


| Блок                                                                                        | Размерность / описание                                                 | Где сохранено                                                        |
| ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Dense (поверхность + textstat + энтропия символов + TTR)                                    | 14 колонок                                                             | `core_dense_features.parquet`                                        |
| Legacy HC (v1-style, NLTK: POS, функциональные слова, пунктуация, MTLD, Yule, hapax, леммы) | **68** колонок с префиксом `hc_*`                                      | тот же parquet, логика в `v2/src/core_legacy_hc_features.py`         |
| LM-scoring                                                                                  | `lm_mean_nll` (+ производная perplexity в EDA), модель **DistilGPT-2** | кэш `core_lm_distilgpt2_scores.parquet`                              |
| TF-IDF                                                                                      | word до 8000 + char до 4000 признаков, **обучение только на train**    | `core_tfidf_word.npz`, `core_tfidf_char.npz`, соответствующие `.pkl` |
| **HC60**                                                                                    | **60** числовых признаков с префиксом `hc60_*` (поверхность, пунктуация, лексика/TTR, readability, эвристики «частей речи» без NLTK legacy) | `core_hc60_v2.parquet`, manifest `v2/data/interim/features/core_hc60_v2_manifest.json` |


**Итого размерность для `lr_dense_tfidf`:** 14 + 68 + 1 (LM, если есть) + 12 000 ≈ **12 083** числовых входов (точное число TF-IDF признаков — по факту fit на train, см. вывод ноутбука 04).

По manifest HC60: **23 827** строк матрицы и **60** признаков на момент фиксации (`created_utc` в `core_hc60_v2_manifest.json`); колонки `split` / `core_eval_slice` / `label` согласованы с протоколом Core для оценки.

---

## 4. Результаты EDA (таблицы из CSV)

### 4.1 Различия human vs LLM на train (Mann–Whitney + Bonferroni)

Источник: `v2/outputs/tables/features/mannwhitney_train_human_vs_llm.csv`. Число признаков в таблице: **83**. Для части признаков Cohen’s d не определён (пустое поле в CSV) — в таблице показано как «—».


| feature                     | p_mannwhitney | Cohen d (human−LLM) | n₀   | n₁   | p_bonferroni |
| --------------------------- | ------------- | ------------------- | ---- | ---- | ------------ |
| `hc_pos_adv_ratio`          | 0             | -0.521299           | 6777 | 6777 | 0            |
| `hc_punct_dash_ratio`       | 0             | 0.391968            | 6777 | 6777 | 0            |
| `hc_avg_sentence_len_words` | 0             | 0.376379            | 6777 | 6777 | 0            |
| `hc_avg_sentence_len_chars` | 0             | 0.365537            | 6777 | 6777 | 0            |
| `hc_punct_colon_ratio`      | 0             | 0.551609            | 6777 | 6777 | 0            |
| `coleman_liau_index`        | 0             | 0.391734            | 6773 | 6777 | 0            |
| `smog_index`                | 0             | 0.519193            | 6773 | 6777 | 0            |
| `flesch_kincaid_grade`      | 0             | 0.417355            | 6773 | 6777 | 0            |
| `hc_pos_verb_ratio`         | 0             | -0.914277           | 6777 | 6777 | 0            |
| `hc_std_sentence_len`       | 0             | 0.510936            | 6777 | 6777 | 0            |
| `char_entropy`              | 0             | 0.548653            | 6777 | 6777 | 0            |
| `hc_pos_pron_ratio`         | 0             | -0.738744           | 6777 | 6777 | 0            |
| `hc_pos_num_ratio`          | 0             | 0.523151            | 6777 | 6777 | 0            |
| `punct_ratio`               | 0             | 0.636858            | 6777 | 6777 | 0            |
| `hc_pos_noun_ratio`         | 0             | 0.829697            | 6777 | 6777 | 0            |
| `digit_ratio`               | 0             | 0.526522            | 6777 | 6777 | 0            |
| `hc_pos_aux_ratio`          | 0             | -0.697748           | 6777 | 6777 | 0            |
| `hc_stopword_ratio`         | 0             | -0.814419           | 6777 | 6777 | 0            |
| `hc_punct_paren_ratio`      | 0             | 0.625140            | 6777 | 6777 | 0            |
| `hc_digit_ratio`            | 0             | 0.526522            | 6777 | 6777 | 0            |
| `hc_punct_period_ratio`     | 8.88e-278     | -0.150194           | 6777 | 6777 | 7.37e-276    |
| `lm_mean_nll`               | 1.62e-248     | 0.463133            | 6777 | 6777 | 1.35e-246    |
| `flesch_reading_ease`       | 1.00e-212     | -0.405234           | 6773 | 6777 | 8.33e-211    |
| `hc_punct_dquote_ratio`     | 1.20e-195     | 0.284230            | 6777 | 6777 | 9.95e-194    |
| `hc_fw_do`                  | 1.06e-155     | -0.406312           | 6777 | 6777 | 8.82e-154    |
| `hc_avg_word_len`           | 2.37e-142     | 0.082422            | 6777 | 6777 | 1.97e-140    |
| `hc_fw_would`               | 4.44e-140     | 0.249037            | 6777 | 6777 | 3.69e-138    |
| `space_ratio`               | 2.38e-131     | -0.230542           | 6777 | 6777 | 1.98e-129    |
| `hc_avg_lemma_len`          | 6.82e-127     | 0.077661            | 6777 | 6777 | 5.66e-125    |
| `hc_pos_part_ratio`         | 3.08e-113     | -0.336904           | 6777 | 6777 | 2.55e-111    |
| `hc_std_word_len`           | 6.77e-108     | 0.339557            | 6777 | 6777 | 5.62e-106    |
| `hc_fw_is`                  | 5.57e-92      | -0.326555           | 6777 | 6777 | 4.63e-90     |
| `hc_pos_punct_ratio`        | 8.30e-85      | 0.428275            | 6777 | 6777 | 6.89e-83     |
| `hc_pos_sym_ratio`          | 9.45e-82      | 0.208302            | 6777 | 6777 | 7.84e-80     |
| `hc_fw_shall`               | 1.77e-80      | 0.125266            | 6777 | 6777 | 1.47e-78     |
| `hc_fw_an`                  | 1.13e-75      | 0.134294            | 6777 | 6777 | 9.41e-74     |
| `hc_punct_semicolon_ratio`  | 2.95e-73      | 0.228139            | 6777 | 6777 | 2.45e-71     |
| `tiktoken_len`              | 1.20e-72      | 0.405470            | 6777 | 6777 | 9.97e-71     |
| `hc_max_word_len`           | 1.62e-61      | 0.202209            | 6777 | 6777 | 1.34e-59     |
| `hc_punct_comma_ratio`      | 5.05e-56      | 0.326844            | 6777 | 6777 | 4.19e-54     |
| `hc_fw_was`                 | 3.36e-50      | 0.085687            | 6777 | 6777 | 2.79e-48     |
| `hc_fw_will`                | 3.02e-48      | -0.292082           | 6777 | 6777 | 2.51e-46     |
| `mean_word_len`             | 1.13e-46      | 0.076117            | 6777 | 6777 | 9.41e-45     |
| `hc_short_word_ratio`       | 2.21e-46      | -0.249239           | 6777 | 6777 | 1.84e-44     |
| `hc_mtld`                   | 1.03e-44      | 0.017986            | 6777 | 6777 | 8.53e-43     |
| `char_len`                  | 1.59e-44      | 0.333826            | 6777 | 6777 | 1.32e-42     |
| `hc_n_chars`                | 1.59e-44      | 0.333826            | 6777 | 6777 | 1.32e-42     |
| `hc_fw_be`                  | 9.36e-43      | -0.275529           | 6777 | 6777 | 7.77e-41     |
| `hc_n_tokens`               | 1.44e-42      | 0.338568            | 6777 | 6777 | 1.20e-40     |
| `hc_fw_had`                 | 2.87e-40      | 0.054464            | 6777 | 6777 | 2.38e-38     |
| `hc_fw_are`                 | 4.00e-40      | 0.109422            | 6777 | 6777 | 3.32e-38     |
| `word_len_ws`               | 1.17e-39      | 0.320589            | 6777 | 6777 | 9.72e-38     |
| `hc_fw_were`                | 4.96e-37      | 0.062729            | 6777 | 6777 | 4.12e-35     |
| `hc_uppercase_ratio`        | 7.08e-36      | 0.234362            | 6777 | 6777 | 5.88e-34     |
| `upper_ratio`               | 7.08e-36      | 0.234362            | 6777 | 6777 | 5.88e-34     |
| `hc_fw_has`                 | 3.64e-35      | 0.140034            | 6777 | 6777 | 3.02e-33     |
| `ttr`                       | 7.23e-34      | -0.224646           | 6777 | 6777 | 6.00e-32     |
| `hc_pos_cconj_ratio`        | 2.43e-33      | -0.125393           | 6777 | 6777 | 2.02e-31     |
| `hc_fw_a`                   | 5.22e-24      | 0.097593            | 6777 | 6777 | 4.33e-22     |
| `hc_pos_det_ratio`          | 5.10e-23      | -0.174256           | 6777 | 6777 | 4.23e-21     |
| `hc_punct_excl_ratio`       | 8.45e-22      | -0.146701           | 6777 | 6777 | 7.01e-20     |
| `hc_fw_does`                | 3.09e-21      | 0.034144            | 6777 | 6777 | 2.56e-19     |
| `hc_hapax_ratio`            | 6.91e-20      | -0.178474           | 6777 | 6777 | 5.74e-18     |
| `hc_corrected_ttr`          | 1.41e-18      | 0.192565            | 6777 | 6777 | 1.17e-16     |
| `hc_lemma_ttr`              | 3.31e-17      | -0.156309           | 6777 | 6777 | 2.74e-15     |
| `hc_ttr`                    | 2.84e-14      | -0.138791           | 6777 | 6777 | 2.36e-12     |
| `hc_n_alpha_tokens`         | 2.73e-13      | 0.217694            | 6777 | 6777 | 2.27e-11     |
| `hc_fw_should`              | 9.30e-13      | -0.051820           | 6777 | 6777 | 7.72e-11     |
| `hc_fw_been`                | 3.05e-12      | 0.024437            | 6777 | 6777 | 2.53e-10     |
| `hc_punct_apos_ratio`       | 8.58e-12      | -0.024329           | 6777 | 6777 | 7.13e-10     |
| `hc_pos_adp_ratio`          | 8.62e-06      | -0.115041           | 6777 | 6777 | 0.000715297  |
| `hc_n_sentences`            | 2.71e-05      | -0.018214           | 6777 | 6777 | 0.00225146   |
| `hc_punct_question_ratio`   | 3.66e-05      | -0.070269           | 6777 | 6777 | 0.00303392   |
| `hc_fw_being`               | 3.67e-05      | -0.114882           | 6777 | 6777 | 0.00304494   |
| `hc_pos_adj_ratio`          | 0.000135818   | 0.018408            | 6777 | 6777 | 0.0112729    |
| `hc_fw_the`                 | 0.000143185   | -0.050158           | 6777 | 6777 | 0.0118843    |
| `hc_pos_intj_ratio`         | 0.000748967   | 0.069321            | 6777 | 6777 | 0.0621643    |
| `hc_fw_did`                 | 0.284362      | -0.108261           | 6777 | 6777 | 1            |
| `hc_yule_k`                 | 0.750122      | 0.131724            | 6777 | 6777 | 1            |
| `hc_fw_have`                | 0.894017      | 0.035388            | 6777 | 6777 | 1            |
| `hc_pos_sconj_ratio`        | 1             | —                   | 6777 | 6777 | 1            |
| `hc_pos_conj_ratio`         | 1             | —                   | 6777 | 6777 | 1            |
| `hc_pos_propn_ratio`        | 1             | —                   | 6777 | 6777 | 1            |


**Интерпретация:** признаки **информативны** для разделения классов на train; часть эффекта может отражать **канал, длину, сценарий**, а не «универсальный след LLM».

**Минус:** **множественные сравнения** — даже с Bonferroni остаётся риск структурной корреляции признаков (дубликаты смысла: например `digit_ratio` и `hc_digit_ratio`), то есть тесты не независимы.

### 4.2 Mutual information с меткой (train, только исходный dense-блок в ячейке MI)

Источник: `v2/outputs/tables/features/mutual_info_train_label.csv`.


| feature                | MI(label)       |
| ---------------------- | --------------- |
| `flesch_kincaid_grade` | 0.132330908302  |
| `smog_index`           | 0.118452052732  |
| `digit_ratio`          | 0.11691914961   |
| `space_ratio`          | 0.107757472815  |
| `char_entropy`         | 0.104895012272  |
| `punct_ratio`          | 0.103522895735  |
| `tiktoken_len`         | 0.0911905466505 |
| `coleman_liau_index`   | 0.0867500224024 |
| `mean_word_len`        | 0.0818138417316 |
| `flesch_reading_ease`  | 0.0804119887378 |
| `char_len`             | 0.0737456117942 |
| `word_len_ws`          | 0.067253506463  |
| `upper_ratio`          | 0.0643137196346 |
| `ttr`                  | 0.0412588168783 |


**Минус:** MI посчитан по **14 dense** колонкам; полный вклад **68 HC** в отдельной таблице MI в ноутбуке не дублировался — для HC см. point-biserial ниже.

### 4.3 Диагностическая логистическая регрессия (dense + LM + TF-IDF, только val)

Источник: `v2/outputs/tables/features/diagnostic_logreg_val_metrics.csv`. Это **sanity-check**, не финальный отчётный baseline.


| split | ROC-AUC        | avg_precision  |
| ----- | -------------- | -------------- |
| val   | 0.958612133995 | 0.932674755336 |


### 4.4 SHAP (линейная модель на dense + HC + LM, подвыборка train)

Источник: `v2/outputs/tables/features/shap_linear_dense_top_features.csv`.

| feature | mean |SHAP| |
|---|---|
| `hc_n_alpha_tokens` | 4.89040742763 |
| `word_len_ws` | 2.97777339523 |
| `tiktoken_len` | 2.74407066323 |
| `hc_pos_noun_ratio` | 2.53002742574 |
| `hc_ttr` | 1.5845167647 |
| `punct_ratio` | 1.30292346586 |
| `hc_n_tokens` | 1.0896184784 |
| `coleman_liau_index` | 1.06442494821 |
| `hc_lemma_ttr` | 1.06003115161 |
| `hc_hapax_ratio` | 0.963188246049 |
| `hc_pos_pron_ratio` | 0.870453741419 |
| `ttr` | 0.704773928183 |
| `hc_n_chars` | 0.696385494343 |
| `char_len` | 0.696385494343 |
| `hc_pos_verb_ratio` | 0.63680221361 |
| `flesch_reading_ease` | 0.616757556612 |
| `hc_pos_det_ratio` | 0.572134020007 |
| `flesch_kincaid_grade` | 0.549239799986 |
| `hc_pos_adp_ratio` | 0.509257029825 |
| `hc_punct_paren_ratio` | 0.505689556584 |
| `lm_mean_nll` | 0.476470638027 |
| `hc_punct_dash_ratio` | 0.421938622039 |
| `hc_n_sentences` | 0.40945865215 |
| `hc_pos_adv_ratio` | 0.349752450303 |
| `hc_pos_adj_ratio` | 0.339488560972 |

**Вывод:** во вкладе линейной модели доминируют **объём текста**, **POS/стилометрия HC** и **пунктуация/читаемость**.

### 4.5 PCA (train, только 14 dense)

Источник: `v2/outputs/tables/features/pca_explained_variance_train.csv`. Сумма долей PC1+PC2 ≈ **0.53** — заметная часть вариации укладывается в плоскость; остальное (в т.ч. HC и TF-IDF) в этой визуализации не отражена.


| PC  | explained_variance_ratio |
| --- | ------------------------ |
| PC1 | 0.354225480551           |
| PC2 | 0.178471329223           |


### 4.6 Стабильность признаков: test_seen vs Claude holdout и медианы по `generator_lane` (LLM-only)

Источник: `v2/outputs/tables/features/stability_llm_slices_and_lanes.csv`. Для строк с `generator_lane` поля сравнения p-value пустые в CSV — в таблице «—».


| slice_a        | slice_b             | feature        | median_a           | median_b          | p_mw       | n_a  | n_b    |
| -------------- | ------------------- | -------------- | ------------------ | ----------------- | ---------- | ---- | ------ |
| test_seen      | test_claude_holdout | `char_len`     | 541.5              | 652.0             | 0.464528   | 908  | 2059.0 |
| test_seen      | test_claude_holdout | `tiktoken_len` | 113.5              | 122.0             | 0.00306077 | 908  | 2059.0 |
| test_seen      | test_claude_holdout | `ttr`          | 0.6653333333333333 | 0.744             | 7.78e-39   | 908  | 2059.0 |
| test_seen      | test_claude_holdout | `char_entropy` | 4.227915876794047  | 4.306405374105465 | 1.80e-88   | 908  | 2059.0 |
| test_seen      | test_claude_holdout | `lm_mean_nll`  | 3.740877628326416  | 4.773796558380127 | 1.12e-156  | 908  | 2059.0 |
| generator_lane | seen_openai         | `char_len`     | 482.0              | —                 | —          | 4933 | —      |
| generator_lane | seen_openai         | `tiktoken_len` | 99.0               | —                 | —          | 4933 | —      |
| generator_lane | seen_openai         | `ttr`          | 0.7087378640776699 | —                 | —          | 4933 | —      |
| generator_lane | seen_openai         | `char_entropy` | 4.23086148881892   | —                 | —          | 4933 | —      |
| generator_lane | seen_openai         | `lm_mean_nll`  | 3.7293953895568848 | —                 | —          | 4933 | —      |
| generator_lane | seen_mistral        | `char_len`     | 569.0              | —                 | —          | 3305 | —      |
| generator_lane | seen_mistral        | `tiktoken_len` | 126.0              | —                 | —          | 3305 | —      |
| generator_lane | seen_mistral        | `ttr`          | 0.636986301369863  | —                 | —          | 3305 | —      |
| generator_lane | seen_mistral        | `char_entropy` | 4.234421835096375  | —                 | —          | 3305 | —      |
| generator_lane | seen_mistral        | `lm_mean_nll`  | 3.9286766052246094 | —                 | —          | 3305 | —      |
| generator_lane | holdout_claude      | `char_len`     | 652.0              | —                 | —          | 2059 | —      |
| generator_lane | holdout_claude      | `tiktoken_len` | 122.0              | —                 | —          | 2059 | —      |
| generator_lane | holdout_claude      | `ttr`          | 0.744              | —                 | —          | 2059 | —      |
| generator_lane | holdout_claude      | `char_entropy` | 4.306405374105465  | —                 | —          | 2059 | —      |
| generator_lane | holdout_claude      | `lm_mean_nll`  | 4.773796558380127  | —                 | —          | 2059 | —      |


**Вывод:** признаки, завязанные на LM и TTR, **не стабильны** между seen и Claude; интерпретировать «высокий AUC на test_seen» как гарантию на Claude **нельзя**.

### 4.7 Legacy HC: связь с меткой на train (point-biserial)

Источник: `v2/outputs/tables/features/hc_pointbiserial_train_label.csv`.


| feature                     | r (point-biserial) | p           |
| --------------------------- | ------------------ | ----------- |
| `hc_pos_verb_ratio`         | 0.415781966547     | 0           |
| `hc_pos_noun_ratio`         | -0.383208170472    | 0           |
| `hc_stopword_ratio`         | 0.37716354836      | 0           |
| `hc_pos_pron_ratio`         | 0.346513263964     | 0           |
| `hc_pos_aux_ratio`          | 0.329424657872     | 0           |
| `hc_punct_paren_ratio`      | -0.298355741624    | 1.01e-276   |
| `hc_punct_colon_ratio`      | -0.265895705544    | 4.30e-218   |
| `hc_digit_ratio`            | -0.254604034164    | 1.60e-199   |
| `hc_pos_num_ratio`          | -0.253078926302    | 4.40e-197   |
| `hc_pos_adv_ratio`          | 0.252240142187     | 9.49e-196   |
| `hc_std_sentence_len`       | -0.247535634526    | 2.32e-188   |
| `hc_pos_punct_ratio`        | -0.209405338197    | 3.60e-134   |
| `hc_fw_do`                  | 0.19910333701      | 3.18e-121   |
| `hc_punct_dash_ratio`       | -0.192338667838    | 4.17e-113   |
| `hc_avg_sentence_len_words` | -0.184956128829    | 1.37e-104   |
| `hc_avg_sentence_len_chars` | -0.179803099156    | 7.43e-99    |
| `hc_std_word_len`           | -0.167395435381    | 9.41e-86    |
| `hc_n_tokens`               | -0.166921336254    | 2.85e-85    |
| `hc_pos_part_ratio`         | 0.166123641277     | 1.82e-84    |
| `hc_n_chars`                | -0.164647443249    | 5.52e-83    |
| `hc_punct_comma_ratio`      | -0.161294020917    | 1.13e-79    |
| `hc_fw_is`                  | 0.161155232985     | 1.55e-79    |
| `hc_fw_will`                | 0.144518597966     | 3.65e-64    |
| `hc_punct_dquote_ratio`     | -0.140711660236    | 6.84e-61    |
| `hc_fw_be`                  | 0.136485372205     | 2.31e-57    |
| `hc_short_word_ratio`       | 0.123671979408     | 2.43e-47    |
| `hc_fw_would`               | -0.123573046526    | 2.87e-47    |
| `hc_uppercase_ratio`        | -0.116393127233    | 4.25e-42    |
| `hc_punct_semicolon_ratio`  | -0.113343002308    | 5.36e-40    |
| `hc_n_alpha_tokens`         | -0.108215756497    | 1.36e-36    |
| `hc_pos_sym_ratio`          | -0.103598435331    | 1.16e-33    |
| `hc_max_word_len`           | -0.100599127124    | 7.89e-32    |
| `hc_corrected_ttr`          | -0.0958465005687   | 4.92e-29    |
| `hc_hapax_ratio`            | 0.0888901907124    | 3.46e-25    |
| `hc_pos_det_ratio`          | 0.0868056015876    | 4.31e-24    |
| `hc_lemma_ttr`              | 0.077922387203     | 1.04e-19    |
| `hc_punct_period_ratio`     | 0.0748915199287    | 2.54e-18    |
| `hc_punct_excl_ratio`       | 0.0731594976394    | 1.49e-17    |
| `hc_fw_has`                 | -0.0698509966742   | 3.91e-16    |
| `hc_ttr`                    | 0.0692342988085    | 7.07e-16    |
| `hc_fw_an`                  | -0.0670009735743   | 5.79e-15    |
| `hc_yule_k`                 | -0.0657243312341   | 1.87e-14    |
| `hc_pos_cconj_ratio`        | 0.062578212594     | 3.06e-13    |
| `hc_fw_shall`               | -0.0625152612106   | 3.23e-13    |
| `hc_pos_adp_ratio`          | 0.0574299222239    | 2.22e-11    |
| `hc_fw_being`               | 0.0573506642733    | 2.36e-11    |
| `hc_fw_are`                 | -0.0546335286228   | 1.96e-10    |
| `hc_fw_did`                 | 0.0540554246878    | 3.03e-10    |
| `hc_fw_a`                   | -0.048741934131    | 1.37e-08    |
| `hc_fw_was`                 | -0.0428075100389   | 6.18e-07    |
| `hc_avg_word_len`           | -0.0411792597458   | 1.62e-06    |
| `hc_avg_lemma_len`          | -0.0388039363963   | 6.22e-06    |
| `hc_punct_question_ratio`   | 0.0351154916275    | 4.33e-05    |
| `hc_pos_intj_ratio`         | -0.0346423073801   | 5.48e-05    |
| `hc_fw_were`                | -0.0313511744313   | 0.000261714 |
| `hc_fw_had`                 | -0.027224012275    | 0.00152565  |
| `hc_fw_should`              | 0.0259030764296    | 0.00256212  |
| `hc_fw_the`                 | 0.0250729150615    | 0.00350909  |
| `hc_fw_have`                | -0.0176923156741   | 0.0394234   |
| `hc_fw_does`                | -0.0170705511642   | 0.0468844   |
| `hc_fw_been`                | -0.0122183693347   | 0.154909    |
| `hc_punct_apos_ratio`       | 0.0121644360476    | 0.156739    |
| `hc_pos_adj_ratio`          | -0.00920449144011  | 0.283933    |
| `hc_n_sentences`            | 0.00910747820878   | 0.289039    |
| `hc_mtld`                   | -0.00899348645185  | 0.295118    |


**Минусы HC-блока:**

- Теггер и стоп-слова **ориентированы на английский**; несоответствие языка в строке Core искажает POS/функциональные доли.
- NLTK ≠ spaCy из v1: **прямое численное сравнение с thesis v1** по HC недопустимо без пересчёта на одном и том же тулчейне.

### 4.8 Медиана `char_len` по каналу и семейству сценария (train)

`v2/outputs/tables/features/median_char_len_by_channel_train.csv`:


| channel | median char_len (label=0) | median char_len (label=1) |
| ------- | ------------------------- | ------------------------- |
| email   | 1046.0                    | 605.0                     |
| qa      | 711.0                     | 1188.0                    |
| sms     | 114.0                     | 64.5                      |


`v2/outputs/tables/features/median_char_len_by_scenario_family_train.csv`:


| scenario_family        | median char_len (label=0) | median char_len (label=1) |
| ---------------------- | ------------------------- | ------------------------- |
| advance_fee_scam_email | 2518.5                    | 2067.0                    |
| financial_qa           | 711.0                     | 1188.0                    |
| fraud_sms_deceptive    | 145.0                     | 122.0                     |
| legitimate_email       | 814.0                     | 210.0                     |
| legitimate_sms         | 58.0                      | 34.0                      |
| phishing_email         | 769.5                     | 587.5                     |


### 4.9 HC60: различия human vs seen_llm на train (Mann–Whitney + Bonferroni)

Источник: `v2/outputs/tables/hc60_v2/mannwhitney_train_human_vs_seenllm.csv`. В файле **60** признаков `hc60_*`; ниже — **12** с наибольшим |Cohen’s d| (human − LLM).


| feature | Cohen d (human−LLM) | p_bonferroni |
| --- | --- | --- |
| `hc60_lexical_burstiness` | 0.7792 | 0 |
| `hc60_stopword_ratio` | −0.6741 | 0 |
| `hc60_modal_ratio` | −0.5987 | 0 |
| `hc60_punctuation_ratio` | 0.5767 | 4.64e-251 |
| `hc60_non_alnum_symbol_ratio` | 0.5767 | 4.64e-251 |
| `hc60_std_sentence_len_words` | 0.5573 | 0 |
| `hc60_pronoun_ratio` | −0.5518 | 7.10e-309 |
| `hc60_adverb_like_ratio` | −0.5436 | 0 |
| `hc60_colon_ratio` | 0.5318 | 0 |
| `hc60_digit_ratio` | 0.5092 | 0 |
| `hc60_smog_index` | 0.5017 | 0 |
| `hc60_conjunction_ratio` | −0.4458 | 3.65e-206 |


**Вывод:** картина согласуется с legacy-треком: сильные сдвиги по **стоп-словам, модальности, пунктуации, читаемости** и эвристикам токен-классов.

### 4.10 HC60: сдвиг распределений `test_seen` vs `test_claude_binary` (Mann–Whitney по LLM, `label = 1`)

Источник: `v2/outputs/tables/hc60_v2/drift_test_seen_vs_test_claude_binary.csv`. Сравниваются только LLM-строки на основном тесте (seen) и на бинарном Claude-срезе; ниже **8** признаков с наименьшим p-value (наиболее выраженный сдвиг медиан).


| feature | p_mann_whitney | median (test_seen LLM) | median (test_claude_binary LLM) |
| --- | --- | --- | --- |
| `hc60_stopword_ratio` | 3.18e-161 | 0.476 | 0.385 |
| `hc60_short_word_ratio` | 4.61e-124 | 0.410 | 0.333 |
| `hc60_avg_token_length_chars` | 8.87e-119 | 4.5 | 4.93 |
| `hc60_avg_word_len_chars` | 8.87e-119 | 4.5 | 4.93 |
| `hc60_chars_per_word` | 1.99e-106 | 5.86 | 6.36 |
| `hc60_long_word_ratio` | 2.00e-99 | 0.211 | 0.269 |
| `hc60_coleman_liau_index` | 1.45e-97 | 8.57 | 11.02 |
| `hc60_whitespace_ratio` | 1.48e-96 | 0.176 | 0.162 |


**Вывод:** даже внутри класса LLM распределения HC60 **различаются** между seen и Claude-binary пулом; это согласуется с ожидаемым **gap** между `test_seen` и `test_claude_binary` в метриках классификаторов (§6.2).

---

## 5. Графические артефакты (встроены)

Пути картинок заданы **относительно этого файла** (`v2/data/docs/draft_conclusion.md` → `../../outputs/...`).

### 5.1 EDA (legacy track, dense + HC)

![`char_len` по `core_eval_slice`, human vs LLM](../../outputs/figures/features/box_char_len_by_eval_slice.png)

![Корреляции 14 dense-признаков на train](../../outputs/figures/features/dense_correlation_heatmap_train.png)

![PCA(2D) по dense, цвет по `label`](../../outputs/figures/features/pca_train_by_label.png)

![PCA по dense, цвет по `scenario_family`](../../outputs/figures/features/pca_train_by_scenario.png)

![Топ-20 HC по |point-biserial|](../../outputs/figures/features/hc_pointbiserial_top20_train.png)

![KDE топ-8 HC, human vs LLM](../../outputs/figures/features/hc_kde_top8_by_label_train.png)

![Корреляции между топ-24 HC](../../outputs/figures/features/hc_correlation_top24_train.png)

### 5.2 Диагностические baseline (classical_ml, dense + HC + LM + TF-IDF)

![LR + dense + HC + LM + TF-IDF — ROC](../../outputs/figures/classical_ml/lr_dense_tfidf_roc.png)

![LR + dense + HC + LM + TF-IDF — PR](../../outputs/figures/classical_ml/lr_dense_tfidf_pr.png)

![LR + dense + HC + LM + TF-IDF — confusion](../../outputs/figures/classical_ml/lr_dense_tfidf_confusion.png)

![LR только числовой блок — ROC](../../outputs/figures/classical_ml/lr_dense_only_roc.png)

![LR только числовой блок — PR](../../outputs/figures/classical_ml/lr_dense_only_pr.png)

![LR только числовой блок — confusion](../../outputs/figures/classical_ml/lr_dense_only_confusion.png)

![Random Forest — ROC](../../outputs/figures/classical_ml/rf_dense_roc.png)

![Random Forest — PR](../../outputs/figures/classical_ml/rf_dense_pr.png)

![Random Forest — confusion](../../outputs/figures/classical_ml/rf_dense_confusion.png)

![XGBoost — ROC](../../outputs/figures/classical_ml/xgb_dense_roc.png)

![XGBoost — PR](../../outputs/figures/classical_ml/xgb_dense_pr.png)

![XGBoost — confusion](../../outputs/figures/classical_ml/xgb_dense_confusion.png)

На ROC/PR для **test_claude_holdout** на графиках отмечено **«N/A (один класс в y)»** — согласовано с протоколом в разделе 2.

### 5.3 HC60: корреляции, SHAP и кривые baseline (все 19 моделей на одной фигуре)

![Корреляции HC60 на train](../../outputs/figures/hc60_v2/corr_heatmap_train.png)

![SHAP bar (линейная модель, подвыборка train)](../../outputs/figures/hc60_v2/shap_bar_linear_train_sub.png)

![ROC, все эксперименты — val](../../outputs/figures/hc60_v2/roc_all_val.png)

![ROC, все эксперименты — test_seen](../../outputs/figures/hc60_v2/roc_all_test_seen.png)

![ROC, все эксперименты — test_claude_binary](../../outputs/figures/hc60_v2/roc_all_test_claude_binary.png)

![F1 vs recall, все эксперименты — val](../../outputs/figures/hc60_v2/f1_vs_recall_all_val.png)

![F1 vs recall, все эксперименты — test_seen](../../outputs/figures/hc60_v2/f1_vs_recall_all_test_seen.png)

*После полного прогона `01_core_hc60_baselines.ipynb` здесь же появятся `f1_vs_recall_all_test_claude_binary.png` и `pr_all_test_claude_binary.png` (кривые для `test_claude_binary`).*

![Precision–recall, все эксперименты — val](../../outputs/figures/hc60_v2/pr_all_val.png)

![Precision–recall, все эксперименты — test_seen](../../outputs/figures/hc60_v2/pr_all_test_seen.png)

![Scatter: test_seen ROC-AUC vs test_claude_binary ROC-AUC](../../outputs/figures/hc60_v2/scatter_leaderboard_seen_vs_claude_binary.png)

---

## 6. Диагностические baseline (полная таблица метрик по сплитам)

Источник: `v2/outputs/tables/classical_ml/baseline_metrics_by_split.csv`. Пустые ячейки в CSV (нет метрики) показаны как «—».


| model          | split               | accuracy           | balanced_accuracy  | roc_auc            | avg_precision      | f1                 | precision          | recall             | llm_predicted_rate |
| -------------- | ------------------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| lr_dense_tfidf | val                 | 0.9186991869918699 | 0.9186991869918699 | 0.9637650639759775 | 0.9397673805254869 | 0.9204052098408104 | 0.9014883061658399 | 0.9401330376940134 | —                  |
| lr_dense_tfidf | test_seen           | 0.9168502202643172 | 0.9168502202643172 | 0.9720448679384426 | 0.9693583007064004 | 0.9190348525469169 | 0.8955067920585162 | 0.9438325991189427 | —                  |
| lr_dense_tfidf | test_claude_holdout | 0.6318601262748907 | 0.6318601262748907 | —                  | —                  | 0.7744047619047619 | 1.0                | 0.6318601262748907 | 0.6318601262748907 |
| lr_dense_only  | val                 | 0.9065040650406504 | 0.9065040650406504 | 0.9554394193407769 | 0.9317342895452606 | 0.9081669691470055 | 0.8922967189728959 | 0.9246119733924612 | —                  |
| lr_dense_only  | test_seen           | 0.9052863436123348 | 0.9052863436123348 | 0.9674248966601331 | 0.9642516779717794 | 0.9071274298056156 | 0.8898305084745762 | 0.9251101321585903 | —                  |
| lr_dense_only  | test_claude_holdout | 0.4706168042739194 | 0.4706168042739194 | —                  | —                  | 0.6400264200792602 | 1.0                | 0.4706168042739194 | 0.4706168042739194 |
| rf_dense       | val                 | 0.9571322985957132 | 0.9571322985957131 | 0.9933366437070942 | 0.9927645728403147 | 0.9568452380952381 | 0.9632958801498127 | 0.950480413895048  | —                  |
| rf_dense       | test_seen           | 0.9708149779735683 | 0.9708149779735683 | 0.9937753498030235 | 0.9944740320359565 | 0.9708310401761144 | 0.9702970297029703 | 0.9713656387665198 | —                  |
| rf_dense       | test_claude_holdout | 0.5541525012141817 | 0.5541525012141817 | —                  | —                  | 0.713125           | 1.0                | 0.5541525012141817 | 0.5541525012141817 |
| xgb_dense      | val                 | 0.9708056171470806 | 0.9708056171470806 | 0.9964612869269188 | 0.9966201207808214 | 0.9707081942899518 | 0.9739583333333334 | 0.967479674796748  | —                  |
| xgb_dense      | test_seen           | 0.9752202643171806 | 0.9752202643171806 | 0.9977367113664151 | 0.9978447593114488 | 0.9751518498067366 | 0.9778516057585825 | 0.9724669603524229 | —                  |
| xgb_dense      | test_claude_holdout | 0.6498300145701797 | 0.6498300145701797 | —                  | —                  | 0.7877539005004416 | 1.0                | 0.6498300145701797 | 0.6498300145701797 |
| lr_dense_tfidf | test_full_aggregate | 0.7654193548387097 | 0.8086009244122945 | 0.9128281878935545 | 0.9649735508687527 | 0.8260287081339713 | 0.9557130203720107 | 0.7273340074148972 | —                  |
| lr_dense_only  | test_full_aggregate | 0.6743225806451613 | 0.7475846647928981 | 0.8573786690304064 | 0.9477033037581453 | 0.7413934426229508 | 0.9456351280710925 | 0.6097067745197169 | —                  |
| rf_dense       | test_full_aggregate | 0.7494193548387097 | 0.8260489095171706 | 0.9305187458519486 | 0.9779452399378987 | 0.8064580426549731 | 0.9868292682926829 | 0.6818335018537243 | —                  |
| xgb_dense      | test_full_aggregate | 0.8023225806451613 | 0.8632705724793581 | 0.9603004562671025 | 0.9875251513651453 | 0.8529185867895546 | 0.9910754127621597 | 0.7485675766767779 | —                  |


**Интерпретация для `test_claude_holdout`:** при пороге 0.5 доля Claude-текстов, классифицированных как LLM; **1.0 − llm_predicted_rate** — доля «принятых за human». Например, у `lr_dense_only` **llm_predicted_rate ≈ 0.471** — около **53%** holdout-текстов уходят в сторону human по этому порогу (симптом сдвига, не AUC).

### 6.1 Разрез по `scenario_family` на test_seen (`lr_dense_tfidf`)

Источник: `v2/outputs/tables/classical_ml/baseline_by_scenario_family_test_seen.csv`.


| scenario_family        | n   | accuracy           | balanced_accuracy  | roc_auc            | avg_precision      | f1                 | precision          | recall             |
| ---------------------- | --- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| financial_qa           | 164 | 0.7621951219512195 | 0.7621951219512195 | 0.8306067816775728 | 0.8526161387774985 | 0.7636363636363637 | 0.7590361445783133 | 0.7682926829268293 |
| legitimate_sms         | 160 | 0.7375             | 0.7375             | 0.86609375         | 0.8771906085822593 | 0.7613636363636364 | 0.6979166666666666 | 0.8375             |
| fraud_sms_deceptive    | 172 | 0.8255813953488372 | 0.8255813953488372 | 0.9288804759329368 | 0.9238092713897912 | 0.8275862068965517 | 0.8181818181818182 | 0.8372093023255814 |
| legitimate_email       | 400 | 0.9625             | 0.9624999999999999 | 0.9913749999999999 | 0.9873775923410765 | 0.9631449631449631 | 0.9468599033816425 | 0.98               |
| phishing_email         | 600 | 0.9616666666666667 | 0.9616666666666667 | 0.9977             | 0.9979350910398987 | 0.9629629629629629 | 0.9314641744548287 | 0.9966666666666667 |
| advance_fee_scam_email | 320 | 0.99375            | 0.99375            | 0.9999609375       | 0.9999611801242236 | 0.9937888198757764 | 0.9876543209876543 | 1.0                |


**Вывод:** email-сценарии с большим n дают очень высокие значения; **SMS и QA** — ниже и на меньших выборках; обобщать «одну цифру AUC на весь Core» **нельзя**.

### 6.2 HC60-only baselines (19 моделей, без TF-IDF и без legacy `hc_*`)

Источник метрик: `v2/outputs/tables/hc60_v2/baseline_all_runs.csv` (генерируется `01_core_hc60_baselines.ipynb`, модуль `v2/src/hc60_baseline_suite.py`). Порог F1 на тестах — **подобран на val** (колонка `val_tuned_threshold_f1` в CSV); ниже для компактности приведены ROC-AUC и F1 на основных срезах.

**Полная таблица (ROC-AUC и F1):**


| exp_id | name | val ROC-AUC | test_seen ROC-AUC | test_claude_binary ROC-AUC | test_seen F1 | test_claude_binary F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | lr_l2_c1 | 0.9407 | 0.9414 | 0.8243 | 0.8844 | 0.6983 |
| 2 | lr_l2_c01 | 0.9308 | 0.9311 | 0.8161 | 0.8765 | 0.6973 |
| 3 | linearsvc_calibrated | 0.9405 | 0.9460 | 0.8623 | 0.8657 | 0.7860 |
| 4 | ridge_classifier | 0.9278 | 0.9317 | 0.8658 | 0.8710 | 0.7792 |
| 5 | decision_tree | 0.9114 | 0.9117 | 0.8005 | 0.8843 | 0.7423 |
| 6 | rf_constrained | 0.9904 | 0.9893 | 0.9120 | 0.9439 | 0.7299 |
| 7 | rf_less_constrained | 0.9917 | 0.9911 | 0.9197 | 0.9511 | 0.7078 |
| 8 | extra_trees | 0.9943 | 0.9936 | 0.9060 | 0.9587 | 0.6692 |
| 9 | xgb_conservative | 0.9907 | 0.9895 | 0.9226 | 0.9473 | 0.7740 |
| 10 | xgb_richer | 0.9948 | 0.9946 | 0.9359 | 0.9656 | 0.7677 |
| 11 | hist_gradient_boosting | 0.9947 | 0.9942 | 0.9331 | 0.9635 | 0.7627 |
| 12 | gradient_boosting | 0.9895 | 0.9879 | 0.9113 | 0.9429 | 0.7615 |
| 13 | adaboost | 0.9844 | 0.9838 | 0.8859 | 0.9344 | 0.7129 |
| 14 | sgd_log | 0.9366 | 0.9355 | 0.8260 | 0.8799 | 0.7016 |
| 15 | bagging_dt | 0.9769 | 0.9780 | 0.8890 | 0.9191 | 0.7740 |
| 16 | knn_15 | 0.9813 | 0.9833 | 0.8752 | 0.9149 | 0.7705 |
| 17 | lr_structural_subset | 0.8812 | 0.8887 | 0.7747 | 0.8213 | 0.6869 |
| 18 | lr_lexical_diversity_only | 0.6324 | 0.6570 | 0.3458 | 0.6156 | 0.3373 |
| 19 | xgb_full_ablation_capstone | 0.9948 | 0.9946 | 0.9359 | 0.9656 | 0.7677 |


**Наблюдения по цифрам из CSV:**

- На **test_seen** лучшие ROC-AUC у деревьев/boosting: до **0.9946** (`xgb_richer`, дубликат-строка `xgb_full_ablation_capstone`).
- На **test_claude_binary** максимум ROC-AUC **0.9359** (те же модели); минимум **0.3458** (`lr_lexical_diversity_only` — намеренно узкий лексический блок).
- **F1 на test_claude_binary** заметно ниже, чем на test_seen, у сильных моделей (например **0.77** vs **0.97** у `xgb_richer`) — отражение более жёсткого сценария и порога, настроенного на val.

**Δ ROC-AUC (test_seen − test_claude_binary)** — таблица для анализа переносимости (тот же расчёт, что в ноутбуке; CSV: `v2/outputs/tables/hc60_v2/delta_roc_seen_minus_claude_binary.csv`):


| exp_id | name | test_seen ROC-AUC | test_claude_binary ROC-AUC | Δ (seen − claude_bin) |
| --- | --- | --- | --- | --- |
| 18 | lr_lexical_diversity_only | 0.6570 | 0.3458 | 0.3113 |
| 1 | lr_l2_c1 | 0.9414 | 0.8243 | 0.1171 |
| 2 | lr_l2_c01 | 0.9311 | 0.8161 | 0.1150 |
| 17 | lr_structural_subset | 0.8887 | 0.7747 | 0.1140 |
| 5 | decision_tree | 0.9117 | 0.8005 | 0.1112 |
| 14 | sgd_log | 0.9355 | 0.8260 | 0.1094 |
| 16 | knn_15 | 0.9833 | 0.8752 | 0.1081 |
| 13 | adaboost | 0.9838 | 0.8859 | 0.0979 |
| 15 | bagging_dt | 0.9780 | 0.8890 | 0.0890 |
| 8 | extra_trees | 0.9936 | 0.9060 | 0.0876 |
| 3 | linearsvc_calibrated | 0.9460 | 0.8623 | 0.0837 |
| 6 | rf_constrained | 0.9893 | 0.9120 | 0.0773 |
| 12 | gradient_boosting | 0.9879 | 0.9113 | 0.0767 |
| 7 | rf_less_constrained | 0.9911 | 0.9197 | 0.0714 |
| 9 | xgb_conservative | 0.9895 | 0.9226 | 0.0669 |
| 4 | ridge_classifier | 0.9317 | 0.8658 | 0.0659 |
| 11 | hist_gradient_boosting | 0.9942 | 0.9331 | 0.0611 |
| 10 | xgb_richer | 0.9946 | 0.9359 | 0.0587 |
| 19 | xgb_full_ablation_capstone | 0.9946 | 0.9359 | 0.0587 |


**Интерпретация:** наименьший gap (Δ ≈ **0.059**) у сильных бустингов; наибольший — у узких/линейных конфигураций и `lr_lexical_diversity_only`. Это **не** доказывает причинную устойчивость к Claude, но количественно фиксирует разрыв между основным тестом и бинарным Claude-срезом при фиксированном протоколе.

---

## 7. Критические минусы, риски и ограничения доверия

1. **`test_claude_only` без human.** На этом срезе нельзя строить классическую бинарную кривую ошибок; `llm_predicted_rate` — огрублённый индикатор, чувствительный к порогу 0.5. Для полноценных ROC/PR/F1 по Claude используйте **`test_claude_binary`** (§2, §6.2), не смешивая интерпретацию с one-class срезом.
2. **Утечка признаков:** TF-IDF и скейлеры fit **только на train** — корректно; но **высокая размерность + сильные модели (RF/XGB)** дают риск **переобучения на val** при будущем тюнинге — сейчас гиперпараметры не системно подбирались.
3. **Множественные сравнения и коррелированные фичи:** десятки тестов Mann–Whitney; дубликаты смысла (dense vs `hc_*`, частично пересекающиеся `hc60_*`-пары).
4. **Язык и токенизация:** legacy HC завязаны на английский NLTK; HC60 — на эвристики под англоязычный Core; многоязычный шум остаётся источником ошибки.
5. **LM-scoring:** DistilGPT-2 ≠ генераторы Core; NLL — не «вероятность от автора», а предсказуемость для маленькой LM.
6. **Смещение по каналу и длине:** PCA на 14 dense не отражает HC/TF-IDF; медианы длины по каналам и сценариям см. таблицы в §4.8.
7. **Размер SMS/QA:** малые n в разрезе семьи → **широкие доверительные интервалы** для AUC (не посчитаны здесь явно).
8. **Два трека признаков:** сравнивать абсолютные AUC **legacy full-dim** и **HC60** как «улучшение/ухудшение» напрямую некорректно — разные входы и ёмкость модели.
9. **Воспроизводимость:** смена версий NLTK/sklearn/порядка строк может слегка сдвинуть веса; зафиксированы `uv.lock` и seed 42.

---

## 8. Насколько можно верить результатам и как их подавать


| Утверждение                                                                     | Уровень доверия                                 | Комментарий                                                      |
| ------------------------------------------------------------------------------- | ----------------------------------------------- | ---------------------------------------------------------------- |
| На train/val/test_seen human vs seen_llm признаки **статистически различаются** | Высокий                                         | Подкреплено Mann–Whitney, MI/SHAP, высоким AUC на сравнимом пуле |
| **Линейные/деревянные модели** хорошо разделяют классы на val/test_seen         | Высокий для относительного ранжирования методов | Цифры в §6 и §6.2                                                |
| На **`test_claude_binary`** HC60-модели сохраняют **высокий ROC-AUC** (boosting до ~0.94) | Средний                                         | Полноценная бинарная метрика; см. §6.2, §4.10; порог F1 с val    |
| На **`test_claude_only`** доля «предсказано как LLM» при 0.5 отражает лишь one-class поведение | Низкий                                          | `llm_predicted_rate`; не путать с ROC на `test_claude_binary`     |
| Признаки **причинно** объясняют «почему LLM»                                    | Низкий                                          | Корреляция/SHAP ≠ каузальность; смешение сценария и генератора   |
| Результат **переносится** на другие домены/языки                                | Очень низкий                                    | Оценка только на Core v2                                         |


**Рекомендуемая формулировка для текста ВКР:**  
«На сопоставимом пуле Core (train/val/test_seen) классические и стилометрические признаки демонстрируют сильную разделимость human vs seen_llm (см. таблицы §6 и кривые §5). Для Claude-holdout следует разделять: (i) **бинарный** срез `test_claude_binary`, где по HC60 признакам сохраняется высокий ROC-AUC, но F1 и gap относительно test_seen хуже, чем на seen-генераторах; (ii) **one-class** срез `test_claude_only`, где без human-негативов остаётся лишь грубый `llm_predicted_rate`. Параллельно фиксируется **сдвиг распределений** признаков между seen и Claude (§4.6, §4.10), поэтому о переносимости на новые генераторы без калибровки заявлять нельзя.»

---

## 9. Воспроизводимость

Исходные CSV/PNG лежат под `v2/outputs/tables/` и `v2/outputs/figures/`; встроенные таблицы собраны из этих CSV при подготовке документа (в т.ч. `hc60_v2/baseline_all_runs.csv`, `delta_roc_seen_minus_claude_binary.csv`). После полного прогона ноутбуков **сверьте** числа и обновите встроенные фрагменты; для HC60 убедитесь, что сгенерированы все файлы `roc_all_*.png`, `pr_all_*.png`, `f1_vs_recall_all_*.png` в `outputs/figures/hc60_v2/`.

---

*Черновик аналитического вывода.*