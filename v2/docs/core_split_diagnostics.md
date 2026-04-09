# Core — split diagnostics

Сгенерировано при сборке Core (UTC: 2026-04-08T23:04:58.210993+00:00). Все таблицы ниже — **факт** по `core_v2.jsonl` после сплита.

Источник спецификации: `v2/docs/dataset_design_final.md`, `v2/docs/dataset_contract.md`.

**Политика:** train / val / test_seen балансируются как **human vs seen_llm**; Claude holdout — отдельный add-on только в test.

## Сводка по строкам

- Всего строк: **20135**
- train / val / test (split): **13554** / **2706** / **3875**
- `core_eval_slice`: {'train': 13554, 'val': 2706, 'test_claude_holdout': 2059, 'test_seen': 1816}
- Claude holdout (`generator_lane`): **2059**

## Интерпретация test

- **test_full** (`core_test_full.jsonl`, `core_test.jsonl`): весь `split=test`; операционный union, не единственная главная метрика.
- **test_non_claude** (`core_test_non_claude.jsonl`): только `core_eval_slice=test_seen` — основной сопоставимый test (seen-генераторы).
- **test_claude_only** (`core_test_claude_only.jsonl`): `test_claude_holdout` — stress-test по holdout-генератору.

## `scenario_family` × `split`

| scenario_family | test | train | val |
|:---|:---|:---|:---|
| advance_fee_scam_email | 720 | 2400 | 480 |
| financial_qa | 164 | 1198 | 238 |
| fraud_sms_deceptive | 385 | 1278 | 254 |
| legitimate_email | 900 | 3000 | 600 |
| legitimate_sms | 356 | 1178 | 234 |
| phishing_email | 1350 | 4500 | 900 |


## `channel` × `split`

| channel | test | train | val |
|:---|:---|:---|:---|
| email | 2970 | 9900 | 1980 |
| qa | 164 | 1198 | 238 |
| sms | 741 | 2456 | 488 |


## `label` × `split`

| label | test | train | val |
|:---|:---|:---|:---|
| 0 | 908 | 6777 | 1353 |
| 1 | 2967 | 6777 | 1353 |


## `length_bin` × `split`

| length_bin | test | train | val |
|:---|:---|:---|:---|
| long | 332 | 2454 | 498 |
| medium | 1936 | 6251 | 1238 |
| short | 1607 | 4849 | 970 |


## `generator_lane` × `split`

| generator_lane | test | train | val |
|:---|:---|:---|:---|
| (none) | 990 | 7376 | 1472 |
| holdout_claude | 2059 | 0 | 0 |
| seen_mistral | 333 | 2477 | 495 |
| seen_openai | 493 | 3701 | 739 |


## `scenario_family` × `label` (counts)

| scenario_family | 0 | 1 |
|:---|:---|:---|
| advance_fee_scam_email | 1600 | 2000 |
| financial_qa | 800 | 800 |
| fraud_sms_deceptive | 852 | 1065 |
| legitimate_email | 2000 | 2500 |
| legitimate_sms | 786 | 982 |
| phishing_email | 3000 | 3750 |


## `scenario_family` × `split` × `label`

| scenario_family | split | 0 | 1 |
|:---|:---|:---|:---|
| advance_fee_scam_email | test | 160 | 560 |
| advance_fee_scam_email | train | 1200 | 1200 |
| advance_fee_scam_email | val | 240 | 240 |
| financial_qa | test | 82 | 82 |
| financial_qa | train | 599 | 599 |
| financial_qa | val | 119 | 119 |
| fraud_sms_deceptive | test | 86 | 299 |
| fraud_sms_deceptive | train | 639 | 639 |
| fraud_sms_deceptive | val | 127 | 127 |
| legitimate_email | test | 200 | 700 |
| legitimate_email | train | 1500 | 1500 |
| legitimate_email | val | 300 | 300 |
| legitimate_sms | test | 80 | 276 |
| legitimate_sms | train | 589 | 589 |
| legitimate_sms | val | 117 | 117 |
| phishing_email | test | 300 | 1050 |
| phishing_email | train | 2250 | 2250 |
| phishing_email | val | 450 | 450 |


## `channel` × `split` × `label`

| channel | split | 0 | 1 |
|:---|:---|:---|:---|
| email | test | 660 | 2310 |
| email | train | 4950 | 4950 |
| email | val | 990 | 990 |
| qa | test | 82 | 82 |
| qa | train | 599 | 599 |
| qa | val | 119 | 119 |
| sms | test | 166 | 575 |
| sms | train | 1228 | 1228 |
| sms | val | 244 | 244 |


## `fraudness` × `split` × `label`

| fraudness | split | 0 | 1 |
|:---|:---|:---|:---|
| fraud | test | 546 | 1909 |
| fraud | train | 4089 | 4089 |
| fraud | val | 817 | 817 |
| legitimate | test | 362 | 1058 |
| legitimate | train | 2688 | 2688 |
| legitimate | val | 536 | 536 |


## `length_bin` × `scenario_family` × `split`

| length_bin | scenario_family | test | train | val |
|:---|:---|:---|:---|:---|
| long | advance_fee_scam_email | 168 | 1260 | 253 |
| long | financial_qa | 61 | 418 | 89 |
| long | fraud_sms_deceptive | 10 | 76 | 16 |
| long | legitimate_email | 44 | 329 | 66 |
| long | legitimate_sms | 1 | 10 | 2 |
| long | phishing_email | 48 | 361 | 72 |
| medium | advance_fee_scam_email | 551 | 1129 | 225 |
| medium | financial_qa | 81 | 646 | 121 |
| medium | fraud_sms_deceptive | 266 | 959 | 190 |
| medium | legitimate_email | 309 | 1018 | 203 |
| medium | legitimate_sms | 29 | 209 | 41 |
| medium | phishing_email | 700 | 2290 | 458 |
| short | advance_fee_scam_email | 1 | 11 | 2 |
| short | financial_qa | 22 | 134 | 28 |
| short | fraud_sms_deceptive | 109 | 243 | 48 |
| short | legitimate_email | 547 | 1653 | 331 |
| short | legitimate_sms | 326 | 959 | 191 |
| short | phishing_email | 602 | 1849 | 370 |


## `source_family` × `scenario_family` × `split`

| source_family | scenario_family | test | train | val |
|:---|:---|:---|:---|:---|
| enron_ham | legitimate_email | 160 | 1201 | 241 |
| hc3_finance | financial_qa | 164 | 1198 | 238 |
| llm_holdout_claude | advance_fee_scam_email | 400 | 0 | 0 |
| llm_holdout_claude | fraud_sms_deceptive | 213 | 0 | 0 |
| llm_holdout_claude | legitimate_email | 500 | 0 | 0 |
| llm_holdout_claude | legitimate_sms | 196 | 0 | 0 |
| llm_holdout_claude | phishing_email | 750 | 0 | 0 |
| llm_seen_mistral | advance_fee_scam_email | 63 | 469 | 94 |
| llm_seen_mistral | fraud_sms_deceptive | 35 | 262 | 52 |
| llm_seen_mistral | legitimate_email | 82 | 613 | 123 |
| llm_seen_mistral | legitimate_sms | 32 | 232 | 46 |
| llm_seen_mistral | phishing_email | 121 | 901 | 180 |
| llm_seen_openai | advance_fee_scam_email | 97 | 731 | 146 |
| llm_seen_openai | fraud_sms_deceptive | 51 | 377 | 75 |
| llm_seen_openai | legitimate_email | 118 | 887 | 177 |
| llm_seen_openai | legitimate_sms | 48 | 357 | 71 |
| llm_seen_openai | phishing_email | 179 | 1349 | 270 |
| mendeley_sms_phishing | fraud_sms_deceptive | 37 | 277 | 55 |
| nazario | phishing_email | 300 | 2250 | 450 |
| nigerian_fraud | advance_fee_scam_email | 160 | 1200 | 240 |
| smishtank | fraud_sms_deceptive | 49 | 362 | 72 |
| sms_ham | legitimate_sms | 80 | 589 | 117 |
| spamassassin_ham | legitimate_email | 40 | 299 | 59 |


## `origin_model` × `scenario_family` × `split`

| origin_model | scenario_family | test | train | val |
|:---|:---|:---|:---|:---|
| anthropic/claude-3.5-haiku | advance_fee_scam_email | 400 | 0 | 0 |
| anthropic/claude-3.5-haiku | fraud_sms_deceptive | 213 | 0 | 0 |
| anthropic/claude-3.5-haiku | legitimate_email | 500 | 0 | 0 |
| anthropic/claude-3.5-haiku | legitimate_sms | 196 | 0 | 0 |
| anthropic/claude-3.5-haiku | phishing_email | 750 | 0 | 0 |
| human | advance_fee_scam_email | 160 | 1200 | 240 |
| human | financial_qa | 82 | 599 | 119 |
| human | fraud_sms_deceptive | 86 | 639 | 127 |
| human | legitimate_email | 200 | 1500 | 300 |
| human | legitimate_sms | 80 | 589 | 117 |
| human | phishing_email | 300 | 2250 | 450 |
| openai/chatgpt | financial_qa | 82 | 599 | 119 |
| openai/gpt-4o-mini | advance_fee_scam_email | 97 | 731 | 146 |
| openai/gpt-4o-mini | fraud_sms_deceptive | 51 | 377 | 75 |
| openai/gpt-4o-mini | legitimate_email | 118 | 887 | 177 |
| openai/gpt-4o-mini | legitimate_sms | 48 | 357 | 71 |
| openai/gpt-4o-mini | phishing_email | 179 | 1349 | 270 |
| openrouter/mistralai/mistral-small-3.1-24b-instruct | advance_fee_scam_email | 63 | 469 | 94 |
| openrouter/mistralai/mistral-small-3.1-24b-instruct | fraud_sms_deceptive | 35 | 262 | 52 |
| openrouter/mistralai/mistral-small-3.1-24b-instruct | legitimate_email | 82 | 613 | 123 |
| openrouter/mistralai/mistral-small-3.1-24b-instruct | legitimate_sms | 32 | 232 | 46 |
| openrouter/mistralai/mistral-small-3.1-24b-instruct | phishing_email | 121 | 901 | 180 |


## `generator_lane` × `scenario_family` × `split`

| generator_lane | scenario_family | test | train | val |
|:---|:---|:---|:---|:---|
| (none) | advance_fee_scam_email | 160 | 1200 | 240 |
| (none) | financial_qa | 164 | 1198 | 238 |
| (none) | fraud_sms_deceptive | 86 | 639 | 127 |
| (none) | legitimate_email | 200 | 1500 | 300 |
| (none) | legitimate_sms | 80 | 589 | 117 |
| (none) | phishing_email | 300 | 2250 | 450 |
| holdout_claude | advance_fee_scam_email | 400 | 0 | 0 |
| holdout_claude | fraud_sms_deceptive | 213 | 0 | 0 |
| holdout_claude | legitimate_email | 500 | 0 | 0 |
| holdout_claude | legitimate_sms | 196 | 0 | 0 |
| holdout_claude | phishing_email | 750 | 0 | 0 |
| seen_mistral | advance_fee_scam_email | 63 | 469 | 94 |
| seen_mistral | fraud_sms_deceptive | 35 | 262 | 52 |
| seen_mistral | legitimate_email | 82 | 613 | 123 |
| seen_mistral | legitimate_sms | 32 | 232 | 46 |
| seen_mistral | phishing_email | 121 | 901 | 180 |
| seen_openai | advance_fee_scam_email | 97 | 731 | 146 |
| seen_openai | fraud_sms_deceptive | 51 | 377 | 75 |
| seen_openai | legitimate_email | 118 | 887 | 177 |
| seen_openai | legitimate_sms | 48 | 357 | 71 |
| seen_openai | phishing_email | 179 | 1349 | 270 |


## `time_band` × `scenario_family` × `split`

| time_band | scenario_family | test | train | val |
|:---|:---|:---|:---|:---|
| legacy | advance_fee_scam_email | 160 | 1200 | 240 |
| legacy | financial_qa | 82 | 599 | 119 |
| legacy | fraud_sms_deceptive | 86 | 639 | 127 |
| legacy | legitimate_email | 200 | 1500 | 300 |
| legacy | legitimate_sms | 80 | 589 | 117 |
| legacy | phishing_email | 151 | 1129 | 225 |
| modern | advance_fee_scam_email | 560 | 1200 | 240 |
| modern | financial_qa | 82 | 599 | 119 |
| modern | fraud_sms_deceptive | 299 | 639 | 127 |
| modern | legitimate_email | 700 | 1500 | 300 |
| modern | legitimate_sms | 276 | 589 | 117 |
| modern | phishing_email | 1199 | 3371 | 675 |


## `scenario_family` × `core_eval_slice` × `label`

| scenario_family | core_eval_slice | 0 | 1 |
|:---|:---|:---|:---|
| advance_fee_scam_email | test_claude_holdout | 0 | 400 |
| advance_fee_scam_email | test_seen | 160 | 160 |
| advance_fee_scam_email | train | 1200 | 1200 |
| advance_fee_scam_email | val | 240 | 240 |
| financial_qa | test_seen | 82 | 82 |
| financial_qa | train | 599 | 599 |
| financial_qa | val | 119 | 119 |
| fraud_sms_deceptive | test_claude_holdout | 0 | 213 |
| fraud_sms_deceptive | test_seen | 86 | 86 |
| fraud_sms_deceptive | train | 639 | 639 |
| fraud_sms_deceptive | val | 127 | 127 |
| legitimate_email | test_claude_holdout | 0 | 500 |
| legitimate_email | test_seen | 200 | 200 |
| legitimate_email | train | 1500 | 1500 |
| legitimate_email | val | 300 | 300 |
| legitimate_sms | test_claude_holdout | 0 | 196 |
| legitimate_sms | test_seen | 80 | 80 |
| legitimate_sms | train | 589 | 589 |
| legitimate_sms | val | 117 | 117 |
| phishing_email | test_claude_holdout | 0 | 750 |
| phishing_email | test_seen | 300 | 300 |
| phishing_email | train | 2250 | 2250 |
| phishing_email | val | 450 | 450 |


---




<!-- AUTO:test_claude_binary:start -->
## test_claude_binary (derived slice, UTC: 2026-04-09T02:29:53.297052+00:00)

Файл: `v2/data/interim/assembled/core_test_claude_binary.jsonl`. Human companions набраны из **резерва** `dataset1_human.jsonl` (строки **не** вошедшие в train/val/test_seen), с тирами согласования страт.

- Строк в срезе: **3692**
- Проверки (см. также `core_manifest.json` → `test_claude_binary.checks`):
  - **claude_binary_has_both_labels**: ok=`True`
  - **claude_binary_no_overlap_with_train_val_testseen**: ok=`True`
  - **claude_binary_family_balance**: ok=`False`
  - **claude_binary_match_quality_check**: ok=`True`

### Доступность vs отобранные пары по `scenario_family`

| scenario_family | n_claude | human_reserve | pairs_written | причина shortfall |
|---|---:|---:|---:|---|
| `advance_fee_scam_email` | 400 | 1633 | 400 | — |
| `fraud_sms_deceptive` | 213 | 0 | 0 | no_human_rows_in_reserve_all_main_pool_consumed |
| `legitimate_email` | 500 | 16841 | 500 | — |
| `legitimate_sms` | 196 | 3275 | 196 | — |
| `phishing_email` | 750 | 2201 | 750 | — |

<!-- AUTO:test_claude_binary:end -->
