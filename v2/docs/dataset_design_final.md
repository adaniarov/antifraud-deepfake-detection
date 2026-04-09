# Dataset Design — итоговая версия Core-датасета

**Проект:** обнаружение LLM-сгенерированного текста в антифрод-системах  
**Статус:** финальная спецификация Core-датасета  
**Назначение:** на основе этого документа собрать итоговый Core-датасет, после чего перейти к генерации LLM-части, сборке train/val/test и обучению baseline-моделей.

---

## 1. Цель и границы Core-датасета

Core-датасет предназначен для **основных экспериментов**:

- classical ML baseline,
- transformer baseline,
- held-out generator evaluation,
- основная оценка `human vs llm` в антифродовом домене.

В Core должны входить только те источники и сценарии, где есть достаточно чистое и методологически корректное соответствие между:

- `label`: `human` / `llm`
- `fraudness`: `fraud` / `legitimate`
- `channel`: `email` / `sms` / `qa`
- `scenario_family`
- `length_bin`
- `time_band`

Core **не должен** включать широкие и внутренне неоднородные spam-пулы как монолитные классы. Именно эта ошибка была одной из главных проблем ранних версий дизайна.

---

## 1.1. Что именно должен измерять Core

Core-датасет должен поддерживать следующий основной исследовательский вопрос:

> Насколько модель способна различать `human-written` и `LLM-generated` тексты внутри нескольких **согласованных антифродовых и контрольных сценариев**, а не на одном узком источнике и не на смеси широких spam-пулов?

То есть Core — это не «универсальный датасет для детекции любого AI-текста», а **композиционный benchmark**, состоящий из нескольких matched slices. Метрики, полученные на нём, должны интерпретироваться как слайсово, так и агрегированно.

---

## 1.2. Что Core не должен делать

Core не должен:

- подменять задачу `human vs llm` задачей `spam vs ham`;
- опираться на широкие и внутренне неоднородные spam-контейнеры как на готовые fraud-классы;
- использовать unmatched LLM scenarios без human-side опоры;
- раздуваться за счёт нерелевантных жанров;
- создавать видимость полной жанровой однородности там, где её нет.

---

## 1.3. Критически честная оценка корректности схемы

Текущая схема датасета **может считаться корректной**, но только в следующем смысле:

### Корректно:

- использовать её как **multi-slice benchmark**;
- требовать хорошего matching `human ↔ llm` **внутри каждого slice**;
- анализировать метрики **по slices**, а не только агрегированно;
- ограничивать claims тем, что реально поддерживает дизайн.

### Некорректно:

- трактовать весь Core как полностью однородный корпус без жанровых различий;
- заявлять, что модель будет различать `human vs llm` независимо от канала, жанра и домена;
- игнорировать то, что `financial_qa`, `email`, `sms` — это разные речевые режимы.

**Вывод:** датасет корректен для задачи ВКР, если в работе он будет интерпретироваться как **составной Core из matched scenario families**, а не как «чистая universal authorship benchmark collection».

---

## 2. Главные принципы, которые считаются зафиксированными

### 2.1. Core должен быть меньше, но чище

Нельзя раздувать Core за счёт:

- broad spam sources,
- unmatched LLM scenarios,
- жанров, для которых нет чистой human-side опоры.

### 2.2. Fraud-side не должен состоять только из одного phishing-email источника

Поэтому Core fraud-side строится не только на `Nazario`, но и расширяется за счёт:

- `SmishTank`
- `Mendeley SMS Phishing dataset` (частично)
- `Nigerian Fraud dataset`

### 2.3. Legitimate-side не должен доминироваться QA

`financial_qa` нужен как контрольный bilateral slice, но не должен быть крупнейшим сегментом Core.

### 2.4. Все правила preprocessing должны применяться симметрично

Особенно:

- URL masking,
- whitespace normalization,
- deduplication logic,
- language filtering.

### 2.5. Claude остаётся только в test

Held-out generator нужен для честной проверки cross-model generalization.

---

### 2.6. Что именно исправляет этот дизайн относительно предыдущих версий

1. Broad spam pools больше не трактуются как узкие fraud classes.
2. Unmatched LLM scenarios удалены из Core.
3. Fraud-email больше не зависит только от Nazario.
4. SMS fraud теперь строится на двух источниках, но по одной единой policy.
5. Legitimate-side anchored on real human data — удалены synthetic-like `bank_notification` и `financial_review`.
6. QA остаётся control slice, а не доминирующим Core-family.
7. Phishing generation больше не сужается до bank-only prompts.
8. Advance-fee scam выделен в отдельный fraud-family.
9. SMS fraud generation строится по реальным human-subtypes.

---

## 3. Финальный состав Core-датасета

## 3.1. Human fraud

### A. `phishing_email`

**Источник:** `Nazario Phishing`  
**Роль:** основной human fraud email anchor

**Статус:** включаем обязательно.

**Комментарий:**  
Nazario подтверждён как coherent phishing-email corpus. Его не нужно дробить на несколько разных core-families, но внутри него полезно хранить subtype-разметку для анализа и генерации.

---

### B. `advance_fee_scam_email`

**Источник:** `Nigerian Fraud dataset`  
**Роль:** второй fraud-email family, отличный от обычного phishing

**Статус:** включаем обязательно.

**Комментарий:**  
Этот источник нужен, чтобы fraud-email часть Core не держалась только на одном `Nazario`. Он расширяет fraud-side за счёт устойчивого scam-family:

- inheritance / estate transfer
- diplomatic package / fund release
- business investment opportunity
- oil contract / large transfer

Важно: этот источник нельзя смешивать с `phishing_email`. Это отдельный `scenario_family`.

---

### C. `fraud_sms_deceptive`

**Источники:**

- `SmishTank`
- `SMS Phishing Dataset for Machine Learning and Pattern Recognition (Mendeley)` — только часть `smishing` subset

**Роль:** human fraud SMS anchor

**Статус:** включаем обязательно, но с фильтрацией категорий.

**Комментарий:**  
SMS fraud-family в Core должен трактоваться не узко как `bank smishing only`, а как **децептивные fraud SMS**, включающие несколько устойчивых подтипов.

---

## 3.2. Human legitimate

### D. `legitimate_email`

**Источники:**

- `Enron ham`
- `SpamAssassin ham`

**Роль:** основной human legitimate email pool

**Статус:** включаем обязательно.

---

### E. `legitimate_sms`

**Источник:** `SMS ham`

**Роль:** основной human legitimate SMS pool

**Статус:** включаем обязательно.

---

### F. `financial_qa`

**Источник:** `HC3 Finance human`

**Роль:** bilateral legitimate control slice

**Статус:** включаем обязательно, но в умеренном объёме.

**Комментарий:**  
Это не fraud-slice, а **контрольный финансовый slice**, который нужен для проверки того, что модель различает `human vs llm` не только на fraud-communications.

---

## 3.3. LLM side

Для Core создаётся LLM-часть для следующих scenario families:

1. `phishing_email`
2. `advance_fee_scam_email`
3. `fraud_sms_deceptive`
4. `legitimate_email`
5. `legitimate_sms`

Для `financial_qa` новый prompt не нужен, так как используется готовый `HC3 Finance chatgpt`.

---

## 3.4. Что не входит в Core

Следующие источники и сценарии в Core **не включаются**:

- `Enron spam`
- `SpamAssassin spam`
- `SMS Spam spam`
- `Enron 419 regex subset`
- `T3 social_engineering`
- `T5 bank_notification`
- `T6 financial_review`
- `HC3 Open QA`
- broad `generic_spam` categories без чистого fraud mapping

Они могут использоваться позже как:

- auxiliary,
- stress-test,
- domain-transfer evaluation,
но не как часть итогового Core.

---

## 4. Финальная онтология Core

## 4.1. Обязательные поля

Каждая запись Core должна содержать:

- `text`
- `label` ∈ `{0,1}`
- `label_str` ∈ `{human,llm}`
- `fraudness` ∈ `{fraud, legitimate}`
- `channel` ∈ `{email, sms, qa}`
- `scenario_family`
- `source_family`
- `dataset_source`
- `time_band`
- `length_bin`
- `origin_model`
- `split`

---

## 4.2. Необязательные диагностические поля

Следующие поля **не являются обязательными** для финальной train/val/test сборки, но могут храниться как diagnostic metadata, если уже были получены:

- `question_id` — для HC3 практически необходим как linkage field, но не как model feature
- `temperature_style`, `temperature_value` — для generated records
- `char_length`, `token_length`
- `phish_subtype` — subtype Nazario
- `scam_subtype` — subtype Nigerian Fraud
- `sms_fraud_subtype` — для SmishTank / Mendeley smishing / generated fraud SMS
- `subject_present`, `html_present`, `body_extraction_status` — для email-источников, полезны для later diagnostics

### Почему subtype-поля не обязательны

- они отсутствуют в исходных источниках;
- они добываются внешней аннотацией;
- они полезны для prompt design и later error analysis;
- но не обязательны для корректной сборки Core и baseline training.

---

## 4.2. Верхнеуровневые scenario families

### Fraud

- `phishing_email`
- `advance_fee_scam_email`
- `fraud_sms_deceptive`

### Legitimate

- `legitimate_email`
- `legitimate_sms`
- `financial_qa`

---

## 4.3. Зачем нужны эти оси


| Ось               | Зачем нужна                                                      | Какой конфаундер убирает          |
| ----------------- | ---------------------------------------------------------------- | --------------------------------- |
| `fraudness`       | human и llm есть и на fraud, и на legitimate стороне             | domain confounding                |
| `channel`         | сравнение идёт внутри email / sms / qa, а не между ними          | genre/channel confounding         |
| `scenario_family` | matched-сценарии держатся согласованно                           | scenario mismatch                 |
| `time_band`       | видна эпоха источника                                            | temporal confounding              |
| `length_bin`      | длина не становится shortcut-признаком                           | length confounding                |
| `source_family`   | можно диагностировать и ограничивать вклад конкретного источника | source artifact confounding       |
| `origin_model`    | можно отдельно оценивать seen vs unseen generators               | generator fingerprint confounding |


---

## 5. Step 1 — формирование human-side датасета

Это этап подготовки Dataset 1: **несгенерированных данных**.

---

## 5.1. Общая рекомендация по preprocessing

Ниже описаны **обязательные элементы**, которые точно не должны быть упущены. Это не исчерпывающая инструкция; pipeline может быть расширен дополнительными шагами, если это улучшает качество данных без нарушения симметрии обработки.

### Обязательно для всех источников

- глобальная дедупликация после нормализации текста;
- language filtering;
- сохранение пунктуации;
- нормализация whitespace;
- фиксация `char_length` и `token_length`;
- assignment `length_bin`;
- assignment `time_band`;
- симметричная URL masking для всех групп.

### Важно

Нельзя применять специальные очистки только к human или только к llm стороне, если это создаёт асимметрию источников.

---

## 5.2. Рекомендации по обработке email-источников

### Источники

- Nazario
- Nigerian Fraud
- Enron ham
- SpamAssassin ham

### Что точно нужно учесть

- удаление RFC 2822 headers;
- аккуратное декодирование quoted-printable / base64 / MIME;
- удаление multipart boundaries;
- очистка HTML-тегов;
- удаление reply-chain мусора и вложенных цитат, если они мешают основной части сообщения;
- нормализация broken encoding;
- извлечение основного text body.

### Важно

- не терять body при слишком агрессивном парсинге;
- не выкидывать punctuation;
- не оставлять source-specific artifacts, если их можно безопасно убрать;
- не считать preprocessing завершённым, пока не сделана ручная sanity-проверка примеров из каждого источника.

### Дополнительная рекомендация

Полезно сохранить:

- `subject_present`
- `html_present`
- `body_extraction_status`
если это поможет later diagnostics, но эти поля не обязательны для Core.

---

## 5.3. Рекомендации по обработке SMS-источников

### Источники

- SmishTank
- Mendeley SMS phishing dataset
- SMS ham

### Что точно нужно учесть

- использовать основной текст сообщения (`MainText` или эквивалент);
- вычищать UI-мусор, если сообщение было собрано из скриншотов или phone-export;
- не терять короткие формы и разговорные сокращения;
- не раскрывать и не нормализовывать вручную разговорные сокращения;
- сохранить SMS-native форму;
- удалить пустые и битые записи;
- отдельно учитывать наличие:
  - URL
  - phone / reply CTA
  - sender type
  - brand mention

### Важно

Если сообщение разворачивается из многопольной структуры, нужно брать именно **основной текст**, а не склеивать в один текст системные поля, timestamp и прочие элементы интерфейса.

---

## 5.4. Рекомендации по обработке QA-источников

### Источник

- HC3 Finance human / chatgpt

### Что точно нужно учесть

- использовать answer text как `text`;
- сохранять `question_id`;
- не смешивать question и answer в одно поле, если это не согласовано с обеими сторонами;
- сохранять pairing между human и chatgpt answer.

### Важно

HC3 должен оставаться отдельным `channel = qa`, а не смешиваться с email/sms.

---

## 5.5. Что делать с Nazario

### Включаем

Да, целиком после очистки и quality filtering.

### Что дополнительно сделать

- split на `legacy` и `modern`;
- сохранить единый `scenario_family = phishing_email`;
- при возможности сохранить диагностический `phish_subtype`.

### Что уже известно по диагностике

Nazario в основном распадается на:

- `account_suspension`
- `suspicious_activity_or_login`
- `invoice_or_payment_lure`
- `kyc_or_identity_update`
- `account_verification`
- `password_reset`
- `refund_or_reward_lure`
- `card_verification_or_card_issue`
- небольшой хвост `nonfinancial_phishing`

### Вывод

Nazario остаётся единым `phishing_email` family, но prompt для него должен покрывать **не только банковские alert-type сценарии**.

---

## 5.6. Что делать с Nigerian Fraud

### Включаем

Да, целиком после очистки и quality filtering.

### `scenario_family`

- `advance_fee_scam_email`

### Что уже известно по диагностике

Внутри доминируют:

- `inheritance_or_estate_transfer`
- `diplomatic_package_or_fund_release`
- `business_investment_opportunity`
- `oil_contract_or_large_transfer`

### Вывод

Это достаточно однородный fraud-email family, который нужно включать как отдельный Core-сценарий, а не смешивать с `phishing_email`.

---

## 5.7. Что делать с SmishTank и Mendeley SMS phishing dataset

### Общий принцип

Для этих двух источников должна применяться **одинаковая policy отбора Core-категорий**.

То есть решение не должно зависеть от имени источника, а должно зависеть от того, входит ли subtype в общий SMS fraud threat model.

### Верхний класс

Для Core используется:

- `scenario_family = fraud_sms_deceptive`

### Подтипы SMS fraud

- `account_alert`
- `delivery_fee_or_service_issue`
- `prize_or_contest_scam`
- `financial_or_crypto_lure`
- `loan_or_credit_lure`
- `generic_deceptive_sms`
- `wrong_number_or_romance_scam`
- `other_or_unclear`

---

## 5.8. Какие SMS fraud-подтипы включать в Core

### Включать точно

- `account_alert`
- `delivery_fee_or_service_issue`
- `prize_or_contest_scam`

### Включать после ручной проверки

- `financial_or_crypto_lure`
- `loan_or_credit_lure`

### Не включать в Core по умолчанию

- `generic_deceptive_sms` — слишком широкий контейнер неясности
- `wrong_number_or_romance_scam` — другой fraud-режим, разговорный и несогласованный с текущим Core
- `other_or_unclear`

### Особая оговорка по `Advertisement`

Категория `Advertisement` сама по себе не означает benign-content.  
Если после ручной проверки конкретные сообщения оказываются deceptive financial lure / scam-like SMS, их можно маппить в:

- `financial_or_crypto_lure`
- `loan_or_credit_lure`
- `prize_or_contest_scam`
и включать в Core по содержанию, а не по исходному имени категории.

---

## 5.9. Что делать с human ham файлами

## A. `legitimate_email`

### Источники

- Enron ham
- SpamAssassin ham

### Что нужно сделать

Не полную переаннотацию всего корпуса, а **диагностический аудит подвыборки**:

- Enron ham: `300–500`
- SpamAssassin ham: `250–350`

### Категории для аудита

- `business_email`
- `personal_email`
- `service_or_system_email`
- `informational_notification_email`
- `mixed_or_unclear`

### Что показала аннотация

- **Enron ham** ≈ в основном `business_email` — рабочая переписка, координация, логистика
- **SpamAssassin ham** — заметно более шумный и смешанный; содержит newsletters, mailing list, personal mail

### Вывод

- **Enron ham** — основной semantic anchor для `legitimate_email`
- **SpamAssassin ham** — дополнительный источник вариативности, но **не главный anchor**; его вклад ограничен

### Зачем это нужно

Не для изменения human-side labels, а для:

- понимания реальной структуры legitimate email,
- правильного проектирования prompt family `legitimate_email`.

---

## B. `legitimate_sms`

### Источник

- SMS ham

### Что нужно сделать

Диагностический аудит подвыборки:

- `250–400` сообщений

### Категории для аудита

- `personal_everyday_sms`
- `coordination_or_logistics_sms`
- `service_notification_sms`
- `transactional_benign_sms`
- `mixed_or_unclear_sms`

### Что показала аннотация

Основная масса — это:

- `personal_everyday_sms` — явно доминирует
- `coordination_or_logistics_sms` — заметная, но меньшая доля

Почти нет evidence, что human-side здесь — это institutional/service SMS.

### Вывод

Prompt `legitimate_sms` должен быть в первую очередь:

- personal,
- informal,
- short,
- everyday.

### Зачем это нужно

Для проектирования `legitimate_sms` prompt.

---

## C. Что важно

Human ham-файлы **включаются как есть после очистки**, а не через полную массовую категоризацию.  
Категоризация нужна только как диагностический инструмент для prompt design.

---

## 6. Step 2 — формирование LLM-side датасета

Это этап подготовки Dataset 2: **сгенерированных данных**.

---

## 6.1. Сколько prompt families нужно

Для итогового Core нужны **5 prompt families**:

1. `phishing_email`
2. `advance_fee_scam_email`
3. `fraud_sms_deceptive`
4. `legitimate_email`
5. `legitimate_sms`

Для `financial_qa` новый prompt не нужен, так как используется готовый `HC3 Finance chatgpt`.

---

## 6.2. Generator split

### Seen generators (train/val)

- один OpenAI-family generator
- один open-weight / non-OpenAI instruction-tuned generator (например, Mistral-family)

### Holdout generator (test only)

- Claude-family

### Жёсткое правило

Claude-generated samples никогда не попадают в train/val.

---

## 6.3. Общие требования к prompt design

Каждый Core prompt должен:

1. быть matched к human anchor;
2. не содержать prompt-level leakage;
3. не быть слишком шаблонным;
4. использовать тот же язык, что и human side;
5. соблюдать channel conventions;
6. использовать реалистичные, но вымышленные entities;
7. передавать `temperature` как API parameter, а не текстовую инструкцию;
8. возвращать plain CSV rows / machine-readable output;
9. не содержать markdown / explanations / assistant chatter;
10. учитывать реальные примеры из **аннотированных human документов**.

### Критически важно

При разработке prompt families обязательно использовать:

- реальные примеры и подтипы из аннотированных human-source diagnostics,
- а не проектировать prompts “из головы”.

Иными словами:

- `phishing_email` prompt должен быть построен с учётом реальных Nazario subtypes;
- `advance_fee_scam_email` — с учётом Nigerian Fraud subtypes;
- `fraud_sms_deceptive` — с учётом реальных SmishTank + Mendeley Core-subtypes;
- `legitimate_email` — с учётом Enron/SpamAssassin ham audit;
- `legitimate_sms` — с учётом SMS ham audit.

---

## 6.4. Prompt family: `phishing_email`

**Human anchor:** Nazario

### Что должно покрываться

Не только bank alert. Prompt family должна покрывать:

- `account_suspension`
- `suspicious_activity_or_login`
- `invoice_or_payment_lure`
- `kyc_or_identity_update`
- `account_verification`
- `password_reset`
- `refund_or_reward_lure`
- `card_verification_or_card_issue`
- ограниченный хвост `nonfinancial_phishing`

### Почему

Nazario не является strictly bank-only corpus. Слишком узкий prompt снова создаст mismatch между human и llm.

---

## 6.5. Prompt family: `advance_fee_scam_email`

**Human anchor:** Nigerian Fraud

### Что должно покрываться

- `inheritance_or_estate_transfer`
- `diplomatic_package_or_fund_release`
- `business_investment_opportunity`
- `oil_contract_or_large_transfer`

### Характерные свойства, которые стоит отражать

- very large sum
- narrative identity / backstory
- confidentiality
- reward / percentage offer
- reply/contact CTA

### Почему

Этот family должен быть отделён от phishing-email и не смешиваться с ним.

---

## 6.6. Prompt family: `fraud_sms_deceptive`

**Human anchors:** SmishTank + Mendeley smishing subset

### Что должно покрываться

- `account_alert`
- `delivery_fee_or_service_issue`
- `prize_or_contest_scam`
- `financial_or_crypto_lure` — только если подтверждён ручной проверкой
- `loan_or_credit_lure` — только если подтверждён ручной проверкой

### Что не должно покрываться

- romance scam conversations
- generic broad spam
- undefined vague messages
- noisy `other`

### Почему

SMS-fraud family должен быть единым и matched across both sources, а не построенным отдельно под каждый датасет.

---

## 6.7. Prompt family: `legitimate_email`

**Human anchors:** Enron ham + SpamAssassin ham

### Что должно покрываться

На основе ham-audit:

- work coordination
- scheduling / logistics
- informational follow-up
- document / file discussion
- administrative communication
- benign service/system-like email

### Важно

Prompt должен в первую очередь отражать **Enron ham distribution** — рабочие письма, координацию, логистику.  
SpamAssassin ham — дополнительный источник вариативности, а не равновесный anchor.  
Нельзя просто переименовать `bank_notification` в `legitimate_email`.

---

## 6.8. Prompt family: `legitimate_sms`

**Human anchor:** SMS ham

### Что должно доминировать

На основе ham-audit:

- personal everyday SMS
- lightweight coordination / logistics

### Чего делать нельзя

- banking notifications не должны становиться ядром family — этого нет в human ham;
- formal service alerts и institutional transaction-like messaging не должны доминировать;
- нельзя превращать этот family в generator of service notifications, если этого не поддерживает human-side.

---

## 6.9. Claude holdout

Для каждой из 5 prompt families нужно сделать **Claude holdout generation**:

- с теми же families,
- с теми же length bins,
- с теми же general constraints,
- только в `test`.

---

## 6.10. Сколько prompt templates получится

### Scenario prompt families

- `5`

### Template structure

Каждая family должна иметь:

- 1 system prompt
- 1 user prompt template

### Итого

- `5 system prompts`
- `5 user prompt templates`

Эти prompt families переиспользуются для:

- seen generator A
- seen generator B
- Claude holdout

---

## 6.11. Length bins

### Email

- `short`
- `medium`
- `long`

### SMS

- `short` only

### QA

No new prompt needed

---

## 7. Корректная оценка размера Core-датасета

## 7.1. Принцип

Нельзя требовать одинаковый размер от всех срезов, если это заставляет:

- использовать слабые источники,
- synthetic overfilling,
- или включать грязные категории.

Приоритет:

1. integrity slice
2. cleanliness
3. match quality
4. size

---

## 7.2. Рекомендуемые target counts по срезам


| Core slice               | Human target | LLM target | Комментарий                       |
| ------------------------ | ------------ | ---------- | --------------------------------- |
| `phishing_email`         | 2500–3000    | 2500–3000  | Nazario, с балансом legacy/modern |
| `advance_fee_scam_email` | 1200–1800    | 1200–1800  | Nigerian Fraud                    |
| `fraud_sms_deceptive`    | 1200–1800    | 1200–1800  | SmishTank + filtered Mendeley     |
| `legitimate_email`       | 2500–3000    | 2500–3000  | Enron ham + SpamAssassin ham      |
| `legitimate_sms`         | 800–1200     | 800–1200   | SMS ham                           |
| `financial_qa`           | 800–1200     | 800–1200   | HC3 Finance, control slice        |


---

## 7.3. Ожидаемый общий размер Core

### Нижняя граница

- около **18,000**

### Рекомендуемый диапазон

- около **19,000–22,000**

### Комфортный верхний диапазон

- около **23,000–24,000**

### Интерпретация

Такой размер достаточно велик для:

- classical ML baseline,
- transformer baseline,
- held-out generator evaluation,
- per-slice analysis,
но при этом не требует возврата broad spam-pools в Core.

---

## 8. Checklist перед запуском baseline

Core считается готовым только если выполнено всё ниже.

### Source validity

- Nazario включён и split by `time_band`
- Nigerian Fraud включён и очищен
- SmishTank включён, очищен и отфильтрован по Core categories
- Mendeley smishing subset включён и отфильтрован по той же policy
- Enron ham очищен
- SpamAssassin ham очищен
- SMS ham очищен
- HC3 Finance paired correctly

### Diagnostic audits

- Enron ham audit complete
- SpamAssassin ham audit complete
- SMS ham audit complete
- Nazario subtype mapping recorded
- Nigerian Fraud subtype mapping recorded
- SmishTank category-to-Core mapping recorded
- Mendeley smishing category-to-Core mapping recorded

### Prompt readiness

- phishing_email prompt finalized
- advance_fee_scam_email prompt finalized
- fraud_sms_deceptive prompt finalized
- legitimate_email prompt finalized
- legitimate_sms prompt finalized

### Generation readiness

- seen-generator outputs produced
- Claude holdout outputs produced
- no prompt leakage
- schema fields complete

### Assembly readiness

- global deduplication complete
- symmetric preprocessing complete
- English-only
- fields assigned
- split integrity checked
- no Claude leakage into train/val

---

## 8.1. Обязательные требования к экспериментальному протоколу

Чтобы схема оставалась корректной, недостаточно одной aggregate metric.

### Обязательно включить в оценку:

- **per-slice metrics** — отдельно по каждому `scenario_family`
- **per-channel metrics** — отдельно email / sms / qa
- **source-family diagnostics** — проверка, что ни один источник не доминирует аномально
- **seen vs unseen generator evaluation** — отдельная строка метрик для Claude holdout
- **sanity-check on held-out Claude** — убедиться, что leakage в train/val отсутствует

Именно это удерживает датасет от ложной интерпретации как «однородного universal benchmark».

---

## 9. Критическая самооценка

### 9.1. Сильные стороны схемы

1. broad spam pools больше не трактуются как узкие fraud classes;
2. unmatched LLM scenarios удалены из Core;
3. fraud-email больше не зависит только от Nazario;
4. SMS fraud теперь строится на двух источниках, но по одной единой policy;
5. legitimate side matched к реальным human distributions;
6. QA остаётся control slice, а не доминирующим Core-family;
7. phishing generation больше не сужается до bank-only prompts;
8. advance-fee scam выделен в отдельный fraud-family;
9. SMS fraud generation строится по реальным human-subtypes;
10. unseen-generator test встроен в протокол.

### 9.2. Неразрешимые ограничения схемы

Эта схема **не устраняет полностью** жанровые различия между:

- fraud email
- legitimate email
- fraud SMS
- legitimate SMS
- financial QA

То есть датасет всё равно остаётся **multi-genre benchmark**, а не perfectly homogeneous corpus.  
Это не ошибка дизайна — это честное ограничение, которое **нужно явно описать в работе**.

### 9.3. Какие риски ещё остаются

1. quality filtering для SmishTank / Mendeley должен быть сделан аккуратно;
2. `Advertisement` и похожие source categories нельзя интерпретировать по названию — только по содержанию;
3. ham audits нужно реально использовать при дизайне prompts;
4. counts должны подгоняться к чистым подмножествам, а не наоборот.

### 9.4. Финальный вердикт

Если цель — получить **идеально жанрово-изолированный** датасет, то текущая схема не идеальна.

Но если цель — получить **максимально корректный и практически реализуемый Core для ВКР**, где:

- убраны главные структурные ошибки,
- slices matched внутри `human ↔ llm`,
- broad spam contamination убрана,
- claims ограничены честно,

то текущая схема является **достаточно корректной и защитимой**.

На текущем этапе этот дизайн можно считать **итоговой спецификацией Core-датасета**, если:

- реально соблюдается единая policy preprocessing,
- новые fraud sources включаются как отдельные matched families,
- LLM prompt design строится на основе уже выполненной human-source диагностики,
- а в работе честно описана multi-genre природа Core.

---

## 10. Definition of done

Итоговый Core-датасет считается собранным корректно, когда:

1. Dataset 1 (human side) подготовлен по этой спецификации;
2. Dataset 2 (generated side) подготовлен по этой спецификации;
3. все обязательные поля присутствуют;
4. quality gates пройдены;
5. train/val/test assembled без leakage;
6. **Claude only in test** — отсутствие Claude-generated samples в train/val подтверждено;
7. baseline можно обучать без изменения source composition и scenario families;
8. **per-slice evaluation включена в экспериментальный протокол** — aggregate-only метрики недостаточны.

