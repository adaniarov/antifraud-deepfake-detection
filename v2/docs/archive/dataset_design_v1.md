> **Archive:** Superseded draft. Not authoritative for Core v2. Current spec: [`dataset_design_final.md`](../dataset_design_final.md).

---

# Спецификация следующей итерации формирования датасета
**Статус:** рабочий документ для финальной итерации дизайна  
**Цель:** собрать максимально корректный датасет для задачи `human vs llm` в антифрод-домене без повторной полной переделки

---

## 1. Главный принцип

Датасет должен строиться не как плоский набор всех найденных источников, а как **иерархическая система**:

1. **Core benchmark** — только те сценарии и источники, где соответствие `human ↔ llm` достаточно надёжно.
2. **Auxiliary slices** — полезные, но менее надёжные сценарии; не должны определять основной claim.
3. **Stress-test only** — сценарии без human-side опоры или с сильной онтологической асимметрией.

---

## 2. Что точно включаем

## 2.1. Источники, которые точно включаются

| Источник | Статус | Роль в датасете | Где используется |
|---|---|---|---|
| **Nazario Phishing** | включаем | основной human fraud email anchor | core |
| **Enron ham** | включаем | основной human legitimate email pool | core |
| **SpamAssassin ham** | включаем | дополнительный human legitimate email pool | core |
| **SMS ham** | включаем | human legitimate SMS pool | core |
| **HC3 Finance human** | включаем | human legitimate financial QA | core (отдельный bilateral slice) |
| **HC3 Finance chatgpt** | включаем | llm legitimate financial QA | core (отдельный bilateral slice) |
| **LLM-generated phishing email** | включаем | matched llm-side к Nazario | core |
| **LLM-generated legitimate email** | нужно сгенерировать | matched llm-side к Enron ham / SA ham | core |
| **LLM-generated legitimate SMS** | нужно сгенерировать | matched llm-side к SMS ham | core |

---

## 2.2. Почему эти источники фиксируются уже сейчас

### Nazario
- самый надёжный human anchor для `phishing_email`
- покрывает именно fraud email
- имеет полезную временную неоднородность (`legacy` / `modern`)

### Enron ham + SpamAssassin ham
- нужны, чтобы legitimate-side имел **email-канал**
- не заменяются HC3, потому что HC3 = `qa`, а не `email`
- используются как benign human email pool

### SMS ham
- нужен, чтобы legitimate-side имел **sms-канал**
- без него SMS-ветка будет односторонней

### HC3 Finance
- нужен как **отдельный bilateral legitimate financial QA slice**
- не заменяет email/sms, а дополняет их

---

## 3. Что пока под вопросом

## 3.1. Источники / подмножества, требующие категоризации или валидации

| Источник | Текущий статус | Возможная роль | Что нужно сделать |
|---|---|---|---|
| **SMS Spam (spam part)** | под вопросом | fraud SMS: core subset + auxiliary spam slices | coarse scenario annotation |
| **Enron spam (all)** | под вопросом | fraud email auxiliary / возможно небольшой core subset | coarse scenario annotation |
| **SpamAssassin spam** | под вопросом | fraud email auxiliary / возможно небольшой core subset | coarse scenario annotation |
| **Enron 419 filter** | под вопросом | candidate subset для `advance_fee / 419` | обязательная валидация, не считать готовым class |
| **Smishtank** (если загрузим) | желателен | основной human fraud SMS anchor для smishing | добавить и использовать как preferred human fraud SMS source |
| **T3 social_engineering** | не core | stress-test only | human-side отсутствует |
| **T5 bank_notification** | не core | stress-test only | нет matched human-side |
| **T6 financial_review** | не core | stress-test only | нет matched human-side |

---

## 3.2. Ключевая позиция по SMS spam и Enron spam

### SMS Spam
Не является автоматически `smishing`.
Это широкий spam-SMS pool.

### Enron spam / SpamAssassin spam
Не являются автоматически `phishing_email`.
Это широкие spam-email pools.

### Enron 419 filter
Это не готовый `scam_419` класс, а только **candidate subset**, отобранный регексом/ключевыми словами.

---

## 4. Финальная структура датасета

## 4.1. Core benchmark

| Канал | Fraud human | Fraud LLM | Legit human | Legit LLM |
|---|---|---|---|---|
| **email** | Nazario (+ возможно подтверждённый subset из Enron/SA spam) | generated `phishing_email` | Enron ham + SA ham | generated `legitimate_email` |
| **sms** | Smishtank (если будет) или подтверждённый fraud-like subset из SMS Spam | generated `fraud_sms_phishing_like` | SMS ham | generated `legitimate_sms` |
| **qa** | HC3 Finance human | HC3 Finance chatgpt | HC3 Finance human | HC3 Finance chatgpt |

### Комментарий по QA
`financial_qa` — это отдельный bilateral legitimate slice, а не замена legitimate email/SMS.

---

## 4.2. Auxiliary slices

| Slice | Источник |
|---|---|
| generic_spam_email | Enron spam / SA spam после coarse annotation |
| generic_spam_sms | SMS Spam после coarse annotation |
| promo_or_marketing_spam | SMS Spam / email spam |
| lottery_or_prize | SMS Spam / email spam |
| candidate_419_email | Enron 419 filter после проверки |

---

## 4.3. Stress-test only

| Сценарий | Источник |
|---|---|
| social_engineering_support | T3 |
| bank_notification | T5 |
| financial_review | T6 |
| open_qa | HC3 Open QA, если понадобится отдельно |

---

## 5. Как категоризировать каждый источник

## 5.1. Nazario
### Нужна ли разметка
Полная scenario-разметка не обязательна.

### Что сделать
- разделить на `legacy` и `modern`
- при необходимости добавить coarse subtag:
  - `phishing_email`
  - `generic_phishing_email`
- основная ось для Nazario — **time_band**, а не тонкая онтология

### Объём
- разметка всех записей не нужна
- нужен audit структуры и временного покрытия

---

## 5.2. Enron ham
### Нужна ли разметка
Полная разметка не нужна.

### Что сделать
Сделать **coarse diagnostic annotation на подвыборке**:
- `business_email`
- `personal_email`
- `service_or_system_email`
- `informational_notification_email`
- `mixed_or_unclear`

### Объём
**300–500** samples достаточно.

### Зачем
- понять внутреннюю структуру legitimate email
- корректно спроектировать `legitimate_email` prompt

---

## 5.3. SpamAssassin ham
### Нужна ли разметка
Полная разметка не нужна.

### Что сделать
Та же coarse diagnostic annotation, что и для Enron ham.

### Объём
**250–350** samples достаточно.

### Зачем
- проверить, насколько этот ham-пул отличается от Enron ham
- не дать LLM-legit prompt переориентироваться только на один style

---

## 5.4. SMS ham
### Нужна ли разметка
Да, но только coarse и на подвыборке.

### Классы
- `personal_everyday_sms`
- `coordination_or_logistics_sms`
- `service_notification_sms`
- `transactional_benign_sms`
- `mixed_or_unclear_sms`

### Объём
**250–400** samples достаточно.

### Зачем
- построить корректный `legitimate_sms` prompt
- понять, насколько SMS ham в целом личные, сервисные или транзакционные

---

## 5.5. SMS Spam (spam part)
### Нужна ли разметка
Да, обязательно.

### Классы
- `bank_or_account_phishing`
- `delivery_or_service_phishing`
- `lottery_or_prize`
- `promo_or_marketing_spam`
- `adult_or_misc_spam`
- `unclear_other`

### Вспомогательные признаки
- `is_deceptive_attack`
- `has_financial_pretence`
- `has_urgency`
- `has_action_request`
- `has_url_or_phone_cta`
- `core_candidate`

### Объём
Так как source маленький, лучше размечать **весь unique spam pool**.

Оценка:
- **642 unique** — размечаем всё

### Зачем
- понять, есть ли вообще валидный human fraud SMS core subset
- решить, нужен ли Smishtank как обязательное добавление
- отделить core vs auxiliary

---

## 5.6. Enron spam (all)
### Нужна ли разметка
Да, но не всего корпуса.

### Классы
- `phishing_email`
- `candidate_419_or_advance_fee`
- `promo_marketing_email`
- `generic_spam_nonphishing`
- `malware_or_attachment_lure`
- `unclear_other`

### Вспомогательные признаки
- `has_financial_pretence`
- `has_credentials_request`
- `has_reward_or_prize`
- `has_confidentiality_appeal`
- `core_candidate`

### Объём
Размечать не всё. Сначала собрать **candidate pool**.

**Рекомендуемый объём разметки:**
- **2500–3000** samples

### Как сократить объём
1. дедупликация
2. prefilter по длине / подозрительным маркерам / CTA / ключевым словам
3. стратифицированный сэмплинг по подкорпусам, длине, наличию URL и keyword hits

---

## 5.7. SpamAssassin spam
### Нужна ли разметка
Да.

### Классы
Та же схема, что и для Enron spam.

### Объём
Можно размечать **весь unique pool**, потому что он manageable.

Оценка:
- **1630 unique** — размечаем всё

### Зачем
- получить более чистый auxiliary spam-email source
- возможно выделить небольшой phishing-like subset

---

## 5.8. Enron 419 filter
### Нужна ли разметка
Да, обязательно.

### Классы
- `confirmed_advance_fee_scam`
- `possible_419_like`
- `not_419`

### Объём
Размечать **весь filtered subset**, так как он небольшой.

Оценка:
- **731 unique**

### Зачем
- решить, пригоден ли этот subset для `T4` pairing
- если чистота низкая, оставить только как candidate / discard

---

## 5.9. HC3 Finance
### Нужна ли разметка
Дополнительная full annotation не нужна.

### Что сделать
- использовать как отдельный `financial_qa`
- можно сделать лёгкий exploratory audit на подвыборке:
  - question style
  - answer type
  - length bins

### Объём
**0–200** samples для аудита, но это не обязательный этап.

---

## 6. Как уменьшить объёмы разметки без потери качества

## 6.1. Общий принцип
Размечать нужно не весь raw corpus, а только те части, которые влияют на итоговый core/auxiliary design.

## 6.2. Что можно не размечать полностью
- весь Nazario
- весь Enron ham
- весь SpamAssassin ham
- весь SMS ham
- HC3 Finance

## 6.3. Что лучше размечать полностью
- весь `SMS Spam spam`
- весь `SpamAssassin spam` unique
- весь `Enron 419 filter`

## 6.4. Что размечать только как candidate pool
- `Enron spam (all)`

## 6.5. Аргументация такого сокращения
1. В финальный датасет всё равно не войдут все raw records.
2. Для core важны matched subsets, а не exhaustiveness.
3. Для ham нужна не fine-grained taxonomy, а понимание типа benign communication.
4. Для больших гетерогенных spam-источников эффективнее сначала сузить candidate pool, чем размечать всё подряд.

---

## 7. Оценка объёмов разметки

## 7.1. Минимальный объём для следующей итерации решений
| Источник | Объём |
|---|---:|
| SMS Spam spam | 642 |
| SpamAssassin spam | 1630 |
| Enron 419 filter | 731 |
| Enron spam candidate pool | 1000–1500 |
| ham audit sample | 500–800 |
| **Итого** | **4500–5300** |

Это хороший объём, чтобы принять следующие архитектурные решения.

## 7.2. Рекомендуемый рабочий объём
| Источник | Объём |
|---|---:|
| SMS Spam spam | 642 |
| SpamAssassin spam | 1630 |
| Enron 419 filter | 731 |
| Enron spam candidate pool | 2500–3000 |
| ham audit sample | 800–1200 |
| **Итого** | **6300–7200** |

Это уже рабочий production-level объём для финализации схемы датасета.

---

## 8. Следующие шаги

## 8.1. Блок A — обязательные ближайшие действия
1. Закончить full coarse annotation для `SMS Spam spam`.
2. Сделать такой же pipeline для `SpamAssassin spam`.
3. Собрать `Enron spam candidate pool`.
4. Провести coarse annotation `Enron 419 filter`.
5. Сделать diagnostic annotation для:
   - Enron ham
   - SpamAssassin ham
   - SMS ham
6. На основе результатов переписать/prompts:
   - `legitimate_email`
   - `legitimate_sms`
   - `fraud_sms_phishing_like`

---

## 8.2. Блок B — аналитика после каждой итерации
После каждой размеченной партии нужно обновлять:

1. распределение классов
2. долю confident / uncertain predictions
3. долю `core_candidate`
4. примеры ошибок по классам
5. длины по классам
6. наличие URL / CTA / urgency по классам

---

## 8.3. Итеративный режим работы
Работать именно в режиме:

1. размечаем одну meaningful часть
2. смотрим распределения
3. читаем 20–50 примеров на класс
4. фиксируем инсайты
5. корректируем:
   - онтологию классов
   - prompts
   - решение core vs auxiliary
6. только потом идём к следующему источнику

### Пример правильного цикла
- размечен SMS Spam
- увидели, что phishing-like subset мал
- вывод: SMS Spam не годится как главный human anchor для smishing
- действие: усиливаем роль Smishtank, а SMS Spam переносим в auxiliary

Именно такой режим и нужно сохранять дальше.

---

## 9. Что должно получиться в конце

К финальной сборке у тебя должен быть документирован такой список:

### Core
- Nazario → `phishing_email`
- Enron ham + SA ham → `legitimate_email`
- SMS ham → `legitimate_sms`
- HC3 Finance → `financial_qa`
- Smishtank или fraud-like SMS subset → `fraud_sms_phishing_like`
- matched LLM generations for all core slices

### Auxiliary
- generic spam email
- promo/lottery spam
- candidate 419 subset (если подтвердится частично)
- broader SMS spam categories

### Stress-test only
- social engineering support
- bank notification
- financial review
- open QA (optional)

---

## 10. Финальный практический принцип

**Лучше принять несколько жёстких ограничений и сделать чистый core benchmark, чем тянуть в ядро сценарии, для которых human-side пока не подтверждён.**
