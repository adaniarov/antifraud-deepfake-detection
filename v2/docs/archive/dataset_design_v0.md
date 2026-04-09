> **Archive:** Superseded draft. Not authoritative for Core v2. Current spec: [`dataset_design_final.md`](../dataset_design_final.md).

---

# Предложение по финальному дизайну датасета для ВКР
**Статус:** проектное решение для последней итерации сборки  
**Основание:** `dataset_design.md`, `dataset_description_v4.docx`, текущие промпты T1–T6

---

## Легенда статусов

- <span style="background-color:#d4edda;padding:2px 6px;border-radius:4px;">ЗЕЛЁНЫЙ — решение можно фиксировать</span>
- <span style="background-color:#fff3cd;padding:2px 6px;border-radius:4px;">ЖЁЛТЫЙ — решение зависит от дополнительной проверки / silver-разметки</span>
- <span style="background-color:#f8d7da;padding:2px 6px;border-radius:4px;">КРИТИЧЕСКОЕ — если не исправить, датасет будет методологически слабым</span>

---

## 1. Исполнительное резюме

### 1.1. Главный вывод
<span style="background-color:#d4edda;padding:2px 6px;border-radius:4px;">ЗЕЛЁНЫЙ</span>  
Корректный датасет **можно** собрать из уже имеющихся источников, но **не** в исходной версии, где все текущие spam-источники автоматически трактуются как узкие fraud-сценарии, а все T1–T6 считаются равноправными core-типами.

### 1.2. Что нужно изменить обязательно
<span style="background-color:#f8d7da;padding:2px 6px;border-radius:4px;">КРИТИЧЕСКОЕ</span>

1. Нельзя трактовать `SMS Spam spam` как прямой human-аналог `smishing`.
2. Нельзя трактовать весь `Enron spam` и весь `SpamAssassin spam` как прямой human-аналог `phishing_email`.
3. Нельзя оставлять `T3 social_engineering`, `T5 bank_notification`, `T6 financial_review` в **core** без human-side опоры.
4. Нельзя смешивать legacy human email-источники и modern LLM generation без явного поля `time_band`.
5. Нельзя строить основной benchmark по плоской оси `content_type`, если human- и llm-ветки покрывают её асимметрично.

### 1.3. Итоговая стратегия
<span style="background-color:#d4edda;padding:2px 6px;border-radius:4px;">ЗЕЛЁНЫЙ</span>  

Финальный датасет должен состоять из:

1. **Core benchmark** — только matched или acceptably matched сценарии.
2. **Auxiliary slices** — полезные, но менее надёжные сценарии.
3. **Stress-test only** — сценарии без human-side опоры или с сильной онтологической асимметрией.

---

## 2. Главные уязвимости текущего набора

### 2.1. Временной конфаундинг
<span style="background-color:#f8d7da;padding:2px 6px;border-radius:4px;">КРИТИЧЕСКОЕ</span>  

Human email-источники в основном старые: Enron ~2000–2002, SpamAssassin 2003–2005, значительная часть Nazario — 2005–2007. При этом LLM-тексты планируются как 2024–2025. Без контроля модель легко учит эпоху, а не authorship.

**Решение**
- Ввести поле `time_band`: `legacy` / `modern`.
- Разделить Nazario минимум на `legacy` и `modern`.
- В основном fraud-email срезе повысить вес modern Nazario относительно legacy email corpora.

### 2.2. Scenario mismatch внутри spam-источников
<span style="background-color:#f8d7da;padding:2px 6px;border-radius:4px;">КРИТИЧЕСКОЕ</span>  

Общий spam-класс шире, чем сценарии в prompts:
- `SMS Spam spam` содержит не только bank/account smishing.
- `Enron spam` и `SpamAssassin spam` содержат не только phishing emails.
- `Enron 419 filter` — silver subset по keyword filter, а не gold-standard corpus.

**Решение**
- Выполнить coarse scenario annotation spam-источников.
- Использовать в core только matched subsets.
- Остальной spam переносить в auxiliary slices.

### 2.3. Отсутствие human-side для части LLM-сценариев
<span style="background-color:#f8d7da;padding:2px 6px;border-radius:4px;">КРИТИЧЕСКОЕ</span>  

Сейчас `social_engineering`, `bank_notification`, `financial_review` не имеют чистой human-side пары среди уже собранных источников.

**Решение**
- Не использовать их в core.
- Оставить как stress-test only или exploratory data.

### 2.4. Канальный и жанровый конфаундинг
<span style="background-color:#fff3cd;padding:2px 6px;border-radius:4px;">ЖЁЛТЫЙ</span>  

Если один сценарий живёт только в одном канале или только в одном происхождении, модель может учить канал, а не authorship.

**Решение**
- Базовая ось стратификации должна идти через `channel`.
- `chat` сейчас не готов к core из-за отсутствия human-side.

---

## 3. Итоговая схема стратификации

## 3.1. Обязательные поля датасета

| Поле | Значения | Статус | Какую уязвимость контролирует |
|---|---|---|---|
| `label` | `human`, `llm` | <span style="background-color:#d4edda;padding:2px 6px;border-radius:4px;">зелёный</span> | основная целевая переменная |
| `fraudness` | `fraud`, `legitimate` | <span style="background-color:#d4edda;padding:2px 6px;border-radius:4px;">зелёный</span> | доменный конфаундинг |
| `channel` | `email`, `sms`, `qa` | <span style="background-color:#d4edda;padding:2px 6px;border-radius:4px;">зелёный</span> | жанровый/канальный конфаундинг |
| `scenario_family` | см. ниже | <span style="background-color:#d4edda;padding:2px 6px;border-radius:4px;">зелёный</span> | scenario mismatch |
| `time_band` | `legacy`, `modern` | <span style="background-color:#d4edda;padding:2px 6px;border-radius:4px;">зелёный</span> | temporal confounding |
| `source_family` | `nazario`, `sms_spam`, `enron`, `spamassassin`, `hc3_finance`, `gpt_gen`, `mistral_gen`, `claude_gen` | <span style="background-color:#d4edda;padding:2px 6px;border-radius:4px;">зелёный</span> | source dominance / diagnostics |
| `length_bin` | `short`, `medium`, `long` | <span style="background-color:#d4edda;padding:2px 6px;border-radius:4px;">зелёный</span> | length confounding |
| `origin_model` | `human`, `gpt-*`, `mistral-*`, `claude-*`, `chatgpt_hc3` | <span style="background-color:#d4edda;padding:2px 6px;border-radius:4px;">зелёный</span> | generator-specific overfit |
| `split` | `train`, `val`, `test`, `test_unseen_model`, `test_robustness` | <span style="background-color:#d4edda;padding:2px 6px;border-radius:4px;">зелёный</span> | leakage / unfair evaluation |

---

## 3.2. Итоговая онтология `scenario_family`

### Core / semi-core families
| scenario_family | channel | fraudness | Статус | Комментарий |
|---|---|---|---|---|
| `phishing_email` | email | fraud | <span style="background-color:#d4edda;padding:2px 6px;border-radius:4px;">зелёный</span> | основа через Nazario + matched LLM prompt |
| `fraud_sms_phishing_like` | sms | fraud | <span style="background-color:#fff3cd;padding:2px 6px;border-radius:4px;">жёлтый</span> | только после выделения subset из SMS Spam |
| `scam_419` | email | fraud | <span style="background-color:#fff3cd;padding:2px 6px;border-radius:4px;">жёлтый</span> | Enron 419 filter — silver subset |
| `legitimate_email` | email | legitimate | <span style="background-color:#d4edda;padding:2px 6px;border-radius:4px;">зелёный</span> | Enron ham + SpamAssassin ham; нужен новый LLM prompt |
| `legitimate_sms` | sms | legitimate | <span style="background-color:#d4edda;padding:2px 6px;border-radius:4px;">зелёный</span> | SMS ham; нужен новый LLM prompt |
| `financial_qa` | qa | legitimate | <span style="background-color:#d4edda;padding:2px 6px;border-radius:4px;">зелёный</span> | нативная bilateral пара HC3 |

### Stress-test only families
| scenario_family | channel | fraudness | Статус | Почему не core |
|---|---|---|---|---|
| `social_engineering_support` | chat/email-thread | fraud | <span style="background-color:#fff3cd;padding:2px 6px;border-radius:4px;">жёлтый</span> | human-side absent |
| `bank_notification` | email/sms | legitimate | <span style="background-color:#fff3cd;padding:2px 6px;border-radius:4px;">жёлтый</span> | human-side absent |
| `financial_review` | review | legitimate | <span style="background-color:#fff3cd;padding:2px 6px;border-radius:4px;">жёлтый</span> | human-side absent |
| `open_qa` | qa | unclear legit | <span style="background-color:#fff3cd;padding:2px 6px;border-radius:4px;">жёлтый</span> | domain TBD + asymmetric pair |

---

## 4. Как стратификация закрывает известные уязвимости

| Уязвимость | Как закрывается |
|---|---|
| Human fraud vs LLM legit confusion | поле `fraudness` и 2×2-дизайн |
| Канал вместо authorship | поле `channel`; core только по matched channels |
| Узкий LLM prompt vs широкий spam source | поле `scenario_family` + coarse annotation spam-источников |
| Legacy human vs modern LLM | поле `time_band`; weighted use of modern Nazario |
| Перекос по длине | `length_bin` |
| Перекос по конкретному корпусу | `source_family` + source caps |
| Overfit на конкретный генератор | `origin_model` + Claude test-only |
| URL leakage | симметричная URL masking |
| Дубликаты между источниками | global deduplication |

---

## 5. Что делать со spam-датасетами

## 5.1. Общий принцип
<span style="background-color:#d4edda;padding:2px 6px;border-radius:4px;">ЗЕЛЁНЫЙ</span>  

Spam-датасеты **не использовать как монолитные классы**.  
Их нужно разложить хотя бы на coarse fraud families.

---

## 5.2. Что делать с `SMS Spam`
<span style="background-color:#f8d7da;padding:2px 6px;border-radius:4px;">КРИТИЧЕСКОЕ</span>  

Нельзя считать весь `SMS spam` эквивалентом `smishing`.

### Рекомендуемая coarse-разметка
| Класс | Идёт в core? | Комментарий |
|---|---:|---|
| `bank_or_account_phishing` | да | лучший matched human subset для T2-подобного fraud SMS |
| `delivery_or_service_phishing` | да / условно | можно включать в core, если LLM prompt будет расширен до phishing-like SMS family |
| `lottery_or_prize` | нет | auxiliary only |
| `promo_or_marketing_spam` | нет | auxiliary only |
| `adult_or_misc_spam` | нет | auxiliary only |
| `unclear_other` | нет | discard or auxiliary |

### Практическое решение по разметке
<span style="background-color:#fff3cd;padding:2px 6px;border-radius:4px;">ЖЁЛТЫЙ</span>  

Полная ручная разметка, вероятно, не нужна. Реалистичный путь:
1. LLM-assisted annotation по фиксированной онтологии.
2. Обязательная ручная проверка небольшой контрольной выборки.
3. В core брать только классы с приемлемой чистотой по аудиту.

### Минимальный audit
- 50–100 примеров на каждый крупный автоматически размеченный класс.
- Если класс даёт слишком много ошибок — не брать его в core.

---

## 5.3. Что делать с `Enron spam` и `SpamAssassin spam`
<span style="background-color:#f8d7da;padding:2px 6px;border-radius:4px;">КРИТИЧЕСКОЕ</span>  

Нельзя считать весь `Enron spam` и весь `SpamAssassin spam` эквивалентом `phishing_email`.

### Рекомендуемая coarse-разметка
| Класс | Идёт в core? | Комментарий |
|---|---:|---|
| `phishing_email` | да | matched subset для T1 |
| `scam_419` | условно | semi-core / auxiliary, если silver subset чистый |
| `generic_spam_nonphishing` | нет | auxiliary only |
| `promo_marketing_email` | нет | auxiliary only |
| `malware_or_attachment_lure` | нет | auxiliary only |
| `unclear_other` | нет | discard or auxiliary |

### Практическое решение
<span style="background-color:#fff3cd;padding:2px 6px;border-radius:4px;">ЖЁЛТЫЙ</span>  

Для email spam тоже нужна LLM-assisted coarse annotation.  
Но здесь есть упрощение:
- **Nazario** уже можно считать основным human source для `phishing_email`;
- значит задача Enron/SA — не “дать весь fraud email”, а **дополнить Nazario** и выделить 419/generic spam slices.

### Итог
- `Nazario` = основной human `phishing_email`
- `Enron/SA spam` = дополнительные email-fraud источники **только после coarse-разметки**
- `Enron 419 filter` = silver subset, использовать осторожно

---

## 5.4. Что делать с `Nazario`
<span style="background-color:#d4edda;padding:2px 6px;border-radius:4px;">ЗЕЛЁНЫЙ</span>  

Nazario — главный human source для `phishing_email`.

### Что зафиксировать
1. Разделить на `legacy` и `modern`.
2. Не смешивать их без учёта `time_band`.
3. В fraud-email core приоритетно использовать `modern Nazario`.
4. `legacy Nazario` оставлять как дополнительный источник, но не доминирующий.

---

## 5.5. Что делать с `HC3 Finance`
<span style="background-color:#d4edda;padding:2px 6px;border-radius:4px;">ЗЕЛЁНЫЙ</span>  

HC3 Finance — отдельный strong auxiliary/core slice для `financial_qa`.

### Правильный статус
- legitimate only
- native bilateral pair
- не смешивать интерпретационно с email/sms fraud slices

---

## 6. Новая логика prompts

## 6.1. Что оставить
| Prompt | Решение | Статус |
|---|---|---|
| `T1_phishing_email` | оставить как core fraud-email prompt | <span style="background-color:#d4edda;padding:2px 6px;border-radius:4px;">зелёный</span> |
| `T4_scam_419` | оставить как semi-core prompt | <span style="background-color:#d4edda;padding:2px 6px;border-radius:4px;">зелёный</span> |

## 6.2. Что изменить
| Prompt | Что не так | Что делать | Статус |
|---|---|---|---|
| `T2_smishing` | human-side source шире, чем bank/account smishing | либо сузить human SMS subset, либо расширить T2 до `fraud_sms_phishing_like` family | <span style="background-color:#f8d7da;padding:2px 6px;border-radius:4px;">критическое</span> |
| `T5_bank_notification` | нет human-side пары | убрать из core; заменить новым prompt `legitimate_email` | <span style="background-color:#f8d7da;padding:2px 6px;border-radius:4px;">критическое</span> |
| `T6_financial_review` | нет human-side пары | убрать из core; заменить новым prompt `legitimate_sms` или оставить как stress-test only | <span style="background-color:#f8d7da;padding:2px 6px;border-radius:4px;">критическое</span> |
| `T3_social_engineering` | нет human-side пары | оставить только как stress-test | <span style="background-color:#d4edda;padding:2px 6px;border-radius:4px;">зелёный</span> |

---

## 6.3. Какие новые prompts нужны

### Новый prompt A — `legitimate_email`
<span style="background-color:#d4edda;padding:2px 6px;border-radius:4px;">ЗЕЛЁНЫЙ</span>  

Нужен вместо `T5 bank_notification`.

**Назначение**
- matched LLM-side для `Enron ham + SpamAssassin ham`

**Подтемы**
- ordinary work coordination
- scheduling / logistics
- informational follow-up
- customer-service-like benign communication
- document / statement availability without threat cues
- internal administrative email

**Ключевой принцип**
Не делать его слишком “банковским”, если human-side не финансовый.  
Это должен быть **обычный legitimate email**, а не “идеальная банковская нотификация”.

---

### Новый prompt B — `legitimate_sms`
<span style="background-color:#d4edda;padding:2px 6px;border-radius:4px;">ЗЕЛЁНЫЙ</span>  

Нужен вместо `T6 financial_review` как core-legit prompt для SMS-канала.

**Назначение**
- matched LLM-side для `SMS ham`

**Подтемы**
- everyday coordination
- reminders
- friendly check-ins
- simple service notifications
- logistics / arrival / timing

**Ключевой принцип**
Не делать его официальным банковским уведомлением, если human-side этого не содержит.

---

### Новый prompt C — `fraud_sms_phishing_like`
<span style="background-color:#fff3cd;padding:2px 6px;border-radius:4px;">ЖЁЛТЫЙ</span>  

Это переработка `T2`.

**Назначение**
- matched LLM-side для размеченного fraud subset из `SMS Spam spam`

**Подтемы**
- account/security alert
- delivery/service issue requiring action
- refund/verification
- limited subset `lottery/prize`, если будет оставлен в core (скорее нет)

**Ключевой принцип**
Онтология T2 должна совпадать с coarse-разметкой human SMS fraud.

---

## 7. Финальный состав датасета

## 7.1. Core benchmark
| Канал | Human fraud | LLM fraud | Human legit | LLM legit | Статус |
|---|---|---|---|---|---|
| `email` | Nazario `phishing_email` + часть размеченного Enron/SA spam | `T1_phishing_email` | Enron ham + SA ham | `legitimate_email` | <span style="background-color:#d4edda;padding:2px 6px;border-radius:4px;">зелёный</span> |
| `sms` | размеченный subset из SMS spam | `fraud_sms_phishing_like` | SMS ham | `legitimate_sms` | <span style="background-color:#fff3cd;padding:2px 6px;border-radius:4px;">жёлтый</span> |
| `qa` | HC3 Finance human | HC3 Finance chatgpt | HC3 Finance human | HC3 Finance chatgpt | <span style="background-color:#d4edda;padding:2px 6px;border-radius:4px;">зелёный</span> |

### Комментарий
QA technically не образует отдельный fraud/legit канал. Его лучше трактовать как отдельный bilateral legitimate sub-benchmark, а не как замену email/sms core.

---

## 7.2. Auxiliary slices
| Slice | Источник | Статус |
|---|---|---|
| `scam_419` | Enron 419 filter + `T4_scam_419` | <span style="background-color:#fff3cd;padding:2px 6px;border-radius:4px;">жёлтый</span> |
| `generic_spam_sms` | остаток SMS spam after annotation | <span style="background-color:#fff3cd;padding:2px 6px;border-radius:4px;">жёлтый</span> |
| `generic_spam_email` | остаток Enron/SA spam after annotation | <span style="background-color:#fff3cd;padding:2px 6px;border-radius:4px;">жёлтый</span> |

---

## 7.3. Stress-test only
| Slice | Источник | Статус |
|---|---|---|
| `social_engineering_support` | T3 | <span style="background-color:#d4edda;padding:2px 6px;border-radius:4px;">зелёный</span> |
| `bank_notification` | T5 | <span style="background-color:#d4edda;padding:2px 6px;border-radius:4px;">зелёный</span> |
| `financial_review` | T6 | <span style="background-color:#d4edda;padding:2px 6px;border-radius:4px;">зелёный</span> |

---

## 8. Оценка размеров и целевой объём

## 8.1. Что известно точно по уже собранным данным
| Источник | Что известно |
|---|---|
| Nazario | 8,472 raw; 5,820 unique |
| SMS spam | 747 raw; 642 unique |
| SMS ham | 4,827 raw; 4,518 unique |
| Enron ham | 16,545 raw; 15,794 unique |
| SpamAssassin ham | 3,900 raw; 3,861 unique |
| HC3 Finance human | 3,933 raw/unique |
| HC3 Finance chatgpt | 4,503 raw; 4,462 unique |
| Enron 419 filter | 872 raw; 731 unique |
| SpamAssassin spam | 3,293 raw; 1,630 unique |

## 8.2. Что нельзя оценить точно сейчас
<span style="background-color:#fff3cd;padding:2px 6px;border-radius:4px;">ЖЁЛТЫЙ</span>  

Нельзя честно назвать точные размеры core SMS fraud и core phishing subset из Enron/SA, пока не сделана coarse annotation spam-источников.

## 8.3. Реалистичный целевой размер финального датасета
### Вариант A — консервативный и самый безопасный
<span style="background-color:#d4edda;padding:2px 6px;border-radius:4px;">ЗЕЛЁНЫЙ</span>

- `phishing_email` human: использовать весь доступный качественный Nazario subset после quality filtering; LLM — matching amount
- `legitimate_email` human: cap из Enron ham + SA ham; LLM — matching amount
- `financial_qa` human/llm: cap по 3,000–3,900 на сторону
- `scam_419`: до 731 human unique и matching LLM
- `sms` channel: размер определяется только после coarse annotation

### Практически безопасная цель по total size
- **минимум**: ~10k–14k записей в финальном core+semi-core корпусе
- **комфортная цель**: ~14k–18k
- **выше 18k** идти только если spam annotation даст чистые matched subsets

### Почему не стоит форсировать 21k любой ценой
<span style="background-color:#f8d7da;padding:2px 6px;border-radius:4px;">КРИТИЧЕСКОЕ</span>  

Если добивать размер до заранее придуманного таргета за счёт онтологически слабых сценариев и плохих соответствий, качество датасета упадёт сильнее, чем выигрыш от объёма.

---

## 9. Решение по spam-источникам

## 9.1. Рекомендуемое решение
<span style="background-color:#d4edda;padding:2px 6px;border-radius:4px;">ЗЕЛЁНЫЙ</span>

1. **Не выбрасывать spam-источники.**
2. **Не использовать их как монолитные content types.**
3. **Сделать LLM-assisted coarse annotation** по фиксированной онтологии.
4. **Провести ручной audit малой выборки**.
5. После этого:
   - matched subsets → в core
   - сомнительные / широкие классы → auxiliary
   - мусор / unclear → discard

## 9.2. Когда это решение не сработает
<span style="background-color:#fff3cd;padding:2px 6px;border-radius:4px;">ЖЁЛТЫЙ</span>  

Если после LLM-assisted annotation:
- классы окажутся слишком маленькими,
- или их чистота окажется низкой,
тогда spam-источники нужно будет использовать только как auxiliary slices, а core сосредоточить на Nazario + ham + HC3 + matching LLM.

---

## 10. Что обязательно сделать до финальной сборки

### Блок A — без этого датасет нельзя фиксировать
<span style="background-color:#f8d7da;padding:2px 6px;border-radius:4px;">КРИТИЧЕСКОЕ</span>

- [ ] Ввести `time_band`
- [ ] Разделить Nazario на `legacy` / `modern`
- [ ] Сделать global deduplication across all sources
- [ ] Убрать `T3/T5/T6` из core
- [ ] Переписать prompt-ontology для SMS и legit-сценариев
- [ ] Сделать coarse scenario annotation для `SMS spam`
- [ ] Сделать coarse scenario annotation для `Enron/SA spam`

### Блок B — желательно сделать
<span style="background-color:#fff3cd;padding:2px 6px;border-radius:4px;">ЖЁЛТЫЙ</span>

- [ ] Провести небольшой ручной audit классов после LLM-assisted annotation
- [ ] Ввести source caps до финальной балансировки
- [ ] Отдельно собрать test_unseen_model и test_robustness
- [ ] Считать per-slice metrics, а не только общий F1

---

## 11. Окончательное решение

### Что можно фиксировать уже сейчас
<span style="background-color:#d4edda;padding:2px 6px;border-radius:4px;">ЗЕЛЁНЫЙ</span>

- Основа датасета — не плоский T1–T6, а matched stratified design.
- Core держать на `email`, `sms`, `qa`.
- `Nazario` — главный human fraud email source.
- `HC3 Finance` — отдельный bilateral legitimate QA slice.
- `T3`, `T5`, `T6` — не core.
- Нужны новые prompts: `legitimate_email`, `legitimate_sms`, переработанный `fraud_sms_phishing_like`.
- Нужен `time_band`.

### Что зависит от ближайшей проверки
<span style="background-color:#fff3cd;padding:2px 6px;border-radius:4px;">ЖЁЛТЫЙ</span>

- Сколько именно SMS spam пойдёт в core после coarse annotation.
- Сколько именно Enron/SA spam даст чистого phishing-like subset.
- Останется ли `scam_419` semi-core или уйдёт в auxiliary.
- Какой будет точный финальный размер корпуса.

### Главный практический принцип
<span style="background-color:#d4edda;padding:2px 6px;border-radius:4px;">ЗЕЛЁНЫЙ</span>

**Лучше меньший, но онтологически корректный датасет, чем большой корпус со скрытым scenario mismatch.**
