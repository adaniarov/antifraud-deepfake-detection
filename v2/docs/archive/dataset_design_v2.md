> **Archive:** Superseded draft. Not authoritative for Core v2. Current spec: [`dataset_design_final.md`](../dataset_design_final.md).

---

# Dataset Design v2 — Core Dataset Specification

**Purpose:** final specification for the **Core dataset** used to train and evaluate baseline models for `human-written` vs `LLM-generated` text detection in anti-fraud contexts.  
**Scope:** only the **Core** dataset. Auxiliary and stress-test slices are intentionally excluded from this specification.  
**Status:** design freeze candidate.

---

## 1. Executive decision

The final **Core** dataset will contain only source families that are **matched strongly enough** across:
- `label` (`human` / `llm`)
- `fraudness` (`fraud` / `legitimate`)
- `channel` (`email` / `sms` / `qa`)
- `scenario_family`
- `time_band`
- `length_bin`

The Core dataset will include:

### Human side
1. **Nazario Phishing** → `fraud / email / phishing_email`
2. **Smishtank** → `fraud / sms / fraud_sms_phishing_like`
3. **Enron ham + SpamAssassin ham** → `legitimate / email / legitimate_email`
4. **SMS ham** → `legitimate / sms / legitimate_sms`
5. **HC3 Finance human** → `legitimate / qa / financial_qa`

### LLM side
6. **Generated phishing_email** → matched to Nazario
7. **Generated fraud_sms_phishing_like** → matched to Smishtank
8. **Generated legitimate_email** → matched to Enron ham + SpamAssassin ham
9. **Generated legitimate_sms** → matched to SMS ham
10. **HC3 Finance chatgpt** → matched to HC3 Finance human
11. **Claude holdout generations** for the 4 generated scenario families → `test-only`

### Explicitly excluded from Core
- Enron spam
- SpamAssassin spam
- SMS Spam spam
- Enron 419 regex subset
- T3 `social_engineering`
- T5 `bank_notification`
- T6 `financial_review`
- HC3 Open QA

These may still be used later as auxiliary or stress-test data, but **not in Core**.

---

## 2. Why this design is the correct correction of v1/v2/v4 mistakes

The previous design directions had several structural risks. Dataset Design v3 removes them directly.

### 2.1. Problem: broad spam pools were treated as narrow fraud scenarios
Examples:
- `SMS Spam spam` was too broad to serve as a clean human anchor for `smishing`
- `Enron spam` / `SpamAssassin spam` were too broad to serve as clean human anchors for `phishing_email`
- `Enron 419 filter` was only a keyword-selected candidate subset, not a confirmed scam corpus

**Fix in v3:** none of these sources are used in Core.

### 2.2. Problem: some LLM scenarios had no human-side anchor
Examples:
- `social_engineering`
- `bank_notification`
- `financial_review`

**Fix in v3:** these scenarios are removed from Core.

### 2.3. Problem: legitimate side was not matched to actual human legitimate text distributions
Examples:
- `T5 bank_notification` is too formal and transactional to match Enron/SpamAssassin ham
- `T6 financial_review` has no open human-side counterpart in the current collected Core

**Fix in v3:** two new prompts are introduced:
- `legitimate_email`
- `legitimate_sms`

### 2.4. Problem: temporal confounding
Human sources span very different eras:
- Enron ~2000–2002
- SpamAssassin ~2003–2005
- Nazario includes both legacy and modern material
- HC3 human ~2008–2022
- planned LLM generations are current-era

**Fix in v3:** `time_band` becomes a required field in Core and Nazario is explicitly split into `legacy` and `modern`.

---

## 3. Final Core ontology

## 3.1. Required fields

Every Core record must contain the following fields:

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

Optional:
- `question_id` for HC3
- `temperature_style`, `temperature_value` for generated texts
- `char_length`, `token_length`

---

## 3.2. Core scenario families

### Fraud
- `phishing_email`
- `fraud_sms_phishing_like`

### Legitimate
- `legitimate_email`
- `legitimate_sms`
- `financial_qa`

---

## 3.3. Why these axes are required

| Axis | Why it is required | Confounder it removes |
|---|---|---|
| `label` | target variable | not a control axis; the prediction target |
| `fraudness` | forces human and llm to exist on both fraud and legitimate sides | domain confounding (`fraud` vs `llm`) |
| `channel` | forces comparison within email / sms / qa rather than across them | genre/channel confounding |
| `scenario_family` | keeps matched scenarios aligned within each channel | scenario mismatch |
| `time_band` | makes age/era visible in the data | temporal confounding |
| `length_bin` | prevents models from overusing length | length confounding |
| `source_family` | allows caps and diagnostics by source | source-specific artifact confounding |
| `origin_model` | allows seen-vs-unseen generator evaluation | generator fingerprint confounding |

---

## 4. Step 1 — build Dataset 1 (human-written source side)

This step creates the **non-generated** half of the Core dataset.

---

## 4.1. Include source: Nazario Phishing

**Role:** `human / fraud / email / phishing_email`

### Use
- Include Nazario after cleaning, deduplication, and language filtering
- Keep only English texts
- Strip headers / MIME / HTML the same way as for all email sources

### Additional requirement
Split Nazario into:
- `time_band = legacy`
- `time_band = modern`

### Why included
Nazario is the strongest open human fraud-email anchor currently available in the collected data.

### What not to do
- Do **not** merge Nazario with Enron spam or SpamAssassin spam and treat them as one fraud-email pool
- Do **not** ignore the temporal split

---

## 4.2. Include source: Smishtank

**Role:** `human / fraud / sms / fraud_sms_phishing_like`

### Use
- Add Smishtank as the main human fraud SMS source
- Keep only English samples
- Deduplicate aggressively
- Remove broken messages / artifacts / empty texts
- Preserve the original sms-style brevity

### Why included
The existing `SMS Spam spam` source is too broad and does not provide a clean enough phishing-like SMS anchor. Smishtank resolves that mismatch.

### What not to do
- Do **not** substitute Smishtank with the full `SMS Spam spam` pool in Core
- Do **not** broaden the scenario to “all SMS spam” just to increase volume

---

## 4.3. Include sources: Enron ham + SpamAssassin ham

**Role:** `human / legitimate / email / legitimate_email`

### Use
- Include both sources after cleaning, deduplication, language filtering
- Strip email headers / MIME / HTML
- Preserve punctuation
- Apply symmetric URL masking downstream

### Diagnostic annotation required
A **small audit sample only**, not full-labeling.

#### Audit size
- Enron ham: `300–500` samples
- SpamAssassin ham: `250–350` samples

#### Audit labels
- `business_email`
- `personal_email`
- `service_or_system_email`
- `informational_notification_email`
- `mixed_or_unclear`

### Why audit is required
Not to relabel the dataset, but to design a correct `legitimate_email` generation prompt.

### What not to do
- Do **not** match Enron ham / SpamAssassin ham with `bank_notification`
- Do **not** fully annotate the whole ham corpora

---

## 4.4. Include source: SMS ham

**Role:** `human / legitimate / sms / legitimate_sms`

### Use
- Include after deduplication and language filtering
- Keep brevity and everyday SMS form

### Diagnostic annotation required
A **small audit sample only**.

#### Audit size
- `250–400` samples

#### Audit labels
- `personal_everyday_sms`
- `coordination_or_logistics_sms`
- `service_notification_sms`
- `transactional_benign_sms`
- `mixed_or_unclear_sms`

### Why audit is required
To design a correct `legitimate_sms` prompt.

### What not to do
- Do **not** treat `SMS ham` as transaction-alert or bank-notification data by default

---

## 4.5. Include source: HC3 Finance human

**Role:** `human / legitimate / qa / financial_qa`

### Use
- Include the human answers
- Preserve `question_id`
- Keep the question-answer pairing metadata

### Why included
HC3 Finance is the cleanest bilateral human↔LLM financial control slice available.

### What not to do
- Do **not** merge HC3 into email or sms channels
- Do **not** reinterpret it as fraud content

---

## 4.6. Exclude these from Dataset 1 Core

- Enron spam
- SpamAssassin spam
- SMS Spam spam
- Enron 419 regex subset
- Yelp
- HC3 Open QA
- any source without a clean Core role

---

## 4.7. Human-side preprocessing specification

### Common preprocessing for all text
- normalize whitespace
- preserve punctuation
- deduplicate globally
- filter non-English
- record `char_length` and `token_length`
- assign `length_bin`
- assign `time_band`
- apply symmetric URL masking to all groups

### Email-specific preprocessing
- remove RFC 2822 headers
- decode MIME / quoted-printable / base64 safely
- strip HTML tags
- remove multipart boundaries

### SMS-specific preprocessing
- normalize whitespace only
- preserve abbreviations and punctuation
- do not expand slang or abbreviations

### QA-specific preprocessing
- use answer text as `text`
- keep `question_id`
- keep paired metadata

---

## 4.8. Human-side quality gates

Before Step 2 begins, the human side must satisfy all of the following:

1. no duplicate leakage across source families
2. English-only
3. all records assigned:
   - `fraudness`
   - `channel`
   - `scenario_family`
   - `time_band`
   - `length_bin`
4. Nazario split into `legacy` and `modern`
5. ham audit completed for:
   - Enron ham
   - SpamAssassin ham
   - SMS ham
6. Smishtank cleaned and deduplicated
7. HC3 human answers aligned by `question_id`

---

## 5. Step 2 — generate Dataset 2 (LLM-generated side)

This step creates the generated half of Core.

---

## 5.1. How many prompt families are required

For Core, exactly **4 prompt families** must be created or finalized:

1. `phishing_email`
2. `fraud_sms_phishing_like`
3. `legitimate_email`
4. `legitimate_sms`

### No prompt required for
- `financial_qa` because HC3 Finance chatgpt is used as a ready paired source

---

## 5.2. Generator split

### Seen generators (train/val eligible)
- one OpenAI-family generator
- one open-weight / non-OpenAI instruction-tuned generator (e.g. Mistral family)

### Holdout generator (test only)
- Claude family

This keeps the required unseen-generator evaluation.

### Hard rule
Claude-generated records must never appear in train or val.

---

## 5.3. Prompt design principles

Every Core prompt must satisfy:

1. **Matched to the human anchor**
2. **No prompt-level leakage**
3. **No unrealistic over-structuring**
4. **Same language as human side**
5. **Same channel conventions as human side**
6. **Length bins consistent with human distribution**
7. **Temperature passed as API parameter only**
8. **All generated names / institutions / URLs fictional**
9. **Output must be plain CSV rows only**
10. **No markdown, no explanations, no assistant chatter**

---

## 5.4. Prompt family 1 — `phishing_email`

**Human anchor:** Nazario  
**Channel:** email  
**Fraudness:** fraud  
**Status:** retain current T1 logic with minor cleanup only

### Required properties
- banking impersonation
- urgency / consequence / CTA
- short / medium / long bins
- realistic but fictional URLs and entities
- plain text email body

### Why it is valid
It is directly matched to a human phishing-email source.

---

## 5.5. Prompt family 2 — `fraud_sms_phishing_like`

**Human anchor:** Smishtank  
**Channel:** sms  
**Fraudness:** fraud  
**Status:** redesign current T2

### Required properties
- only phishing-like / deceptive fraud SMS
- no broad “all spam” scenario family
- short only
- one CTA
- sms-native style
- fictional sender IDs / URLs / numbers

### Allowed themes
Keep only themes that match the human Smishtank distribution.  
Examples:
- account security alert
- suspicious transaction
- card blocked
- delivery/service action required
- refund verification

### Remove from this prompt if not supported by human anchor
- overly broad promo spam
- generic marketing spam
- adult spam
- purely commercial acquisition texts

### Why redesign is necessary
The earlier T2 was too easy to mismatch with a broad spam-SMS source.

---

## 5.6. Prompt family 3 — `legitimate_email`

**Human anchor:** Enron ham + SpamAssassin ham  
**Channel:** email  
**Fraudness:** legitimate  
**Status:** new prompt required

### Required properties
- benign human-like email communication
- not phishing
- not fake bank-notification by default
- not too templated
- should reflect the ham audit distribution

### Theme pool must be derived from ham audit
The exact theme list must be created from the audit, but may include:
- work coordination
- scheduling / logistics
- informational follow-up
- document / file discussion
- administrative message
- service/system-like benign message

### Critical rule
This prompt must **not** be a renamed version of `bank_notification`.

### Why this prompt exists
The collected human legitimate email side is not a bank-notification corpus; it is a broader benign email pool.

---

## 5.7. Prompt family 4 — `legitimate_sms`

**Human anchor:** SMS ham  
**Channel:** sms  
**Fraudness:** legitimate  
**Status:** new prompt required

### Required properties
- benign SMS only
- short only
- should reflect the ham audit distribution
- not fake transaction alerts unless the ham audit shows a strong benign transactional slice

### Theme pool must be derived from ham audit
Possible examples:
- coordination
- personal check-ins
- reminders
- service notification
- logistics

### Critical rule
This prompt must **not** become a banking-notification SMS generator unless the human legitimate SMS side actually supports that.

---

## 5.8. Claude holdout set

### Purpose
To test cross-model generalization.

### Construction
For each of the 4 prompt families:
- run the same prompt family on Claude
- keep generation settings parallel to the seen generators
- generate only for `test`

### Rules
- same schema
- same length bins
- same scenario families
- same anonymization principles
- same preprocessing
- **never mixed into train/val**

---

## 5.9. How many prompts in total

### Core prompt families
- `4` scenario prompts:
  - phishing_email
  - fraud_sms_phishing_like
  - legitimate_email
  - legitimate_sms

### Prompt templates
Each scenario prompt should contain:
- one **system prompt**
- one **user prompt template**

Thus:
- `4 system prompts`
- `4 user prompt templates`

### Generator usage
The same prompt family is reused across:
- seen generator A
- seen generator B
- Claude holdout

So the number of prompt families stays `4`, not `12`.

---

## 5.10. Length bins

### Email prompts
- `short`
- `medium`
- `long`

### SMS prompts
- `short` only

### QA
No new prompt needed; use existing HC3 structure

### Why length bins are required
They remove a shortcut where models would distinguish classes by length rather than authorship.

---

## 5.11. Generated-side quality gates

Before assembly, generated data must satisfy:

1. no CSV wrapper corruption
2. plain text only
3. no prompt leakage
4. no repeated templated artifacts dominating a slice
5. correct `scenario_family`
6. correct `channel`
7. correct `length_bin`
8. correct `origin_model`
9. no real institutions, real URLs, or real personal identifiers
10. Claude outputs isolated to holdout test

---

## 6. Final Core assembly

---

## 6.1. Core source map

| Group | Human | LLM |
|---|---|---|
| fraud / email | Nazario | generated `phishing_email` |
| fraud / sms | Smishtank | generated `fraud_sms_phishing_like` |
| legitimate / email | Enron ham + SpamAssassin ham | generated `legitimate_email` |
| legitimate / sms | SMS ham | generated `legitimate_sms` |
| legitimate / qa | HC3 Finance human | HC3 Finance chatgpt |

---

## 6.2. Split design

### Train / val
- human sources above
- seen generator outputs
- HC3 Finance chatgpt

### Test
- held-out slices from the same source families
- Claude-generated versions of the 4 prompt families
- HC3 Finance chatgpt held-out split as needed

### Hard constraints
- no duplicate leakage across splits
- no same generated record across train and test
- no Claude in train/val

---

## 6.3. Target counts

### Principle
- do **not** force all slices to identical size if that requires weak sources
- cap large sources
- use all clean records from scarce but valid sources
- match LLM volume to human-side volume per slice where feasible

### Priority order
1. slice integrity
2. source cleanliness
3. match quality
4. size

### Important note
A smaller but cleaner Core is preferable to a larger, mismatched Core.

---

## 7. Final checklist before baseline training

The Core dataset is **ready** only if every item below is satisfied:

### Source validity
- [ ] Nazario included and split by `time_band`
- [ ] Smishtank included and cleaned
- [ ] Enron ham cleaned
- [ ] SpamAssassin ham cleaned
- [ ] SMS ham cleaned
- [ ] HC3 Finance paired correctly

### Diagnostic audits
- [ ] Enron ham audit complete
- [ ] SpamAssassin ham audit complete
- [ ] SMS ham audit complete

### Prompt readiness
- [ ] phishing_email prompt finalized
- [ ] fraud_sms_phishing_like prompt finalized
- [ ] legitimate_email prompt finalized
- [ ] legitimate_sms prompt finalized

### Generation readiness
- [ ] seen-generator outputs produced
- [ ] Claude holdout outputs produced
- [ ] no prompt leakage
- [ ] schema fields complete

### Assembly readiness
- [ ] global deduplication complete
- [ ] symmetric preprocessing complete
- [ ] English-only
- [ ] fields assigned
- [ ] split integrity checked
- [ ] no Claude leakage into train/val

---

## 8. Critical self-assessment: is this enough to fix the earlier mistakes?

### Yes — for the following earlier mistakes
1. broad spam pools are no longer treated as narrow fraud classes
2. unmatched LLM scenarios are removed from Core
3. legitimate side is matched to actual human legitimate distributions
4. `financial_qa` is treated as a bilateral control slice rather than a fake fraud slice
5. temporal confounding is made explicit via `time_band`
6. generator confounding is handled via Claude holdout
7. URL asymmetry is removed via symmetric masking

### Remaining risk
The one real dependency is **Smishtank availability and quality**.  
If Smishtank cannot be used, the Core fraud SMS slice is unresolved and the Core should either:
- proceed temporarily without fraud SMS, or
- be explicitly marked incomplete.

### Final judgment
If Smishtank is successfully integrated and the two new legitimate prompts are designed from the ham audits, then this Core specification is sufficient to correct the most important design errors made in the earlier dataset versions.

---

## 9. Definition of done

Dataset Design v3 is considered successfully implemented when:

1. Dataset 1 (human side) is prepared exactly as specified
2. Dataset 2 (generated side) is produced exactly as specified
3. all Core fields are present
4. all Core quality gates pass
5. train/val/test are assembled with no leakage
6. the resulting Core dataset can be used directly for baseline training without changing source composition or scenario families

