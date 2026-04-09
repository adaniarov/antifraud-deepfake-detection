# LLM prompt families — generation contract (v2)

Canonical specification for Core **Dataset 2** text generation: five scenario families implemented as JSON under [`v2/data/prompts/`](../data/prompts/) (`phishing_email.json`, `advance_fee_scam_email.json`, `fraud_sms_deceptive.json`, `legitimate_email.json`, `legitimate_sms.json`). Normative dataset rules: [`dataset_design_final.md`](dataset_design_final.md) §6; implementation: [`core_as_built.md`](core_as_built.md), [`llm_mass_generation.py`](../src/llm_mass_generation.py).

**Pilot refinements (post-review):** this revision tightened two weak areas:

1. `legitimate_email` — further away from office-assistant / update language, toward narrow operational mail.
2. `phishing_email` — tighter guard against billing/helpdesk/support tone, especially `invoice_or_payment_lure`, with a small cleanup for `account_verification`.

All other families were left unchanged as already fit for production-scale generation.

## Scope

| scenario_family | channel | fraudness | length_bins | subtypes |
|----------------|---------|-----------|-------------|---------|
| `phishing_email` | email | fraud | short / medium / long | 9 |
| `advance_fee_scam_email` | email | fraud | medium / long | 4 |
| `fraud_sms_deceptive` | sms | fraud | short / medium | 3 |
| `legitimate_email` | email | legitimate | short / medium / long | 5 |
| `legitimate_sms` | sms | legitimate | short only | 2 |

`financial_qa` has **no prompt family** in Core.

---

## Generation contract

### Output format

- Output **plain text only**
- Email: body only, no `Subject:` / `From:` / `To:` / `Date:`
- SMS: message text only
- No markdown, HTML, JSON, lists, labels, explanations, wrapper text
- No mention that the text is generated or simulated

### Masking and anonymization

Apply to **all generated text** before saving:

| Entity type | Replacement |
|-------------|-------------|
| URL | `[URL]` |
| Email | `[EMAIL]` |
| Phone | `[PHONE]` |
| Real personal names | fictionalize |
| Real institutions / brands | fictionalize |

### Length policy

`length_bin_word_guide` is a **soft generation hint** only.

Hard validation uses token counts:

| Channel | short | medium | long |
|---------|-------|--------|------|
| sms | < 20 tokens | 20–59 tokens | ≥ 60 tokens |
| email | < 100 tokens | 100–399 tokens | ≥ 400 tokens |

### Few-shot anchors

`few_shot_anchors` are stored for offline reference / QC only.  
They are **not injected** into production prompts.

### Sampling policy

| family | length split |
|--------|--------------|
| `phishing_email` | short 20% / medium 50% / long 30% |
| `advance_fee_scam_email` | medium 40% / long 60% |
| `fraud_sms_deceptive` | short 15% / medium 85% |
| `legitimate_email` | short 65% / medium 25% / long 10% |
| `legitimate_sms` | short only |

### Temperature

Use temperature as API parameter only.  
Recommended default: `0.9`.

### Acceptance / retry protocol

For each candidate:

1. Generate candidate
2. Validate:
   - non-empty (`>= 5 tokens`)
   - plain text only
   - no unmasked URLs
   - no unmasked emails
   - no unmasked phone numbers
   - token count in valid family/channel range
   - no emojis for SMS families
   - no obviously unchanged real brands / personal identifiers
   - flag if output reads like an office-assistant update / PM summary / campaign-planning mail when family is `legitimate_email`
   - flag if output reads like a customer-support / billing escalation letter when family is `phishing_email`, especially `invoice_or_payment_lure`
3. If validation passes → save
4. Else retry up to `max_retries = 3`
5. If still failing → log and skip

### Pilot QC before full generation

Before large-scale generation, run a pilot:

- 5–8 samples per subtype
- manually review:
  - family correctness
  - over-polish vs intended register
  - repeated openings / repeated templates
  - masking failures
  - legitimate email not drifting into budget/project/update/campaign style
  - phishing not drifting into helpdesk/billing style

### Soft word-count reference

| bin | email hint | sms hint (fraud) | sms hint (legitimate) |
|-----|------------|------------------|-----------------------|
| short | 24–70 words | 5–15 words | 4–14 words |
| medium | 75–160 words | 16–35 words | — |
| long | 200–460 words | — | — |

These are hints only. Final assignment uses token counts.
