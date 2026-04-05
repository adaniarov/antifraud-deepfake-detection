# docs/dataset_contract.md

## Task
Binary classification:
- human-written -> label 0
- LLM-generated -> label 1

## Factorial design
- A = Human + Fraudulent
- B = Human + Legitimate
- C = LLM + Fraudulent
- D = LLM + Legitimate

## Content types
- T1: Phishing email
- T2: Smishing SMS
- T3: Social engineering support message
- T4: 419 / advance-fee fraud letter
- T5: Bank / financial notification
- T6: Financial review / feedback

## Human sources
- Nazario Phishing Corpus
- Enron-Spam / SpamAssassin
- SMS Spam Collection
- HC3 Finance
- Yelp Reviews

## LLM sources
Train / val:
- GPT-4o-mini
- Mistral-Small-3.2-24B-Instruct

Test only (held-out):
- Claude Haiku

## Split sizes
- train: 10,658
- val: 1,732
- test: 6,603

## Test partition design
Claude partition:
- total n = 5,669
- Claude-generated LLM texts = 2,574
- Human companion texts = 3,095

Non-Claude partition:
- total n = 934
- seen-generator LLM texts = 467
- Human companion texts = 467

## Hard invariants
- Held-out Claude must remain test-only.
- Temperature is an API parameter, not a prompt instruction.
- URL masking must be symmetric across all groups.
- Do not silently redefine labels.
- Do not merge content-type classification with authorship detection in interpretation.

## Known limitations
- temporal bias
- genre imbalance
- limited generator coverage

## Use rule
If any code or draft conflicts with this file, this file wins unless the user explicitly changes the project design.
