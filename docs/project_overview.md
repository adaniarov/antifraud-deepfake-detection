# docs/project_overview.md

## Project summary
This repository contains code and materials for a bachelor thesis on detecting LLM-generated anti-fraud texts.

## Goal
Develop and compare algorithms for detecting human-written vs LLM-generated texts in anti-fraud settings.

## Main components
1. Dataset creation and curation
2. Preprocessing and feature engineering
3. Classical ML baselines
4. Transformer fine-tuning
5. Ensemble methods
6. Robustness evaluation
7. Thesis writing and prototype development

## Research question
Which methods provide the best trade-off between performance, cross-model generalization, and robustness for detecting textual deepfake attacks in anti-fraud systems?

## Main task
Binary classification:
- 0 = human
- 1 = LLM

## Domain
Anti-fraud messages only:
- phishing
- smishing
- social engineering
- 419 scam
- bank notifications
- financial reviews

## Evaluation focus
- in-domain quality
- generalization to unseen generator
- partition-based evaluation
- per-content-type diagnostics
- robustness under paraphrasing attacks

## Key deliverables
- reproducible dataset
- strong classical baseline
- transformer baseline(s)
- ensemble comparison
- robustness experiments
- thesis chapters and prototype

## Alignment with agreed thesis outline (methods + experiments)
- **Methods (thesis Ch. 3):** classical feature engineering and TF-IDF vectorization are implemented; perplexity features exist in code but the published baseline table excludes them; neural methods (pretrained embeddings, fine-tuning, DistilBERT-style speed trade-offs) are **not yet** in the training pipeline.
- **Experiments (thesis Ch. 4):** fixed train/val/test evaluation and partition metrics are implemented; **k-fold cross-validation** and a **full comparative feature-importance study** (e.g. SHAP) are **not** part of the current baseline reporting—see `docs/project_status.md` for the exact status.
