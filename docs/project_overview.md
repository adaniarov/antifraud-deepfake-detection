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
