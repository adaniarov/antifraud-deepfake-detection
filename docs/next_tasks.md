# docs/next_tasks.md

## Priority queue

### P0
- Classical thesis §4.2: add stratified k-fold CV (protocol + stored metrics) **or** document why a single split is sufficient; extend interpretability beyond hand-crafted XGBoost gain (e.g. SHAP on subsample, permutation importance, or coefficient analysis for sparse LR)
- Implement automated per-content-type evaluation for all model outputs
- Prepare transformer training data loaders and baseline configs
- Define final reporting format for val / test full / Claude / Non-Claude
- Freeze a short **environment and hardware** paragraph (Python, library versions, machine specs) for thesis §4.1

### P1
- Train DistilBERT full fine-tuning baseline
- Train RoBERTa + LoRA baseline
- Compare transformer outputs with classical baseline
- Build stacking ensemble from classical probabilities + transformer logits

### P2
- Build robustness attack generation pipeline
- Evaluate degradation by attack type
- Try adversarial augmentation defense

## How Cursor should choose the next task
Prefer tasks that:
1. produce measurable outputs
2. reduce uncertainty in the thesis
3. improve reproducibility
4. do not change dataset semantics
