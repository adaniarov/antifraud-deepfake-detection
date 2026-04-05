# docs/next_tasks.md

## Priority queue

### P0
- Implement automated per-content-type evaluation for all model outputs
- Prepare transformer training data loaders and baseline configs
- Define final reporting format for val / test full / Claude / Non-Claude

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
