# docs/project_status.md

## Current stage
Dataset assembled; classical baseline complete; transformer and robustness stages are next.

## Completed
- Chapter 1 literature review completed
- Anti-fraud task formulation fixed
- 2x2 dataset design fixed
- Train / val / test splits assembled
- Held-out Claude test partition defined
- Classical ML baseline trained and evaluated
- Partition-based evaluation protocol defined

## Current best confirmed model
Best classical baseline:
- XGBoost + hc_plus_char
- feature count: 10,044
- no perplexity features

## Confirmed performance snapshot
Validation:
- Accuracy: 0.9931
- F1 (LLM): 0.9931
- ROC-AUC: 0.9997

Test Full:
- Accuracy: 0.9612
- F1 (LLM): 0.9567
- F1 (Human): 0.9649
- ROC-AUC: 0.9966

Test Claude partition:
- Accuracy: 0.9566
- F1 (LLM): 0.9505
- F1 (Human): 0.9614
- ROC-AUC: 0.9963

Test Non-Claude partition:
- Accuracy: 0.9893
- F1 (LLM): 0.9893
- F1 (Human): 0.9893
- ROC-AUC: 0.9983

## Main confirmed finding
There is a noticeable cross-model generalization gap:
- Non-Claude partition is stronger than Claude partition
- smishing is the hardest content type

## Next high-value tasks
1. Add per-content-type evaluation into the main reporting pipeline
2. Prepare DistilBERT full fine-tuning baseline
3. Prepare RoBERTa + LoRA baseline
4. Build ensemble evaluation plan
5. Design robustness benchmark and attack generation pipeline

## Known risks and caveats
- Temporal bias: older human texts vs recent LLM texts
- Genre imbalance: some content types exist only for one class in current version
- Limited generator coverage
- Partition metrics must be interpreted carefully

## Source of truth for numbers
- outputs/tables/classical_results_no_ppl.csv
- data/final/dataset_stats.json

## Maintenance rule
Update this file after every major milestone or validated experiment.
