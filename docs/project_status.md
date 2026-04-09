# docs/project_status.md

## Current stage
Dataset assembled; classical baseline complete on fixed splits; interpretability and cross-validation gaps remain for thesis §4.2; transformer and robustness stages are next.

## Completed
- Chapter 1 literature review completed
- Anti-fraud task formulation fixed
- 2x2 dataset design fixed
- Train / val / test splits assembled
- Held-out Claude test partition defined
- Classical ML baseline trained and evaluated
- Partition-based evaluation protocol defined

### Baseline methodology (fact-checked against code)
- **Splits:** metrics come from a **single fixed** train / val / test partition (`docs/dataset_contract.md`). The main script `src/models/train_classical.py` does **not** use k-fold cross-validation; validation is one hold-out set, test is used for final reporting.
- **Feature groups:** hand-crafted stylometric / lexical / linguistic features plus separate TF-IDF word and char matrices (`src/features/extract.py`). Perplexity and burstiness are implemented there but the **reported baseline numbers** use the no-perplexity run (`outputs/tables/classical_results_no_ppl.csv`).
- **Models:** logistic regression, random forest, and XGBoost on several feature-set combinations; best configuration documented below.
- **Feature importance:** `src/features/visualize_features.py` can train XGBoost on **hand-crafted features only** and export top features via native `feature_importances_` (gain) to `outputs/figures/feature_importance.csv`. This is **not** SHAP, **not** integrated into `train_classical.py`, and **not** computed for the full sparse best model (e.g. hc_plus_char with TF-IDF).

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

## Thesis plan alignment (agreed Ch. 3–4 outline)

### Chapter 3. Methods and algorithms
| Block | Status |
| --- | --- |
| 3.1 Features: statistical, stylometric, linguistic | Done (extraction + classical baselines) |
| 3.1 Perplexity-based features | Implemented in code; **excluded** from current reported baseline table (no-ppl CSV) |
| 3.1 TF-IDF n-grams | Done |
| 3.2 Classical ML (LR, RF, XGBoost) | Done |
| 3.3 Neural: embeddings as featurizers, fine-tuning, distilled models | **Not done** (planned) |

### Chapter 4. Experimental study
| Block | Status |
| --- | --- |
| 4.1 Environment, splits, metrics | Splits + metrics in pipeline; **full software/hardware write-up for thesis** still to finalize |
| 4.2 Classical stage: CV + feature-importance analysis | **CV not run** for reported baseline; **partial** importance (hand-crafted XGBoost only, no SHAP) |
| 4.3 Deep learning stage | **Not done** |
| 4.4 Robustness (cross-model, paraphrase / rewrite) | **Not done** |
| 4.5 Summary comparison | Pending prior stages |

## Next high-value tasks
1. Close thesis §4.2 gaps: stratified k-fold CV on train+val (or train-only) for classical models **or** justify single-split protocol; add interpretability (e.g. SHAP or permutation importance on a tractable subset / grouped features)
2. Add per-content-type evaluation into the main reporting pipeline
3. Prepare DistilBERT full fine-tuning baseline
4. Prepare RoBERTa + LoRA baseline
5. Build ensemble evaluation plan
6. Design robustness benchmark and attack generation pipeline

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
