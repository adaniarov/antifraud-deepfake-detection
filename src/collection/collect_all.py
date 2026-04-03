"""
collect_all.py — Stage 1 orchestrator.

Outputs:
  data/raw/
    human_fraud/   label=0, fraud-related human text
    human_legit/   label=0, legitimate human text
    llm_fraud/     label=1, LLM-generated fraud-related text (HC3 chatgpt)
    llm_legit/     label=1, LLM-generated legitimate text   (HC3 chatgpt)

Stage 2 will add more LLM-generated texts produced via API calls.

Usage:
    uv run python -m src.collection.collect_all
    uv run python -m src.collection.collect_all --skip-enron
"""

import argparse
import time
from pathlib import Path

from loguru import logger

HUMAN_FRAUD = Path("data/raw/human_fraud")
HUMAN_LEGIT = Path("data/raw/human_legit")
LLM_FRAUD   = Path("data/raw/llm_fraud")
LLM_LEGIT   = Path("data/raw/llm_legit")


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in open(path, encoding="utf-8") if line.strip())


def print_summary(t0: float) -> None:
    groups = {
        "human_fraud (label=0)": (HUMAN_FRAUD, [
            "spamassassin_spam.jsonl",
            "enron_spam.jsonl",
            "sms_spam.jsonl",
            "nazario.jsonl",
            "nigerian_fraud.jsonl",
            "hc3_finance.jsonl",
        ]),
        "human_legit  (label=0)": (HUMAN_LEGIT, [
            "spamassassin_ham.jsonl",
            "enron_ham.jsonl",
            "sms_ham.jsonl",
            "yelp.jsonl",
        ]),
        "llm_fraud    (label=1)": (LLM_FRAUD, [
            "hc3_finance.jsonl",
        ]),
        "llm_legit    (label=1)": (LLM_LEGIT, [
            "hc3_legit.jsonl",
        ]),
    }

    print("\n" + "=" * 60)
    print("  COLLECTION SUMMARY  (raw, before preprocessing)")
    print("=" * 60)
    grand_total = 0
    for group_name, (base_dir, files) in groups.items():
        group_total = 0
        print(f"\n  {group_name}:")
        for fname in files:
            n = count_jsonl(base_dir / fname)
            group_total += n
            tag = f"{n:>6}" if n > 0 else "     - (not collected)"
            print(f"    {fname:<35} {tag}")
        print(f"    {'TOTAL':<35} {group_total:>6}")
        grand_total += group_total

    print(f"\n  Grand total raw records : {grand_total}")
    print(f"  Elapsed                 : {time.time() - t0:.0f}s")
    print("=" * 60)
    print("\nNext: uv run python -m src.preprocessing.preprocess\n")


def main(skip_enron: bool = False) -> None:
    t0 = time.time()

    # Create output dirs
    for d in (HUMAN_FRAUD, HUMAN_LEGIT, LLM_FRAUD, LLM_LEGIT):
        d.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 50)
    logger.info("Stage 1 — Human + Ready-made LLM Corpus Collection")
    logger.info("=" * 50)

    # 1. SpamAssassin
    logger.info("\n[1/6] SpamAssassin")
    from .spamassassin import collect as c
    c(HUMAN_FRAUD, HUMAN_LEGIT)

    # 2. Enron-Spam
    if skip_enron:
        logger.info("\n[2/6] Enron — skipped (--skip-enron)")
    else:
        logger.info("\n[2/6] Enron-Spam  (~50 MB)")
        from .enron import collect as c
        c(HUMAN_FRAUD, HUMAN_LEGIT)

    # 3. SMS Spam
    logger.info("\n[3/6] SMS Spam Collection")
    from .sms_spam import collect as c
    c(HUMAN_FRAUD, HUMAN_LEGIT)

    # 4. Nazario
    logger.info("\n[4/6] Nazario Phishing")
    from .nazario import collect as c
    c(HUMAN_FRAUD)

    # # 5. Nigerian Fraud
    # logger.info("\n[5/6] Nigerian Fraud  (manual Kaggle download)")
    # from .nigerian_fraud import collect as c
    # c(HUMAN_FRAUD)

    # 6. HC3 — both human and chatgpt answers
    logger.info("\n[6/6] HC3 Finance  (human + chatgpt answers)")
    from .hc3 import collect as c
    c(HUMAN_FRAUD, HUMAN_LEGIT, LLM_FRAUD, LLM_LEGIT)

    # 7. Yelp
    logger.info("\n[+] Yelp Financial Reviews")
    from .yelp import collect as c
    c(HUMAN_LEGIT)

    print_summary(t0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-enron", action="store_true")
    args = parser.parse_args()
    main(skip_enron=args.skip_enron)