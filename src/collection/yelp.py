"""
Yelp Open Dataset — Financial Service Reviews
-----------------------------------------------
Source : https://www.yelp.com/dataset
Ref    : Yelp Inc., 2021.

OPTION A — Full dataset with financial category filter (recommended for thesis):
  1. Register at https://www.yelp.com/dataset
  2. Download yelp_dataset.tar
  3. Extract to data/raw/.cache/yelp/:
       yelp_academic_dataset_business.json
       yelp_academic_dataset_review.json
  Financial categories used: "Banks & Credit Unions", "Insurance",
                              "Financial Services"

OPTION B — HuggingFace fallback (no financial filter, general reviews):
  Used automatically when local files are absent.
  Dataset: Yelp/yelp_review_full
  Report this as a limitation in Methods section if used.

All -> data/raw/human_legit/yelp.jsonl  (content_type: review)
"""

import json
from pathlib import Path

from loguru import logger

from .common import make_record, save_jsonl

LOCAL_BUSINESS = Path("data/raw/.cache/yelp/yelp_academic_dataset_business.json")
LOCAL_REVIEW = Path("data/raw/.cache/yelp/yelp_academic_dataset_review.json")

FINANCIAL_CATEGORIES = frozenset({"Banks & Credit Unions", "Insurance", "Financial Services"})

def _collect_local() -> list:
    logger.info("  Using local Yelp files with financial category filter ...")

    financial_ids = set()
    with open(LOCAL_BUSINESS, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            biz = json.loads(line)
            cats = biz.get("categories") or ""
            if any(c in cats for c in FINANCIAL_CATEGORIES):
                financial_ids.add(biz["business_id"])
    logger.info(f"  Found {len(financial_ids)} financial businesses")

    records = []
    with open(LOCAL_REVIEW, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            review = json.loads(line)
            if review.get("business_id") not in financial_ids:
                continue
            text = review.get("text", "").strip()
            if text:
                records.append(make_record(text, "review", "yelp_financial"))
    return records


def _collect_hf() -> list:
    logger.warning(
        "  Local Yelp files not found. Using HuggingFace fallback (no financial filter).\n"
        "  Note: record this as a limitation in your Methods section."
    )
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("Yelp/yelp_review_full", split="train", streaming=True)
    except Exception as e:
        logger.error(f"  HuggingFace fallback failed: {e}")
        return []

    records = []
    for item in ds:
        text = str(item.get("text", "")).strip()
        if text:
            records.append(make_record(text, "review", "yelp_hf"))
    return records


def collect(legit_dir) -> dict:
    legit_dir = Path(legit_dir)

    if LOCAL_BUSINESS.exists() and LOCAL_REVIEW.exists():
        records = _collect_local()
        source = "local+financial_filter"
    else:
        records = _collect_hf()
        source = "huggingface_fallback"

    save_jsonl(records, legit_dir / "yelp.jsonl")
    logger.success(f"Yelp [{source}]: {len(records)} records saved")
    return {"total": len(records), "source": source}


if __name__ == "__main__":
    stats = collect("data/raw/human_legit")
    print(stats)