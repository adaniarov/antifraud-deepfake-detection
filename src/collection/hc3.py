"""
Human ChatGPT Comparison Corpus (HC3)
--------------------------------------
Source : https://github.com/Hello-SimpleAI/chatgpt-comparison-detection
Ref    : W. Guo et al., arXiv:2301.07597, 2023.

Each record:
  {"id": ..., "question": ..., "human_answers": [...], "chatgpt_answers": [...]}

Both splits are collected here — HC3 is a finished dataset, not something
we generate ourselves. Stage 2 (LLM generation) is reserved for texts
we produce via API calls to GPT-4, Claude, Mistral, etc.

Outputs:
  human_answers    -> data/raw/human_fraud/hc3_finance.jsonl   label=0
  chatgpt_answers  -> data/raw/llm_fraud/hc3_finance.jsonl     label=1
"""

import json
from pathlib import Path

import httpx
from loguru import logger

from .common import make_record, save_jsonl

HF_BASE = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main"

DOMAINS = {
    "finance": {
        "human_group": "fraud",     # -> human_fraud/
        "llm_group": "fraud",       # -> llm_fraud/
        "content_type": "financial_qa",
    },
    "open_qa": {
        "human_group": "legit",
        "llm_group": "legit",
        "content_type": "legitimate",
    },
}


def _make_llm_record(text: str, content_type: str, source_id: str) -> dict:
    """Build a record for LLM-generated text (label=1)."""
    return {
        "text": text,
        "label": 1,
        "label_str": "llm",
        "origin_model": "chatgpt",   # HC3 doesn't specify exact version
        "content_type": content_type,
        "dataset_source": source_id,
        "char_length": len(text),
        "split": None,
    }


def _download_domain(domain: str) -> list:
    url = f"{HF_BASE}/{domain}.jsonl"
    logger.info(f"  Fetching HC3/{domain} ...")
    try:
        resp = httpx.get(url, timeout=60, follow_redirects=True)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"  Failed HC3/{domain}: {e}")
        return []
    return [json.loads(line) for line in resp.text.splitlines() if line.strip()]


def collect(fraud_dir, legit_dir, llm_fraud_dir, llm_legit_dir) -> dict:
    fraud_dir     = Path(fraud_dir)
    legit_dir     = Path(legit_dir)
    llm_fraud_dir = Path(llm_fraud_dir)
    llm_legit_dir = Path(llm_legit_dir)

    stats = {}

    human_fraud, human_legit = [], []
    llm_fraud, llm_legit     = [], []

    for domain, cfg in DOMAINS.items():
        items = _download_domain(domain)
        if not items:
            stats[domain] = {"human": 0, "llm": 0}
            continue

        h_count = l_count = 0
        source_id = f"hc3_{domain}"

        for item in items:
            ct = cfg["content_type"]

            for answer in item.get("human_answers", []):
                text = str(answer).strip()
                if not text:
                    continue
                rec = make_record(text, ct, source_id)
                if cfg["human_group"] == "fraud":
                    human_fraud.append(rec)
                else:
                    human_legit.append(rec)
                h_count += 1

            for answer in item.get("chatgpt_answers", []):
                text = str(answer).strip()
                if not text:
                    continue
                rec = _make_llm_record(text, ct, source_id)
                if cfg["llm_group"] == "fraud":
                    llm_fraud.append(rec)
                else:
                    llm_legit.append(rec)
                l_count += 1

        logger.info(f"  HC3/{domain}: {h_count} human | {l_count} chatgpt")
        stats[domain] = {"human": h_count, "llm": l_count}

    # Save human records
    if human_fraud:
        save_jsonl(human_fraud, fraud_dir / "hc3_finance.jsonl")
        logger.success(f"HC3 human -> human_fraud: {len(human_fraud)}")
    if human_legit:
        save_jsonl(human_legit, legit_dir / "hc3_legit.jsonl")
        logger.success(f"HC3 human -> human_legit: {len(human_legit)}")

    # Save LLM records
    if llm_fraud:
        save_jsonl(llm_fraud, llm_fraud_dir / "hc3_finance.jsonl")
        logger.success(f"HC3 chatgpt -> llm_fraud: {len(llm_fraud)}")
    if llm_legit:
        save_jsonl(llm_legit, llm_legit_dir / "hc3_legit.jsonl")
        logger.success(f"HC3 chatgpt -> llm_legit: {len(llm_legit)}")

    return stats


if __name__ == "__main__":
    stats = collect(
        "data/raw/human_fraud",
        "data/raw/human_legit",
        "data/raw/llm_fraud",
        "data/raw/llm_legit",
    )
    print(stats)