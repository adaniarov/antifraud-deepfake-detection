"""Shared helpers for LLM annotation notebooks (flat JSONL output)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from sklearn.model_selection import train_test_split

# --- I/O ---


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def md5_text_key(text: str) -> str:
    return hashlib.md5((text or "").strip().encode("utf-8", errors="replace")).hexdigest()


def load_flat_annotation_index(path: Path) -> dict[str, dict]:
    """Map md5(text) -> full row for rows that already have scenario_family."""
    m: dict[str, dict] = {}
    for r in load_jsonl(path):
        if not r.get("scenario_family"):
            continue
        k = md5_text_key(r.get("text", ""))
        m[k] = r
    return m


# --- Text / strata ---


def wc(t: str) -> int:
    return len((t or "").split())


def wc_bin(n: int) -> str:
    if n < 40:
        return "short"
    if n < 200:
        return "medium"
    return "long"


def proxy_bucket_spam(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ("beneficiary", "next of kin", "inheritance", "million dollar", "barrister")):
        return "p419"
    if any(k in t for k in ("verify", "password", "account", "suspended", "click here", "paypal")):
        return "pphish"
    if any(k in t for k in (".exe", "attachment", "virus", "zip")):
        return "pmalw"
    if any(k in t for k in ("unsubscribe", "free", "offer", "viagra", "discount", "casino")):
        return "ppromo"
    return "pother"


KW_419 = (
    "beneficiary",
    "next of kin",
    "deceased",
    "inheritance",
    "million dollar",
    "barrister",
    "foreign account",
    "strictly confidential",
    "late client",
    "transfer of funds",
    "funds transfer",
    "oil contract",
    "dear friend",
    "dearest beloved",
    "am contacting you",
    "good day",
)


def is_419_candidate(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in KW_419)


def stratified_sample_df(df: pd.DataFrame, n: int, strata_cols: list[str], seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    df = df.copy().reset_index(drop=True)
    for c in strata_cols:
        if c not in df.columns:
            df[c] = "_na_"
        df[c] = df[c].fillna("_na_").astype(str)
    y = df[strata_cols].agg("|".join, axis=1)
    try:
        sample, _ = train_test_split(df, train_size=n, stratify=y, random_state=seed, shuffle=True)
        return sample.reset_index(drop=True)
    except ValueError:
        return df.sample(n=n, random_state=seed).reset_index(drop=True)


def dedupe_records_by_text_sha(records: list[dict], text_key: str = "text") -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for r in records:
        k = hashlib.sha256((r.get(text_key) or "").strip().encode("utf-8", errors="replace")).hexdigest()
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def ensure_sample(path: Path, builder: Callable[[], list[dict]]) -> list[dict]:
    if path.exists():
        return load_jsonl(path)
    rows = builder()
    save_jsonl(path, rows)
    print(f"Created sample {path.name}  n={len(rows)}")
    return rows


# --- Flat record (unified output schema) ---


def make_flat_record(
    raw: dict[str, Any],
    *,
    scenario_family: str,
    annotation_confidence: str,
    annotation_model: str,
    annotated_at: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Flat JSONL row: all original raw fields + scenario_family + annotation_* + optional flags.
    Same shape as sms_spam_annotated.jsonl (plus optional diagnostic booleans).
    """
    out = {k: v for k, v in raw.items() if not str(k).startswith("_")}
    out["scenario_family"] = scenario_family
    out["annotation_confidence"] = annotation_confidence
    out["annotation_model"] = annotation_model
    if annotated_at:
        out["annotated_at"] = annotated_at
    if extra:
        for k, v in extra.items():
            if v is not None:
                out[k] = v
    return out


def nested_legacy_line_to_flat(obj: dict[str, Any]) -> dict[str, Any] | None:
    """Convert old block_ab/cache/*.jsonl nested row to flat schema. Returns None if not nested."""
    if "raw" not in obj or "annotation" not in obj:
        return None
    raw = obj["raw"]
    ann = obj["annotation"]
    extra = {
        k: ann[k]
        for k in (
            "core_candidate",
            "has_financial_pretence",
            "has_credentials_request",
            "has_reward_or_prize",
            "has_confidentiality_appeal",
            "is_deceptive_attack",
            "has_urgency",
            "has_action_request",
            "has_url_or_phone_cta",
        )
        if k in ann
    }
    return make_flat_record(
        raw,
        scenario_family=str(ann.get("category", "unclear_other")),
        annotation_confidence=str(ann.get("confidence", "low")),
        annotation_model=str(obj.get("annotation_model", "")),
        annotated_at=obj.get("annotated_at"),
        extra=extra if extra else None,
    )


def migrate_nested_cache_to_flat(nested_path: Path, flat_path: Path, skip_existing: bool = True) -> tuple[int, int]:
    """
    Read nested JSONL, append flat lines to flat_path.
    Returns (written, skipped_duplicate).
    """
    existing = load_flat_annotation_index(flat_path) if skip_existing and flat_path.exists() else {}
    written = 0
    skipped = 0
    for obj in load_jsonl(nested_path):
        flat = nested_legacy_line_to_flat(obj)
        if flat is None:
            continue
        k = md5_text_key(flat.get("text", ""))
        if k in existing:
            skipped += 1
            continue
        append_jsonl(flat_path, flat)
        existing[k] = flat
        written += 1
    return written, skipped
