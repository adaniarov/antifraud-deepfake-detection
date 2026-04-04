"""
assemble.py — Stage 4: build final train / val / test splits.

Reads  : data/processed/{human_fraud, human_legit, llm_fraud, llm_legit}/*.jsonl
Writes : data/final/train.jsonl
         data/final/val.jsonl
         data/final/test.jsonl
         data/final/dataset_stats.json

Design decisions (dataset_description_v4):

1. PER-SOURCE CAPS (applied first, before any balancing)
   Yelp sources  → 3 000 records max
   All others    → 2 000 records max
   Random undersampling, seed=42.

2. CROSS-MODEL HELD-OUT (Claude → test only)
   Records with "claude" in origin_model are routed exclusively to the
   test split.  A *human companion* set is reserved alongside them so
   that the Claude partition contains both classes and allows full
   binary evaluation (precision, recall, F1, FPR).

   Train/val models : gpt-4o-mini, mistral-small-3.2-24b-instruct
   Test-only model  : claude-haiku (held-out)

3. HUMAN A:B BALANCE (within label=0)
   Group A (human_fraud) : 45 % of human pool
   Group B (human_legit) : 55 % of human pool
   Undersamples the larger group.

4. CLASS BALANCE (label=0 vs label=1 → 50 / 50)
   Random undersampling of the majority class.

5. STRATIFIED SPLIT (on the balanced non-Claude pool)
   Key: (label, content_type)
   Default: train=80 %, val=13 %, test_slice=7 %
   test_slice is merged with Claude LLM + human companion to form
   the final test set.

Target sizes (approximate, depends on collection results):
  Train ~10 000  |  Val ~2 000  |  Test ~3 500

Usage:
    uv run python -m src.preprocessing.assemble
    uv run python -m src.preprocessing.assemble --dry-run
    uv run python -m src.preprocessing.assemble --train-ratio 0.80 --val-ratio 0.13
    uv run python -m src.preprocessing.assemble --claude-balanced
"""

import argparse
import collections
import json
import random
from pathlib import Path

from loguru import logger

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

PROCESSED_DIRS: dict[str, Path] = {
    "human_fraud": Path("data/processed/human_fraud"),
    "human_legit": Path("data/processed/human_legit"),
    "llm_fraud":   Path("data/processed/llm_fraud"),
    "llm_legit":   Path("data/processed/llm_legit"),
}
OUT_DIR = Path("data/final")

YELP_SOURCES  = {"yelp_financial", "yelp_hf", "yelp"}
YELP_CAP      = 3_000
DEFAULT_CAP   = 2_000

DEFAULT_TRAIN_RATIO = 0.80
DEFAULT_VAL_RATIO   = 0.13
# test_slice = 1.0 - train - val = 0.07

HUMAN_FRAUD_RATIO = 0.45   # Group A within label=0
HUMAN_LEGIT_RATIO = 0.55   # Group B within label=0

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON error in {path.name}: {e}")
    return records


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_all() -> list[dict]:
    all_records = []
    for group, d in PROCESSED_DIRS.items():
        if not d.exists():
            logger.warning(f"  {d} not found — skipping")
            continue
        group_records = []
        for f in sorted(d.glob("*.jsonl")):
            batch = load_jsonl(f)
            for r in batch:
                r["_group"] = group
            group_records.extend(batch)
        logger.info(f"  {group:<20} {len(group_records):>6,} records")
        all_records.extend(group_records)
    return all_records


# ---------------------------------------------------------------------------
# Per-source caps
# ---------------------------------------------------------------------------
def apply_caps(records: list[dict], rng: random.Random) -> list[dict]:
    by_source: dict[str, list[dict]] = collections.defaultdict(list)
    for r in records:
        by_source[r.get("dataset_source", "unknown")].append(r)

    result = []
    for src, recs in sorted(by_source.items()):
        cap = YELP_CAP if src in YELP_SOURCES else DEFAULT_CAP
        if len(recs) > cap:
            selected = rng.sample(recs, cap)
            logger.info(f"  Cap [{src}]: {len(recs):,} → {cap:,}")
        else:
            selected = recs
            logger.info(f"  Cap [{src}]: {len(recs):,} (below cap {cap:,})")
        result.extend(selected)
    return result


# ---------------------------------------------------------------------------
# Claude separation (cross-model held-out rule)
# ---------------------------------------------------------------------------
def is_claude(record: dict) -> bool:
    model = record.get("origin_model") or ""
    return "claude" in model.lower()


def separate_claude_llm(
    records: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Split into non-Claude records and Claude LLM records."""
    normal, claude_llm = [], []
    for r in records:
        if is_claude(r):
            claude_llm.append(r)
        else:
            normal.append(r)
    logger.info(f"  Claude LLM held-out (→ test only) : {len(claude_llm):,}")
    logger.info(f"  Non-Claude (→ train/val/test)      : {len(normal):,}")
    return normal, claude_llm


# ---------------------------------------------------------------------------
# Reserve human companion for Claude partition
# ---------------------------------------------------------------------------
def reserve_human_companion(
    human_records: list[dict],
    n_companion: int,
    rng: random.Random,
) -> tuple[list[dict], list[dict]]:
    """
    Stratified sample of human records to pair with Claude LLM in test.

    Returns (companion, remaining_human).
    Stratification key: content_type — ensures all human content types
    are represented in the Claude test partition.
    """
    if n_companion <= 0:
        return [], human_records

    if n_companion >= len(human_records):
        logger.warning(
            f"  Requested companion={n_companion:,} >= available "
            f"human={len(human_records):,}; using all human records"
        )
        return list(human_records), []

    # --- stratified sampling by content_type ---
    by_ct: dict[str, list[dict]] = collections.defaultdict(list)
    for r in human_records:
        by_ct[r.get("content_type", "unknown")].append(r)

    companion, remaining = [], []
    for ct, recs in by_ct.items():
        # proportional allocation
        ct_share = len(recs) / len(human_records)
        ct_n = max(1, round(n_companion * ct_share))
        ct_n = min(ct_n, len(recs))  # safety
        rng.shuffle(recs)
        companion.extend(recs[:ct_n])
        remaining.extend(recs[ct_n:])

    # adjust if rounding gave us slightly more/less than target
    if len(companion) > n_companion:
        rng.shuffle(companion)
        overflow = companion[n_companion:]
        companion = companion[:n_companion]
        remaining.extend(overflow)
    elif len(companion) < n_companion:
        deficit = n_companion - len(companion)
        rng.shuffle(remaining)
        companion.extend(remaining[:deficit])
        remaining = remaining[deficit:]

    logger.info(
        f"  Human companion for Claude partition: {len(companion):,} "
        f"(remaining human for train/val/test: {len(remaining):,})"
    )
    return companion, remaining


# ---------------------------------------------------------------------------
# Human A:B balance
# ---------------------------------------------------------------------------
def balance_human_groups(
    human_records: list[dict],
    rng: random.Random,
) -> list[dict]:
    fraud = [r for r in human_records if r["_group"] == "human_fraud"]
    legit = [r for r in human_records if r["_group"] == "human_legit"]

    total        = len(fraud) + len(legit)
    target_fraud = min(round(total * HUMAN_FRAUD_RATIO), len(fraud))
    target_legit = min(round(total * HUMAN_LEGIT_RATIO), len(legit))

    if len(fraud) > target_fraud:
        fraud = rng.sample(fraud, target_fraud)
    if len(legit) > target_legit:
        legit = rng.sample(legit, target_legit)

    logger.info(
        f"  Human A (fraud): {target_fraud:,}  |  "
        f"Human B (legit): {target_legit:,}  "
        f"(ratio {HUMAN_FRAUD_RATIO:.0%}/{HUMAN_LEGIT_RATIO:.0%})"
    )
    return fraud + legit


# ---------------------------------------------------------------------------
# Class balance (50/50)
# ---------------------------------------------------------------------------
def balance_classes(records: list[dict], rng: random.Random) -> list[dict]:
    label0 = [r for r in records if r["label"] == 0]
    label1 = [r for r in records if r["label"] == 1]
    target = min(len(label0), len(label1))
    if len(label0) > target:
        label0 = rng.sample(label0, target)
    if len(label1) > target:
        label1 = rng.sample(label1, target)
    logger.info(
        f"  After 50/50 balance: "
        f"label=0 {len(label0):,} | label=1 {len(label1):,}"
    )
    return label0 + label1


# ---------------------------------------------------------------------------
# Stratified split by (label, content_type)
# ---------------------------------------------------------------------------
def stratified_split(
    records: list[dict],
    train_ratio: float,
    val_ratio: float,
    rng: random.Random,
) -> tuple[list[dict], list[dict], list[dict]]:
    test_ratio = 1.0 - train_ratio - val_ratio

    strata: dict[tuple, list[dict]] = collections.defaultdict(list)
    for r in records:
        key = (r["label"], r.get("content_type", "unknown"))
        strata[key].append(r)

    train_all, val_all, test_all = [], [], []
    for key, recs in strata.items():
        rng.shuffle(recs)
        n = len(recs)
        if n < 3:
            train_all.extend(recs)
            continue
        n_test  = max(1, round(n * test_ratio))
        n_val   = max(1, round(n * val_ratio))
        n_train = n - n_val - n_test
        if n_train <= 0:
            train_all.extend(recs)
            continue
        test_all.extend(recs[:n_test])
        val_all.extend(recs[n_test:n_test + n_val])
        train_all.extend(recs[n_test + n_val:])

    return train_all, val_all, test_all


# ---------------------------------------------------------------------------
# Assign split field and strip internal tag
# ---------------------------------------------------------------------------
def assign_split(records: list[dict], split_name: str) -> list[dict]:
    out = []
    for r in records:
        r = dict(r)
        r["split"] = split_name
        r.pop("_group", None)
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def compute_stats(train, val, test) -> dict:
    def _s(recs):
        return {
            "total":            len(recs),
            "label_counts":     dict(collections.Counter(
                                    r["label"] for r in recs)),
            "by_source":        dict(collections.Counter(
                                    r.get("dataset_source", "?") for r in recs)),
            "by_content_type":  dict(collections.Counter(
                                    r.get("content_type", "?") for r in recs)),
            "by_model":         dict(collections.Counter(
                                    r.get("origin_model") or "human"
                                    for r in recs)),
        }

    # --- Claude partition stats (for logging / validation) ---
    claude_part = [r for r in test if is_claude(r) or
                   r.get("_companion", False)]
    non_claude_part = [r for r in test if not is_claude(r) and
                       not r.get("_companion", False)]

    return {
        "train": _s(train),
        "val":   _s(val),
        "test":  _s(test),
        "test_claude_partition":     _s(claude_part),
        "test_non_claude_partition": _s(non_claude_part),
        "total": len(train) + len(val) + len(test),
    }


def print_stats(stats: dict) -> None:
    print("\n" + "=" * 64)
    print("  DATASET ASSEMBLY SUMMARY")
    print("=" * 64)

    partitions = [
        ("train", "TRAIN"),
        ("val",   "VAL"),
        ("test",  "TEST (full)"),
        ("test_claude_partition",     "TEST — Claude partition"),
        ("test_non_claude_partition", "TEST — Non-Claude partition"),
    ]

    for key, title in partitions:
        s = stats.get(key)
        if s is None or s["total"] == 0:
            continue
        l   = s["label_counts"]
        n0  = l.get(0, l.get("0", 0))
        n1  = l.get(1, l.get("1", 0))
        tot = s["total"]
        bal = f"{n0/tot:.0%} / {n1/tot:.0%}" if tot else "—"

        print(f"\n  {title:<34} total={tot:>6,}  "
              f"human={n0:>5,}  llm={n1:>5,}  balance={bal}")

        print("    content_type breakdown:")
        for ct, n in sorted(s["by_content_type"].items(),
                            key=lambda x: -x[1]):
            print(f"      {ct:<38} {n:>5,}")

        print("    models:")
        for m, n in sorted(s["by_model"].items(), key=lambda x: -x[1]):
            if m != "human":
                print(f"      {m:<50} {n:>5,}")

    print(f"\n  Grand total: {stats['total']:,}")
    print("=" * 64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio:   float = DEFAULT_VAL_RATIO,
    dry_run:     bool  = False,
    claude_balanced: bool = False,
) -> None:
    logger.info("=" * 60)
    logger.info("Stage 4 — Dataset Assembly")
    logger.info(f"Ratios: train={train_ratio:.0%} / val={val_ratio:.0%} / "
                f"test_slice={1-train_ratio-val_ratio:.0%}")
    logger.info("=" * 60)

    rng = random.Random(RANDOM_SEED)

    # ------------------------------------------------------------------
    logger.info("\n[1/7] Loading processed records ...")
    all_records = load_all()
    logger.info(f"  Total: {len(all_records):,}")

    # ------------------------------------------------------------------
    logger.info("\n[2/7] Separating Claude LLM held-out records ...")
    non_claude, claude_llm = separate_claude_llm(all_records)

    # ------------------------------------------------------------------
    logger.info("\n[3/7] Reserving human companion for Claude partition ...")

    # Determine companion size
    if claude_balanced:
        # 50/50 in Claude partition
        n_companion = len(claude_llm)
        logger.info(f"  --claude-balanced: companion = {n_companion:,} "
                     "(matching Claude LLM count)")
    else:
        # Match the Non-Claude test_slice human count (≈7% of human pool)
        # Use a reasonable estimate; exact number depends on downstream
        # balancing, so we use test_ratio × estimated human pool size.
        # Conservative default: ~460 (same order as test_slice human).
        test_ratio = 1.0 - train_ratio - val_ratio
        human_in_pool = sum(1 for r in non_claude if r["label"] == 0)
        n_companion = max(50, round(human_in_pool * test_ratio))
        logger.info(f"  Default companion size: {n_companion:,} "
                     f"(~{test_ratio:.0%} of {human_in_pool:,} human records)")

    human_pool = [r for r in non_claude if r["label"] == 0]
    llm_pool   = [r for r in non_claude if r["label"] == 1]

    human_companion, remaining_human = reserve_human_companion(
        human_pool, n_companion, rng,
    )
    # Tag companion records for partition tracking in stats
    for r in human_companion:
        r["_companion"] = True

    # Remaining pool for train/val/test_slice
    pool_for_split = remaining_human + llm_pool

    # ------------------------------------------------------------------
    logger.info("\n[4/7] Applying per-source caps ...")
    capped = apply_caps(pool_for_split, rng)
    logger.info(f"  Total after caps: {len(capped):,}")

    # ------------------------------------------------------------------
    logger.info("\n[5/7] Balancing human fraud vs legit (A:B = 45:55) ...")
    human_recs  = [r for r in capped if r["label"] == 0]
    llm_recs    = [r for r in capped if r["label"] == 1]
    human_bal   = balance_human_groups(human_recs, rng)
    capped_rebal = human_bal + llm_recs

    # ------------------------------------------------------------------
    logger.info("\n[6/7] Balancing classes (50/50) ...")
    balanced = balance_classes(capped_rebal, rng)

    # ------------------------------------------------------------------
    logger.info("\n[7/7] Stratified split + merging Claude partition ...")
    rng.shuffle(balanced)
    train_raw, val_raw, test_slice = stratified_split(
        balanced, train_ratio, val_ratio, rng,
    )

    # Claude test = Claude LLM + human companion (both classes)
    claude_test = claude_llm + human_companion
    test_raw = test_slice + claude_test

    train = assign_split(train_raw, "train")
    val   = assign_split(val_raw,   "val")
    test  = assign_split(test_raw,  "test")

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    # ------------------------------------------------------------------
    stats = compute_stats(train, val, test)
    print_stats(stats)

    # Validate Claude partition has both classes
    cp = stats.get("test_claude_partition", {})
    cp_labels = cp.get("label_counts", {})
    n0_claude = cp_labels.get(0, cp_labels.get("0", 0))
    n1_claude = cp_labels.get(1, cp_labels.get("1", 0))
    if n0_claude == 0 or n1_claude == 0:
        logger.error(
            f"Claude partition is missing a class! "
            f"human={n0_claude}, llm={n1_claude}"
        )
        raise ValueError("Claude partition must contain both classes.")
    else:
        logger.success(
            f"Claude partition OK: human={n0_claude:,}, llm={n1_claude:,}"
        )

    if dry_run:
        logger.info("\n[DRY RUN] No files written.")
        return

    # ------------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_jsonl(train, OUT_DIR / "train.jsonl")
    save_jsonl(val,   OUT_DIR / "val.jsonl")
    save_jsonl(test,  OUT_DIR / "test.jsonl")

    # Clean internal tags before saving stats
    def _clean_stats(s):
        """Remove _companion tag from stats keys."""
        return s
    (OUT_DIR / "dataset_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
    )

    logger.success(
        f"Saved → {OUT_DIR}/  "
        f"train={len(train):,}  val={len(val):,}  test={len(test):,}"
    )
    logger.info("Next: uv run python -m src.preprocessing.validate_dataset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assemble final splits (Stage 4)",
    )
    parser.add_argument("--train-ratio", type=float,
                        default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--val-ratio",   type=float,
                        default=DEFAULT_VAL_RATIO)
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument(
        "--claude-balanced", action="store_true",
        help="Make Claude partition 50/50 (companion = n_claude_llm). "
             "Default: companion ≈ test_ratio × human_pool.",
    )
    args = parser.parse_args()

    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    main(args.train_ratio, args.val_ratio, args.dry_run, args.claude_balanced)