"""
validate_dataset.py — Stage 5: integrity checks on the assembled splits.

Reads  : data/final/train.jsonl
         data/final/val.jsonl
         data/final/test.jsonl

Checks:
  1.  No Claude records in train or val  (cross-model held-out rule)
  2.  No text overlap between splits     (SHA-256 on text_clean)
  3.  Class balance per split            (warns if > 60/40 skew)
  4.  Source dominance                   (warns if any source > 35 %)
  5.  Required fields in all records
  6.  LLM-specific fields in label=1 records
      NOTE: HC3 chatgpt records may legitimately lack length_bin /
      temperature_style / temperature_value — these produce warnings,
      not errors, since HC3 is a pre-existing dataset.
  7.  Split field matches filename
  8.  No empty text_clean fields
  9.  temperature_value is float (or None) — never a string
  10. Claude records only in test, split field == "test"
  11. Smishing records use length_bin = "short" only

Usage:
    uv run python -m src.preprocessing.validate_dataset
    uv run python -m src.preprocessing.validate_dataset --strict
"""

import argparse
import collections
import hashlib
import json
from pathlib import Path

from loguru import logger

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FINAL_DIR = Path("data/final")
SPLITS = {
    "train": FINAL_DIR / "train.jsonl",
    "val":   FINAL_DIR / "val.jsonl",
    "test":  FINAL_DIR / "test.jsonl",
}

# Fields every record must have
REQUIRED_FIELDS = {
    "text_clean", "label", "label_str",
    "dataset_source", "content_type", "char_length", "split",
}

# Fields LLM records (label=1) should have.
# HC3 records may lack the generation-specific ones → warning only, not error.
LLM_FIELDS_GENERATION = {"length_bin", "temperature_style", "temperature_value"}
# origin_model is required for ALL label=1 records (including HC3).
# generation_type is checked separately: required for api_generated, warned for HC3.
LLM_FIELDS_ALWAYS = {"origin_model"}

BALANCE_TOLERANCE      = 0.10   # warn if split deviates > 10 % from 50/50
SOURCE_DOMINANCE_LIMIT = 0.35   # warn if any single source > 35 % of a split

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.error(f"  JSON error {path.name}:{i} — {e}")
    return records


def text_hash(r: dict) -> str:
    text = (r.get("text_clean") or r.get("text") or "").strip()
    return hashlib.sha256(text.encode()).hexdigest()


def is_claude(r: dict) -> bool:
    return "claude" in (r.get("origin_model") or "").lower()

# ---------------------------------------------------------------------------
# Result collector
# ---------------------------------------------------------------------------

class Result:
    def __init__(self):
        self.errors:   list[str] = []
        self.warnings: list[str] = []
        self.passed:   list[str] = []

    def error(self, msg: str):
        self.errors.append(msg)
        logger.error(f"  ✗  {msg}")

    def warn(self, msg: str):
        self.warnings.append(msg)
        logger.warning(f"  ⚠  {msg}")

    def ok(self, msg: str):
        self.passed.append(msg)
        logger.info(f"  ✓  {msg}")

    @property
    def is_valid(self) -> bool:
        return not self.errors

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_no_claude_in_train_val(splits: dict, r: Result):
    for name in ("train", "val"):
        leaks = [rec for rec in splits[name] if is_claude(rec)]
        if leaks:
            models = {rec.get("origin_model") for rec in leaks}
            r.error(f"Claude records in {name}: {len(leaks)} (models: {models}) "
                    f"— cross-model held-out VIOLATED")
        else:
            r.ok(f"No Claude records in {name}")


def check_no_overlap(splits: dict, r: Result):
    hashes = {name: {text_hash(rec) for rec in recs}
              for name, recs in splits.items()}
    names = list(splits)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = hashes[a] & hashes[b]
            if overlap:
                r.error(f"Text overlap between {a} and {b}: {len(overlap)} records")
            else:
                r.ok(f"No text overlap: {a} ↔ {b}")


def check_class_balance(splits: dict, r: Result):
    for name, recs in splits.items():
        if not recs:
            r.warn(f"{name} is empty")
            continue
        n0 = sum(1 for rec in recs if rec.get("label") == 0)
        n1 = len(recs) - n0
        ratio0 = n0 / len(recs)
        if abs(ratio0 - 0.5) > BALANCE_TOLERANCE:
            r.warn(f"{name} class balance: human={ratio0:.1%} llm={n1/len(recs):.1%} "
                   f"(expected ~50/50 ± {BALANCE_TOLERANCE:.0%})")
        else:
            r.ok(f"{name} class balance: human={ratio0:.1%} llm={n1/len(recs):.1%}")


def check_source_dominance(splits: dict, r: Result):
    for name, recs in splits.items():
        if not recs:
            continue
        counts = collections.Counter(rec.get("dataset_source", "?") for rec in recs)
        dominant = {s: n / len(recs) for s, n in counts.items()
                    if n / len(recs) > SOURCE_DOMINANCE_LIMIT}
        if dominant:
            detail = ", ".join(f"{s}={v:.0%}" for s, v in dominant.items())
            r.warn(f"{name} source dominance: {detail}")
        else:
            r.ok(f"{name} source distribution OK (no source > {SOURCE_DOMINANCE_LIMIT:.0%})")


def check_required_fields(splits: dict, r: Result):
    for name, recs in splits.items():
        bad = [i for i, rec in enumerate(recs)
               if REQUIRED_FIELDS - set(rec.keys())]
        if bad:
            missing = REQUIRED_FIELDS - set(recs[bad[0]].keys())
            r.error(f"{name}: {len(bad)} records missing required fields "
                    f"(e.g. row {bad[0]}: {missing})")
        else:
            r.ok(f"{name}: all required fields present")


def check_llm_fields(splits: dict, r: Result):
    """
    origin_model is required for ALL label=1 records (including HC3).
    generation_type: required for api_generated, optional for HC3 (pre-existing dataset).
    length_bin / temperature_style / temperature_value: required for api_generated only.
    """
    for name, recs in splits.items():
        llm = [rec for rec in recs if rec.get("label") == 1]
        if not llm:
            continue

        # Hard check: origin_model must exist in every LLM record
        bad_origin = [rec for rec in llm if not rec.get("origin_model")]
        if bad_origin:
            r.error(f"{name}: {len(bad_origin)} LLM records missing 'origin_model'")
        else:
            r.ok(f"{name}: origin_model present in all LLM records")

        # Soft check: generation_type missing → warn (HC3 records legitimately lack it)
        missing_gen_type = [rec for rec in llm if not rec.get("generation_type")]
        if missing_gen_type:
            # Show which sources are affected for transparency
            sources = set(rec.get("dataset_source", "?") for rec in missing_gen_type)
            r.warn(f"{name}: {len(missing_gen_type)} LLM records missing "
                   f"'generation_type' (sources: {sources}) "
                   f"— expected for pre-existing datasets like HC3")

        # Soft check: generation fields required for api_generated only
        api_recs = [rec for rec in llm
                    if rec.get("generation_type") == "api_generated"]
        bad_gen = [rec for rec in api_recs
                   if LLM_FIELDS_GENERATION - set(rec.keys())]
        if bad_gen:
            r.warn(f"{name}: {len(bad_gen)} api_generated records missing "
                   f"generation fields {LLM_FIELDS_GENERATION}")


def check_split_field(splits: dict, r: Result):
    for name, recs in splits.items():
        wrong = [rec for rec in recs if rec.get("split") != name]
        if wrong:
            r.error(f"{name}: {len(wrong)} records have wrong 'split' field "
                    f"(e.g. '{wrong[0].get('split')}')")
        else:
            r.ok(f"{name}: split field consistent")


def check_no_empty_text(splits: dict, r: Result):
    for name, recs in splits.items():
        empty = [i for i, rec in enumerate(recs)
                 if not (rec.get("text_clean") or "").strip()]
        if empty:
            r.error(f"{name}: {len(empty)} records have empty text_clean")
        else:
            r.ok(f"{name}: no empty text_clean")


def check_temperature_type(splits: dict, r: Result):
    """temperature_value must be float, int, or None — never a string."""
    for name, recs in splits.items():
        bad = [rec for rec in recs
               if "temperature_value" in rec
               and rec["temperature_value"] is not None
               and not isinstance(rec["temperature_value"], (int, float))]
        if bad:
            r.error(f"{name}: {len(bad)} records have string temperature_value "
                    f"(e.g. '{bad[0]['temperature_value']}') — must be float")
        else:
            r.ok(f"{name}: temperature_value type OK")


def check_claude_test_only(splits: dict, r: Result):
    # Claude must NOT be in train/val (already checked above, but also
    # positively verify they ARE in test)
    claude_in_test = [rec for rec in splits["test"] if is_claude(rec)]
    wrong_split    = [rec for rec in claude_in_test if rec.get("split") != "test"]
    if wrong_split:
        r.error(f"{len(wrong_split)} Claude records in test have split != 'test'")
    if claude_in_test:
        pct = len(claude_in_test) / len(splits["test"])
        r.ok(f"test: {len(claude_in_test):,} Claude records ({pct:.0%} of test set)")
    else:
        r.warn("test: no Claude records found — expected held-out model samples")


def check_smishing_bins(splits: dict, r: Result):
    for name, recs in splits.items():
        bad = [rec for rec in recs
               if rec.get("content_type") == "smishing"
               and rec.get("length_bin") not in (None, "short")]
        if bad:
            r.error(f"{name}: {len(bad)} smishing records with non-short length_bin")
        else:
            r.ok(f"{name}: smishing records are all short-bin (or no smishing)")

# ---------------------------------------------------------------------------
# Distribution report
# ---------------------------------------------------------------------------

def print_distribution(splits: dict) -> None:
    print("\n" + "=" * 70)
    print("  DISTRIBUTION REPORT")
    print("=" * 70)
    for name, recs in splits.items():
        if not recs:
            continue
        print(f"\n  {name.upper()} — {len(recs):,} records")
        by_ct = collections.Counter(
            (rec.get("content_type", "?"), rec["label"]) for rec in recs
        )
        cts = sorted({k[0] for k in by_ct})
        print(f"    {'content_type':<35} {'human':>7} {'llm':>7} {'total':>7}")
        print(f"    {'-'*35} {'-'*7} {'-'*7} {'-'*7}")
        for ct in cts:
            n0 = by_ct.get((ct, 0), 0)
            n1 = by_ct.get((ct, 1), 0)
            print(f"    {ct:<35} {n0:>7,} {n1:>7,} {n0+n1:>7,}")

        llm_recs = [rec for rec in recs if rec["label"] == 1]
        if llm_recs:
            mc = collections.Counter(
                rec.get("origin_model", "?") for rec in llm_recs
            )
            print(f"\n    LLM origin models:")
            for model, n in sorted(mc.items(), key=lambda x: -x[1]):
                print(f"      {model:<50} {n:>5,}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(strict: bool = False) -> None:
    logger.info("=" * 60)
    logger.info("Stage 5 — Dataset Validation")
    logger.info("=" * 60)

    splits: dict[str, list[dict]] = {}
    for name, path in SPLITS.items():
        if not path.exists():
            logger.error(f"  {path} not found — run assemble.py first")
            return
        recs = load_jsonl(path)
        splits[name] = recs
        logger.info(f"  Loaded {name}: {len(recs):,}")

    result = Result()

    logger.info("\nRunning checks …")
    check_no_claude_in_train_val(splits, result)
    check_no_overlap(splits, result)
    check_class_balance(splits, result)
    check_source_dominance(splits, result)
    check_required_fields(splits, result)
    check_llm_fields(splits, result)
    check_split_field(splits, result)
    check_no_empty_text(splits, result)
    check_temperature_type(splits, result)
    check_claude_test_only(splits, result)
    check_smishing_bins(splits, result)

    print_distribution(splits)

    print("\n" + "=" * 60)
    print("  VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Passed   : {len(result.passed)}")
    print(f"  Warnings : {len(result.warnings)}")
    print(f"  Errors   : {len(result.errors)}")

    if result.warnings:
        print("\n  Warnings:")
        for w in result.warnings:
            print(f"    ⚠  {w}")

    if result.errors:
        print("\n  Errors:")
        for e in result.errors:
            print(f"    ✗  {e}")
        print("\n  Dataset NOT VALID. Fix errors before training.")
        if strict:
            raise SystemExit(1)
    else:
        print("\n  Dataset VALID ✓")
        print("  Next: uv run python -m src.models.train")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate assembled dataset (Stage 5)")
    parser.add_argument("--strict", action="store_true",
                        help="Exit with code 1 if any errors found")
    args = parser.parse_args()
    main(strict=args.strict)