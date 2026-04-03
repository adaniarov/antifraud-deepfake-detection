"""
generate.py — Stage 2 (LLM): generate synthetic fraud/legit texts via API.

Outputs:
  data/raw/llm_fraud/   label=1, LLM-generated fraud-related texts
  data/raw/llm_legit/   label=1, LLM-generated legit texts

Models:
  gpt4o   — primary generator, train/val  (100 samples/batch)
  mistral — primary generator, train/val  (100 samples/batch)
  claude  — held-out test set, NEVER in train (80 samples/batch, split="test")

Content types:
  T1 phishing           (3 bins)  | T2 smishing        (short only)
  T3 social_engineering (3 bins)  | T4 scam_419         (3 bins)
  T5 bank_notification  (3 bins)  | T6 financial_review (3 bins)

Batches per model: (5 types × 3 bins + 1 type × 1 bin) × 2 temp = 32
Total: 96 batches across 3 models

Usage:
    uv run python -m src.generation.generate
    uv run python -m src.generation.generate --model gpt4o
    uv run python -m src.generation.generate --type T1
    uv run python -m src.generation.generate --dry-run
    uv run python -m src.generation.generate --model claude --dry-run
"""

import argparse
import csv
import io
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


# ---------------------------------------------------------------------------
# Temperature — passed as real API parameter, never as prompt instruction
# ---------------------------------------------------------------------------

TEMPERATURE_MAP = {"low": 0.2, "high": 0.8}


# ---------------------------------------------------------------------------
# Content types T1–T6
# ---------------------------------------------------------------------------

# (type_id, prompt_file, group, csv_id_prefix, content_type_name)
CONTENT_TYPES: list[tuple[str, str, str, str, str]] = [
    ("T1", "T1_phishing_email.txt",      "llm_fraud", "PHISH",   "phishing"),
    ("T2", "T2_smishing.txt",            "llm_fraud", "SMISH",   "smishing"),
    ("T3", "T3_social_engineering.txt",  "llm_fraud", "SOCENG",  "social_engineering"),
    ("T4", "T4_scam_419.txt",            "llm_fraud", "SCAM",    "scam_419"),
    ("T5", "T5_bank_notification.txt",   "llm_legit", "BANKNOT", "bank_notification"),
    ("T6", "T6_financial_review.txt",    "llm_legit", "FINREV",  "financial_review"),
]

LENGTH_BINS: list[str] = ["short", "medium", "long"]
# short:  20–100 tokens  (~80–400 chars)
# medium: 101–200 tokens (~401–800 chars)
# long:   201–400 tokens (~801–1600 chars)

PROMPTS_DIR = Path("data/prompts")

OUT_DIRS = {
    "llm_fraud": Path("data/raw/llm_fraud"),
    "llm_legit": Path("data/raw/llm_legit"),
}

CHECKPOINT_FILE = Path("data/raw/generation_checkpoint.json")
MAX_RETRIES = 3

N_SAMPLES: dict[str, int] = {
    "gpt4o":   100,   # train/val
    "mistral": 100,   # train/val
    "claude":  80,    # test only (held-out)
}

MODEL_IDS: dict[str, str] = {
    "gpt4o":   "gpt-4o-mini",
    "mistral": "mistralai/mistral-small-3.2-24b-instruct",   # OpenRouter model id
    "claude":  "anthropic/claude-haiku-4-5",       # OpenRouter model id (held-out test)
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ---------------------------------------------------------------------------
# Chunked generation config
#
# Each model has a hard output-token limit:
#   gpt-4o-mini  : 16 384
#   mistral-small: 32 768
#   claude-haiku : 8 192
#
# We split N_SAMPLES into chunks small enough to fit safely within the limit.
# max_tokens is set per (model, bin) with headroom.
#
# Rough cost per record:
#   short  ~60  output tokens  (SMS / one-liner)
#   medium ~180 output tokens  (paragraph email)
#   long   ~450 output tokens  (full email / review)
# Plus CSV overhead ~15 tokens/row.
# ---------------------------------------------------------------------------

# chunk_size: how many records to request per single API call
CHUNK_SIZES: dict[str, dict[str, int]] = {
    #               short  medium  long
    "gpt4o":   {"short": 100, "medium": 25, "long": 10},
    "mistral": {"short": 100, "medium": 50, "long": 20},
    "claude":  {"short": 50,  "medium": 20, "long":  8},
}

# max_tokens passed to the API for each (model, bin) combination
MAX_TOKENS_MAP: dict[str, dict[str, int]] = {
    "gpt4o":   {"short": 10000, "medium": 6000, "long": 7000},
    "mistral": {"short": 10000, "medium": 12000, "long": 14000},
    "claude":  {"short":  6000, "medium":  5000, "long":  6000},
}


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

_SEP_RE = re.compile(r"^─{10,}$")   # separator: line of ─ (U+2500), len ≥ 10


def load_prompt_file(path: Path) -> tuple[str, str]:
    """
    Parse a prompt file with the structure:

        SYSTEM PROMPT
        ────────────────────
        <system text>
        ────────────────────
        USER PROMPT TEMPLATE
        ────────────────────
        <user template text with {placeholders}>
        ────────────────────

    Returns (system_prompt, user_prompt_template), both stripped.
    Separator = line consisting entirely of ─ (U+2500), length ≥ 10.
    """
    lines = path.read_text(encoding="utf-8").splitlines()

    def extract_block(keyword: str) -> str:
        found_header = False
        sep_count = 0
        content: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not found_header:
                if keyword in stripped:
                    found_header = True
                continue
            if _SEP_RE.match(stripped):
                sep_count += 1
                if sep_count == 2:
                    break        # end of block
                continue         # skip opening separator line
            if sep_count == 1:
                content.append(line)
        return "\n".join(content).strip()

    system = extract_block("SYSTEM PROMPT")
    user   = extract_block("USER PROMPT TEMPLATE")
    return system, user


# Load all prompts once at module import; missing files produce warnings only
# (batch execution will fail gracefully if a prompt is still absent).
PROMPTS: dict[str, tuple[str, str]] = {}


def _load_all_prompts() -> None:
    for type_id, prompt_file, _, _, _ in CONTENT_TYPES:
        path = PROMPTS_DIR / prompt_file
        if path.exists():
            try:
                PROMPTS[type_id] = load_prompt_file(path)
                logger.debug(f"Loaded prompt: {path}")
            except Exception as exc:
                logger.warning(f"Failed to parse {path}: {exc}")
        else:
            logger.warning(f"Prompt file not found: {path}")


_load_all_prompts()


# ---------------------------------------------------------------------------
# User prompt construction
# ---------------------------------------------------------------------------

def build_user_prompt(template: str, batch: dict) -> str:
    """
    Substitute batch parameters into the user prompt template.

    Available placeholders (extra keys are ignored by str.format):
      {n_samples}, {target_bin}, {temperature_style}, {start_id}
      {n_positive}, {n_negative}, {n_mixed}   — used by T6 sentiment split
      {pct_pos}, {pct_neg}
    """
    n = batch["n_samples"]
    n_positive = round(n * 0.40)
    n_negative = round(n * 0.35)
    n_mixed    = n - n_positive - n_negative
    try:
        return template.format(
            n_samples=n,
            target_bin=batch["length_bin"],
            temperature_style=batch["temp_style"],
            start_id=batch["start_id"],
            n_positive=n_positive,
            n_negative=n_negative,
            n_mixed=n_mixed,
            pct_pos=40,
            pct_neg=35,
        )
    except KeyError as exc:
        logger.warning(f"Unknown placeholder in prompt template: {exc}. Returning template as-is.")
        return template


# ---------------------------------------------------------------------------
# Generation plan
# ---------------------------------------------------------------------------

def build_generation_plan() -> list[dict]:
    """
    Build the full list of batches.

    T2 (smishing) uses only the 'short' bin — SMS is inherently short.
    All other types use all three bins.

    Returns list of dicts, each with:
      batch_id, model, type_id, content_type, group,
      length_bin, temp_style, n_samples, start_id
    """
    plan: list[dict] = []
    global_id = 0

    for model_key in ("gpt4o", "mistral", "claude"):
        for type_id, _, group, _, content_type in CONTENT_TYPES:
            allowed_bins = ["short"] if type_id == "T2" else LENGTH_BINS
            for length_bin in allowed_bins:
                for temp_style in ("low", "high"):
                    batch_id = f"{model_key}__{type_id}__{length_bin}__{temp_style}"
                    plan.append({
                        "batch_id":     batch_id,
                        "model":        model_key,
                        "type_id":      type_id,
                        "content_type": content_type,
                        "group":        group,
                        "length_bin":   length_bin,
                        "temp_style":   temp_style,
                        "n_samples":    N_SAMPLES[model_key],
                        "start_id":     global_id,
                    })
                    global_id += N_SAMPLES[model_key]

    return plan


GENERATION_PLAN = build_generation_plan()


# ---------------------------------------------------------------------------
# API clients (lazy initialisation — key checked only when model is first used)
# ---------------------------------------------------------------------------

_openai_client     = None
_openrouter_client = None


def _get_openai():
    global _openai_client
    if _openai_client is None:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set. Export it before running:\n"
                "  export OPENAI_API_KEY=sk-..."
            )
        import openai  # type: ignore
        _openai_client = openai.OpenAI(api_key=key)
    return _openai_client


def _get_openrouter():
    global _openrouter_client
    if _openrouter_client is None:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY not set. Export it before running:\n"
                "  export OPENROUTER_API_KEY=sk-or-..."
            )
        import openai  # type: ignore
        _openrouter_client = openai.OpenAI(
            api_key=key,
            base_url=OPENROUTER_BASE_URL,
        )
    return _openrouter_client


# ---------------------------------------------------------------------------
# API call functions — temperature is a real API parameter, never prompt text
# ---------------------------------------------------------------------------

def call_openai(
    system_prompt: str,
    user_prompt: str,
    temperature_style: str,
    max_tokens: int,
) -> str:
    temp = TEMPERATURE_MAP[temperature_style]
    response = _get_openai().chat.completions.create(
        model=MODEL_IDS["gpt4o"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=temp,       # real API parameter
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def call_mistral(
    system_prompt: str,
    user_prompt: str,
    temperature_style: str,
    max_tokens: int,
) -> str:
    temp = TEMPERATURE_MAP[temperature_style]
    response = _get_openrouter().chat.completions.create(
        model=MODEL_IDS["mistral"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=temp,       # real API parameter
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def call_claude(
    system_prompt: str,
    user_prompt: str,
    temperature_style: str,
    max_tokens: int,
) -> str:
    temp = TEMPERATURE_MAP[temperature_style]
    response = _get_openrouter().chat.completions.create(
        model=MODEL_IDS["claude"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=temp,       # real API parameter
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


API_CALLERS = {
    "gpt4o":   call_openai,
    "mistral": call_mistral,
    "claude":  call_claude,
}


# ---------------------------------------------------------------------------
# CSV response parser
# ---------------------------------------------------------------------------

def parse_csv_response(raw_text: str, batch: dict) -> list[dict]:
    """
    Parse the model's CSV response into JSONL records.

    Expected columns (found by name from header row):
        id, theme, target_bin, temperature_style, text_raw, estimated_tokens

    Falls back to positional order if no header row is detected.
    Falls back to 'text' column if 'text_raw' is absent.
    Rows missing any text field are skipped with a warning.

    split="test" for claude (held-out), None for all other models.
    """
    model_key    = batch["model"]
    content_type = batch["content_type"]
    length_bin   = batch["length_bin"]
    temp_style   = batch["temp_style"]
    n_samples    = batch["n_samples"]
    temp_value   = TEMPERATURE_MAP[temp_style]
    split        = "test" if model_key == "claude" else None

    # Positional fallback column order (matches prompt spec)
    POSITIONAL_HEADER = ["id", "theme", "target_bin", "temperature_style",
                         "text_raw", "estimated_tokens"]

    records: list[dict] = []
    header: list[str] | None = None

    try:
        reader = csv.reader(io.StringIO(raw_text.strip()))
        for row in reader:
            if not row:
                continue

            # Detect header row on first non-empty row
            if header is None:
                first = row[0].strip().lower()
                if first in ("id", "#", ""):
                    header = [col.strip().lower() for col in row]
                    continue
                else:
                    # No header — use positional fallback
                    header = POSITIONAL_HEADER

            # Map row cells to column names
            row_dict = {
                header[i]: row[i].strip()
                for i in range(min(len(header), len(row)))
            }

            # Extract text (text_raw preferred, text as fallback)
            text = row_dict.get("text_raw") or row_dict.get("text", "")
            text = text.strip()
            if not text:
                logger.warning(f"Row missing text_raw/text, skipping: {row_dict}")
                continue

            # Extract sample id — strip non-numeric prefix (e.g. "PHISH_07" -> 7)
            raw_id = row_dict.get("id", "0").strip()
            numeric_id = re.sub(r"^[^0-9]*", "", raw_id)
            try:
                sample_id = int(numeric_id) if numeric_id else 0
            except ValueError:
                logger.warning(f"Non-integer id {raw_id!r}, skipping row")
                continue

            theme = row_dict.get("theme", "")

            records.append({
                "text":              text,
                "label":             1,
                "label_str":         "llm",
                "origin_model":      MODEL_IDS[model_key],
                "content_type":      content_type,
                "dataset_source":    f"generated_{model_key}",
                "char_length":       len(text),
                "length_bin":        length_bin,
                "temperature_style": temp_style,
                "temperature_value": temp_value,
                "generation_type":   "api_generated",
                "theme":             theme,
                "sample_id":         sample_id,
                "split":             split,
            })

    except Exception as exc:
        logger.error(f"CSV parse error in batch {batch['batch_id']}: {exc}")

    if len(records) < n_samples:
        logger.warning(
            f"  Parsed {len(records)}/{n_samples} records for {batch['batch_id']}"
        )

    return records


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint() -> tuple[set[str], dict[str, int]]:
    """Returns (completed_batch_ids, actual_counts {batch_id: n_records})."""
    if CHECKPOINT_FILE.exists():
        data = json.loads(CHECKPOINT_FILE.read_text())
        return set(data.get("completed", [])), data.get("actual_counts", {})
    return set(), {}


def save_checkpoint(completed: set[str], actual_counts: dict[str, int]) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(
        json.dumps(
            {"completed": sorted(completed), "actual_counts": actual_counts},
            indent=2,
        )
    )


def count_existing_records(batch: dict) -> int:
    """Count lines in the JSONL file for this batch (0 if file absent)."""
    type_id      = batch["type_id"]
    content_type = batch["content_type"]
    model_key    = batch["model"]
    length_bin   = batch["length_bin"]
    temp_style   = batch["temp_style"]
    group        = batch["group"]

    path = OUT_DIRS[group] / f"{type_id}__{content_type}__{model_key}__{length_bin}__{temp_style}.jsonl"
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


# ---------------------------------------------------------------------------
# Record persistence
# ---------------------------------------------------------------------------

def save_records(records: list[dict], batch: dict) -> Path:
    """Append records to JSONL file. Filename encodes all batch dimensions."""
    type_id      = batch["type_id"]
    content_type = batch["content_type"]
    model_key    = batch["model"]
    length_bin   = batch["length_bin"]
    temp_style   = batch["temp_style"]
    group        = batch["group"]

    out_dir = OUT_DIRS[group]
    out_dir.mkdir(parents=True, exist_ok=True)

    # e.g. T1__phishing__gpt4o__short__low.jsonl
    fname    = f"{type_id}__{content_type}__{model_key}__{length_bin}__{temp_style}.jsonl"
    out_path = out_dir / fname

    with open(out_path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return out_path


# ---------------------------------------------------------------------------
# Batch runner with exponential backoff retry
# ---------------------------------------------------------------------------

def _call_with_retry(
    caller,
    system_prompt: str,
    user_prompt: str,
    temp_style: str,
    max_tokens: int,
    batch_id: str,
) -> str:
    """Single API call with exponential backoff retry. Returns raw response text."""
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return caller(system_prompt, user_prompt, temp_style, max_tokens)
        except Exception as exc:
            last_error = exc
            is_rate_limit = "429" in str(exc) or "rate" in str(exc).lower()
            wait = 30 * attempt if is_rate_limit else 2 ** attempt
            logger.warning(
                f"  Attempt {attempt}/{MAX_RETRIES} failed ({batch_id}): {exc}. "
                f"Retrying in {wait}s..."
            )
            time.sleep(wait)
    raise RuntimeError(
        f"Batch {batch_id} failed after {MAX_RETRIES} attempts: {last_error}"
    )


def run_batch(batch: dict, dry_run: bool = False) -> list[dict]:
    """
    Execute one generation batch, splitting into chunks to fit model token limits.

    Chunk sizes and max_tokens are looked up from CHUNK_SIZES / MAX_TOKENS_MAP
    by (model, length_bin).  Multiple API calls are made when n_samples > chunk_size;
    all records are accumulated and returned together.

    Returns parsed records; empty list on total failure or missing prompt.
    """
    model_key  = batch["model"]
    type_id    = batch["type_id"]
    length_bin = batch["length_bin"]
    n_samples  = batch["n_samples"]

    if type_id not in PROMPTS:
        logger.error(
            f"No prompt loaded for {type_id} — "
            f"create prompts/{type_id}_*.txt and re-run."
        )
        return []

    system_prompt, user_template = PROMPTS[type_id]

    chunk_size = CHUNK_SIZES[model_key][length_bin]
    max_tokens = MAX_TOKENS_MAP[model_key][length_bin]

    # Build list of chunk sizes that sum to n_samples
    chunks: list[int] = []
    remaining = n_samples
    while remaining > 0:
        chunks.append(min(chunk_size, remaining))
        remaining -= chunk_size

    if dry_run:
        split = "test" if model_key == "claude" else None
        logger.info(
            f"[DRY RUN] {batch['batch_id']} | "
            f"temp={batch['temp_style']} ({TEMPERATURE_MAP[batch['temp_style']]}) | "
            f"n={n_samples} | chunks={chunks} | max_tokens={max_tokens} | split={split}"
        )
        logger.debug(f"  system: {system_prompt[:100]}...")
        return []

    caller = API_CALLERS[model_key]
    all_records: list[dict] = []
    current_id = batch["start_id"]

    for chunk_idx, chunk_n in enumerate(chunks, 1):
        chunk_batch = dict(batch)
        chunk_batch["n_samples"] = chunk_n
        chunk_batch["start_id"]  = current_id

        user_prompt = build_user_prompt(user_template, chunk_batch)

        logger.info(
            f"  chunk {chunk_idx}/{len(chunks)} | {batch['batch_id']} | "
            f"n={chunk_n} | max_tokens={max_tokens} | "
            f"temp={batch['temp_style']} ({TEMPERATURE_MAP[batch['temp_style']]})"
        )

        try:
            raw_response = _call_with_retry(
                caller, system_prompt, user_prompt,
                batch["temp_style"], max_tokens, batch["batch_id"],
            )
            records = parse_csv_response(raw_response, chunk_batch)
            logger.info(f"  -> parsed {len(records)}/{chunk_n} records")
            all_records.extend(records)
            current_id += chunk_n
        except RuntimeError as exc:
            logger.error(str(exc))
            break   # stop further chunks on total failure

    logger.info(
        f"  batch total: {len(all_records)}/{n_samples} records "
        f"({len(chunks)} chunk(s))"
    )
    return all_records


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(
    model_filter: str | None = None,
    type_filter: str | None = None,
    dry_run: bool = False,
    limit: int | None = None,
    top_up: bool = False,
) -> None:
    plan = list(GENERATION_PLAN)

    if model_filter:
        plan = [b for b in plan if b["model"] == model_filter]
    if type_filter:
        plan = [b for b in plan if b["type_id"] == type_filter]
    if limit:
        plan = plan[:limit]

    completed, actual_counts = load_checkpoint()

    if top_up:
        # Find batches that are "completed" but have fewer records than target
        top_up_batches = []
        for b in plan:
            bid = b["batch_id"]
            if bid not in completed:
                continue
            existing = count_existing_records(b)
            target   = b["n_samples"]
            if existing < target:
                shortage = target - existing
                top_up_batches.append((b, existing, shortage))

        if not top_up_batches:
            logger.info("All completed batches are at full count. Nothing to top up.")
            return

        logger.info("=" * 60)
        logger.info("Stage 2 — Top-up mode: filling incomplete batches")
        logger.info("=" * 60)
        for b, existing, shortage in top_up_batches:
            logger.info(f"  {b['batch_id']}: {existing}/{b['n_samples']} — need {shortage} more")

        total_generated = 0
        for b, existing, shortage in top_up_batches:
            top_up_batch = dict(b)
            top_up_batch["n_samples"] = shortage
            top_up_batch["start_id"]  = b["start_id"] + existing

            logger.info(f"\nTop-up: {b['batch_id']} (+{shortage})")
            records = run_batch(top_up_batch, dry_run=dry_run)

            if records:
                out_path = save_records(records, b)
                logger.info(f"  Saved {len(records)} records -> {out_path}")
                total_generated += len(records)
                if not dry_run:
                    new_count = existing + len(records)
                    actual_counts[b["batch_id"]] = new_count
                    save_checkpoint(completed, actual_counts)
            else:
                logger.warning(f"  Top-up produced no records for {b['batch_id']}")

        logger.info("\n" + "=" * 60)
        logger.info(f"Top-up complete. Added {total_generated} records.")
        logger.info("=" * 60)
        return

    # --- Normal generation mode ---
    pending = [b for b in plan if b["batch_id"] not in completed]
    estimated_new = sum(b["n_samples"] for b in pending)

    total_plan = GENERATION_PLAN
    gpt4o_n   = sum(1 for b in total_plan if b["model"] == "gpt4o")
    mistral_n = sum(1 for b in total_plan if b["model"] == "mistral")
    claude_n  = sum(1 for b in total_plan if b["model"] == "claude")

    logger.info("=" * 60)
    logger.info("Stage 2 — LLM Text Generation")
    logger.info(
        f"Models:  gpt4o (train/val, 100/batch, {gpt4o_n} batches), "
        f"mistral (train/val, 100/batch, {mistral_n} batches),\n"
        f"         {MODEL_IDS['claude']} "
        f"(TEST ONLY / held-out, 80/batch, {claude_n} batches)"
    )
    logger.info(
        f"Batches: {len(plan)} total | {len(completed)} done | {len(pending)} pending"
    )
    logger.info(f"Estimated new records: {estimated_new}")
    logger.info("=" * 60)

    total_generated = 0

    for i, batch in enumerate(plan, 1):
        batch_id = batch["batch_id"]

        if batch_id in completed:
            logger.info(f"[{i}/{len(plan)}] SKIP (checkpoint): {batch_id}")
            continue

        logger.info(f"\n[{i}/{len(plan)}] {batch_id}")
        records = run_batch(batch, dry_run=dry_run)

        if records:
            out_path = save_records(records, batch)
            logger.info(f"  Saved {len(records)} records -> {out_path}")
            total_generated += len(records)
            if not dry_run:
                completed.add(batch_id)
                actual_counts[batch_id] = len(records)
                save_checkpoint(completed, actual_counts)
        elif not dry_run:
            logger.warning(f"  Batch {batch_id} produced no records — NOT marked as completed")

    logger.info("\n" + "=" * 60)
    logger.info(f"Generation complete. Total new records: {total_generated}")
    logger.info("=" * 60)
    logger.info("Next step: uv run python -m src.preprocessing.preprocess")


if __name__ == "__main__":
    _type_ids = [t[0] for t in CONTENT_TYPES]

    parser = argparse.ArgumentParser(
        description="Generate synthetic LLM texts for the antifraud deepfake dataset"
    )
    parser.add_argument(
        "--model",
        choices=["gpt4o", "mistral", "claude"],
        default=None,
        help="Run only batches for this model (default: all models)",
    )
    parser.add_argument(
        "--type",
        choices=_type_ids,
        default=None,
        dest="type_filter",
        metavar="TYPE",
        help=f"Run only batches for this content type: {', '.join(_type_ids)}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print batch plan without calling any API",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Run only the first N batches (useful for testing)",
    )
    parser.add_argument(
        "--top-up",
        action="store_true",
        help="Fill incomplete batches (actual < target) without regenerating complete ones",
    )
    args = parser.parse_args()
    main(
        model_filter=args.model,
        type_filter=args.type_filter,
        dry_run=args.dry_run,
        limit=args.limit,
        top_up=args.top_up,
    )
