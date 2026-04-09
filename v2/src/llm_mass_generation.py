"""
Mass LLM generation for Core Dataset 2 — resumable, OpenAI-compatible chat APIs.

Why a separate module (not only notebook cells)
-----------------------------------------------
- Three notebooks share one implementation: job grid, CRC split seen_openai/seen_mistral,
  holdout_claude grid, QC (format / masking / length_bin), near-dedup, retries, tqdm,
  append-only JSONL + resume by ``generation_job_id``.
- Editing one file avoids copy-paste drift and keeps behaviour identical across providers.

What it does
------------
- Loads the five prompt JSON specs from ``v2/data/prompts/``.
- Builds deterministic jobs per lane; skips jobs already present in the output JSONL.
- Calls the chat API (OpenAI, Mistral, or OpenRouter) and writes valid rows to
  ``v2/data/interim/llm-generated/core_llm_<lane>_<model>.jsonl``.
- Optional ``MassGenConfig.max_workers > 1``: parallel API calls (thread pool); QC/dedup/file
  writes stay serialized under locks.

Used by notebooks:
  11_mass_generation_openai.ipynb
  12_mass_generation_mistral.ipynb
  13_mass_generation_claude_openrouter.ipynb

Design: v2/docs/dataset_design_final.md §6.2–6.9 (seen generators train/val; Claude holdout test only).
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import uuid
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Literal

from openai import APIStatusError, OpenAI, RateLimitError
from tqdm.auto import tqdm
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# ── length bins (token thresholds) ───────────────────────────────────────────
# Imported lazily in run() after sys.path is set by the notebook.


FAMILIES_ORDERED = [
    "phishing_email",
    "advance_fee_scam_email",
    "fraud_sms_deceptive",
    "legitimate_email",
    "legitimate_sms",
]

# Per README.md Generation Contract (sampling policy)
BIN_SPLITS: dict[str, list[str]] = {
    "phishing_email": ["short"] * 20 + ["medium"] * 50 + ["long"] * 30,
    "advance_fee_scam_email": ["medium"] * 40 + ["long"] * 60,
    "fraud_sms_deceptive": ["short"] * 15 + ["medium"] * 85,
    "legitimate_email": ["short"] * 55 + ["medium"] * 35 + ["long"] * 10,
    "legitimate_sms": ["short"] * 100,
}

WRAPPER_PREFIXES = (
    "here is",
    "here's",
    "certainly",
    "sure!",
    "sure,",
    "as requested",
    "of course",
    "absolutely",
    "below is",
    "the following",
)

_SMS_EMOJI_RE = re.compile(
    r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F600-\U0001F64F\U0001F1E0-\U0001F1FF]+",
    flags=re.UNICODE,
)


def format_qc(text: str, channel: str | None = None) -> list[str]:
    issues: list[str] = []
    ntok = len(text.split())
    if not text or ntok < 5:
        issues.append("empty_or_too_short")
    if re.search(r"^#{1,6}\s", text, re.MULTILINE):
        issues.append("markdown_header")
    if re.search(r"\*\*|__|\*[^*]|_[^_]", text):
        issues.append("markdown_emphasis")
    if re.search(r"<[a-zA-Z][^>]*>", text):
        issues.append("html_tag")
    if any(text.lower().lstrip().startswith(w) for w in WRAPPER_PREFIXES):
        issues.append("wrapper_phrase")
    if channel == "sms" and _SMS_EMOJI_RE.search(text):
        issues.append("emoji_in_sms")
    return issues


def masking_qc(text: str) -> list[str]:
    issues: list[str] = []
    if re.search(r"https?://", text):
        issues.append("unmasked_url")
    if re.search(r"\b[\w.+-]+@[\w-]+\.\w+", text):
        issues.append("unmasked_email")
    if re.search(r"\b(\+?\d[\d\s\-().]{6,}\d)\b", text):
        issues.append("unmasked_phone")
    return issues


def length_qc(text: str, channel: str, family_bins: list[str], compute_length_bin) -> dict:
    token_count = len(text.split())
    actual_bin = compute_length_bin(token_count, channel)
    accepted = actual_bin in family_bins
    return {
        "token_count": token_count,
        "actual_bin": actual_bin,
        "accepted": accepted,
    }


def validate(text: str, channel: str, family_bins: list[str], compute_length_bin) -> dict:
    fmt = format_qc(text, channel=channel)
    mask = masking_qc(text)
    lqc = length_qc(text, channel, family_bins, compute_length_bin)
    issues = fmt + mask + ([] if lqc["accepted"] else [f"bin_out_of_family:{lqc['actual_bin']}"])
    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "token_count": lqc["token_count"],
        "actual_bin": lqc["actual_bin"],
    }


def is_near_duplicate(text: str, seen_texts: list[str], threshold: float = 0.85) -> bool:
    for s in seen_texts:
        if SequenceMatcher(None, text, s).ratio() > threshold:
            return True
    return False


def slug_model(model: str) -> str:
    return model.replace("/", "_").replace(":", "_").replace(".", "_")


def load_specs(prompts_dir: Path) -> dict[str, dict]:
    specs: dict[str, dict] = {}
    for name in FAMILIES_ORDERED:
        path = prompts_dir / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing prompt spec: {path}")
        specs[name] = json.loads(path.read_text(encoding="utf-8"))
    return specs


_SEEN_PARTITION_MOD = 10_000


def _seen_generator_slot(job_base: str, openai_seen_share: float) -> Literal["openai", "mistral"]:
    """Deterministic OpenAI vs Mistral assignment for one logical job (same base string → same slot).

    *openai_seen_share* is the approximate fraction of all seen jobs that go to OpenAI
    (remainder to Mistral). Uses a fixed 10_000-bucket quantile on crc32 for stability.
    """
    share = max(0.0, min(1.0, float(openai_seen_share)))
    h = zlib.crc32(job_base.encode("utf-8")) % _SEEN_PARTITION_MOD
    cut = int(share * _SEEN_PARTITION_MOD)
    if share <= 0.0:
        return "mistral"
    if share >= 1.0:
        return "openai"
    return "openai" if h < cut else "mistral"


def iter_jobs(
    specs: dict[str, dict],
    samples_per_subtype: int,
    *,
    mode: Literal["seen_openai", "seen_mistral", "holdout_claude"],
    openai_seen_share: float = 0.6,
) -> Iterable[tuple[str, str, str, int, str]]:
    """
    Yields (family, subtype, target_bin, idx, generation_job_id).

    Seen OpenAI vs Mistral: partition the same logical grid deterministically so the two
    notebooks never claim the same (family, subtype, bin, idx) slot. *openai_seen_share*
    is the approximate OpenAI fraction (e.g. 0.6 for 60/40). **Must match** in both seen
    notebooks (or the same env ``OPENAI_SEEN_SHARE``).

    Holdout Claude: full grid, independent job_id prefix (separate test-only corpus).
    """
    for family in FAMILIES_ORDERED:
        spec = specs[family]
        pool = BIN_SPLITS[family]
        for subtype in spec["subtypes"]:
            for idx in range(samples_per_subtype):
                target_bin = pool[idx % len(pool)]
                base = f"{family}||{subtype}||{target_bin}||{idx}"

                if mode == "holdout_claude":
                    yield family, subtype, target_bin, idx, f"holdout_claude||{base}"
                    continue

                slot = _seen_generator_slot(base, openai_seen_share)
                if mode == "seen_openai" and slot != "openai":
                    continue
                if mode == "seen_mistral" and slot != "mistral":
                    continue
                yield family, subtype, target_bin, idx, f"{mode}||{base}"


def load_done_job_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            jid = rec.get("generation_job_id")
            if jid:
                done.add(jid)
    return done


def load_existing_texts(path: Path) -> list[str]:
    if not path.is_file():
        return []
    texts: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = rec.get("text")
            if isinstance(t, str) and t:
                texts.append(t)
    return texts


@dataclass
class MassGenConfig:
    """Notebook fills this and calls run_mass_generation."""

    base: Path
    lane: str  # "seen_openai" | "seen_mistral" | "holdout_claude"
    api_key_env: str
    model: str
    origin_model: str  # e.g. openai/gpt-4o-mini, mistral/mistral-small-latest, anthropic/claude-3-5-haiku
    split: str  # "seen" or "test" (per dataset_design_final.md §6.9)
    samples_per_subtype: int = 400
    temperature: float = 0.9
    max_tokens: int = 900
    max_retries: int = 3
    # Concurrent HTTP requests per lane. API + validate run in parallel; near-dedup + JSONL write stay locked.
    max_workers: int = 1
    openai_base_url: str | None = None  # None → api.openai.com; or https://openrouter.ai/api/v1
    default_headers: dict[str, str] | None = None  # OpenRouter: HTTP-Referer, X-Title on the client
    # Seen lanes only: fraction of jobs assigned to OpenAI (Mistral gets 1 − this). Same value in 11 & 12.
    openai_seen_share: float = 0.6


def run_mass_generation(cfg: MassGenConfig) -> Path:
    sys.path.insert(0, str(cfg.base))
    from config import length_bin as compute_length_bin

    prompts_dir = cfg.base / "data" / "prompts"
    out_dir = cfg.base / "data" / "interim" / "llm-generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get(cfg.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(
            f"Set {cfg.api_key_env} in the environment (e.g. v2/.env or repo-root .env). "
            "See v2/.env.example."
        )

    client_kw: dict = {"api_key": api_key}
    if cfg.openai_base_url:
        client_kw["base_url"] = cfg.openai_base_url
    if cfg.default_headers:
        client_kw["default_headers"] = cfg.default_headers
    client = OpenAI(**client_kw)

    specs = load_specs(prompts_dir)
    model_slug = slug_model(cfg.model)
    out_name = f"core_llm_{cfg.lane}_{model_slug}.jsonl"
    out_path = out_dir / out_name

    done = load_done_job_ids(out_path)
    seen_texts = load_existing_texts(out_path)

    if cfg.lane not in ("seen_openai", "seen_mistral", "holdout_claude"):
        raise ValueError(f"Unknown lane: {cfg.lane}")

    if cfg.lane == "holdout_claude":
        jobs = list(iter_jobs(specs, cfg.samples_per_subtype, mode=cfg.lane))  # type: ignore[arg-type]
    else:
        jobs = list(
            iter_jobs(
                specs,
                cfg.samples_per_subtype,
                mode=cfg.lane,  # type: ignore[arg-type]
                openai_seen_share=cfg.openai_seen_share,
            )
        )

    pending = [(f, s, b, i, jid) for f, s, b, i, jid in jobs if jid not in done]

    print(f"Output:        {out_path}")
    print(f"Lane:          {cfg.lane}")
    print(f"Model:         {cfg.model}")
    print(f"origin_model:  {cfg.origin_model}")
    print(f"split:         {cfg.split}")
    if cfg.lane in ("seen_openai", "seen_mistral"):
        print(f"openai_seen_share (seen split): {cfg.openai_seen_share}")
    print(f"max_workers:   {cfg.max_workers}")
    print(f"Jobs total:    {len(jobs)}")
    print(f"Already done:  {len(done)}")
    print(f"Pending:       {len(pending)}")

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIStatusError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=45),
    )
    def _api_call(system_msg: str, user_msg: str) -> str:
        response = client.chat.completions.create(
            model=cfg.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
        return (response.choices[0].message.content or "").strip()

    n_in_file = len(done)
    total_lane_jobs = len(jobs)
    state_lock = threading.Lock()
    log_lock = threading.Lock()
    counter = {"n": n_in_file}

    with out_path.open("a", encoding="utf-8") as fout:
        bar = tqdm(
            total=len(pending),
            desc=f"{cfg.lane}",
            unit="job",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )
        bar.set_postfix_str(
            f"в_файле={counter['n']}/{total_lane_jobs} осталось_всего={max(0, total_lane_jobs - counter['n'])}",
            refresh=False,
        )

        def _log_line(msg: str) -> None:
            with log_lock:
                bar.write(msg)
                if not msg.endswith("\n"):
                    bar.write("\n")

        def process_job(job: tuple[str, str, str, int, str]) -> None:
            family, subtype, target_bin, idx, job_id = job
            spec = specs[family]
            fam = spec["scenario_family"]
            channel = spec["channel"]
            family_bins = spec["length_bins"]
            length_hint = spec["length_bin_word_guide"][target_bin]
            system_msg = spec["system_prompt"]
            user_msg = spec["user_template"].format(
                subtype=subtype,
                length_bin=target_bin,
                length_hint=length_hint,
            )

            for attempt in range(1, cfg.max_retries + 1):
                try:
                    text = _api_call(system_msg, user_msg)
                except Exception as exc:
                    _log_line(f"    [API error] {job_id}: {exc}")
                    return

                v = validate(text, channel, family_bins, compute_length_bin)
                if not v["passed"]:
                    _log_line(
                        f"    [QC fail {attempt}/{cfg.max_retries}] {job_id}: {v['issues']}"
                    )
                    continue

                postfix_update = ""
                with state_lock:
                    if is_near_duplicate(text, seen_texts):
                        is_dup = True
                    else:
                        is_dup = False
                        seen_texts.append(text)
                        rec = {
                            "gen_id": str(uuid.uuid4()),
                            "generation_job_id": job_id,
                            "text": text,
                            "label": 1,
                            "label_str": "llm",
                            "fraudness": spec["fraudness"],
                            "channel": channel,
                            "scenario_family": fam,
                            "subtype": subtype,
                            "target_bin": target_bin,
                            "actual_bin": v["actual_bin"],
                            "token_count": v["token_count"],
                            "origin_model": cfg.origin_model,
                            "split": cfg.split,
                            "generator_lane": cfg.lane,
                            "generated_at": datetime.now(timezone.utc).isoformat(),
                            "qc_issues": [],
                        }
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fout.flush()
                        counter["n"] += 1
                        postfix_update = (
                            f"в_файле={counter['n']}/{total_lane_jobs} "
                            f"осталось_всего={max(0, total_lane_jobs - counter['n'])}"
                        )

                if is_dup:
                    _log_line(f"    [near-dup {attempt}/{cfg.max_retries}] {job_id}")
                    continue

                with log_lock:
                    bar.set_postfix_str(postfix_update, refresh=False)
                return

            _log_line(f"    [SKIP] {job_id}: exhausted retries")

        workers = max(1, int(cfg.max_workers))
        if workers <= 1:
            for job in pending:
                process_job(job)
                bar.update(1)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(process_job, job) for job in pending]
                for fut in as_completed(futs):
                    fut.result()
                    bar.update(1)

    print(f"Done. In file: {counter['n']}/{total_lane_jobs} rows. Path: {out_path}")
    return out_path


def find_v2_root(start: Path | None = None) -> Path:
    """Resolve the ``v2/`` directory that contains ``config.py`` and ``src/llm_mass_generation.py``.

    Works when ``cwd`` is:
    - inside ``v2/`` (e.g. ``v2/notebooks/...``) — walks up to ``v2/``;
    - repo root (parent of ``v2/``) — detects ``./v2/config.py``.
    """
    cur = (start or Path.cwd()).resolve()
    for _ in range(24):
        if (cur / "config.py").is_file() and (cur / "src" / "llm_mass_generation.py").is_file():
            return cur
        nested = cur / "v2"
        if (nested / "config.py").is_file() and (nested / "src" / "llm_mass_generation.py").is_file():
            return nested
        if cur.parent == cur:
            break
        cur = cur.parent
    raise RuntimeError(
        "Could not find v2/: need v2/config.py and v2/src/llm_mass_generation.py. "
        "Set Jupyter/Cursor cwd to the repo root, to v2/, or open the folder that contains v2/."
    )


def resolve_v2_base(start: Path | None = None) -> Path:
    """Alias for :func:`find_v2_root` (older name)."""
    return find_v2_root(start)
