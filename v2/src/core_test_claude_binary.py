"""
Build derived evaluation slice `test_claude_binary`: Claude holdout (label=1) +
matched human companions from human reserve (not in train/val/test_seen).

Reproducible: fixed RNG seed (default 42). Does not modify core_train/val/test_non_claude.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _resolve_v2_base() -> Path:
    cur = Path(__file__).resolve().parent.parent
    if (cur / "pyproject.toml").is_file():
        return cur
    raise FileNotFoundError("Run from v2 layout (pyproject.toml next to src/).")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _sanitize(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: _sanitize(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_sanitize(v) for v in x]
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return x

    with path.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(_sanitize(rec), ensure_ascii=False) + "\n")


def human_fingerprint(r: dict[str, Any]) -> tuple[Any, ...]:
    """Stable id for a human row (dataset1 or core human)."""
    sf = r.get("scenario_family")
    if sf == "financial_qa":
        qid = r.get("question_id")
        if qid is None:
            return ("fq", None)
        if isinstance(qid, float) and math.isnan(qid):
            return ("fq", None)
        return ("fq", int(qid) if float(qid) == int(qid) else float(qid))
    pf = r.get("provenance_source_file")
    pl = r.get("provenance_line_no")
    if pl is None or (isinstance(pl, float) and math.isnan(pl)):
        pln = -1
    else:
        pln = int(pl)
    return ("std", str(pf), pln)


def collect_main_pool_human_keys(paths: list[Path]) -> set[tuple[Any, ...]]:
    keys: set[tuple[Any, ...]] = set()
    for p in paths:
        for r in load_jsonl(p):
            if int(r.get("label", -1)) != 0:
                continue
            keys.add(human_fingerprint(r))
    return keys


def _align_key_for_match(r: dict[str, Any], *, with_time_band: bool) -> tuple[Any, ...]:
    """
    Keys for Claude↔human alignment. Do **not** compare LLM `source_family` to human
    `source_family` (Claude rows use placeholders like `llm_holdout_claude`).
    """
    t: tuple[Any, ...] = (r.get("channel"), r.get("fraudness"), r.get("length_bin"))
    if with_time_band:
        t = t + (r.get("time_band"),)
    return t


@dataclass
class MatchAttempt:
    claude_gen_id: str | None
    human_fingerprint: tuple[Any, ...]
    scenario_family: str
    tier: int
    stratum_claude: tuple[Any, ...]
    stratum_human: tuple[Any, ...]


@dataclass
class BuildReport:
    per_family: dict[str, dict[str, Any]] = field(default_factory=dict)
    match_attempts: list[MatchAttempt] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)


def _family_reserve_humans(
    dataset1_humans: list[dict[str, Any]],
    main_keys: set[tuple[Any, ...]],
    scenario_family: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in dataset1_humans:
        if r.get("scenario_family") != scenario_family:
            continue
        if int(r.get("label", 0)) != 0:
            continue
        fp = human_fingerprint(r)
        if fp in main_keys:
            continue
        out.append(r)
    return out


def _try_match_human(
    c: dict[str, Any],
    pool: list[dict[str, Any]],
    used_fp: set[tuple[Any, ...]],
    rng,
) -> tuple[dict[str, Any] | None, int, tuple[Any, ...], tuple[Any, ...]]:
    """
    Tiers (strict → loose), same axes for Claude and human:
      0 — channel, fraudness, length_bin, time_band
      1 — channel, fraudness, length_bin
      2 — channel, fraudness (last resort within scenario_family)
    """
    for tier_idx, wtb in enumerate((True, False)):
        c_key = _align_key_for_match(c, with_time_band=wtb)
        cand: list[dict[str, Any]] = []
        for h in pool:
            fp = human_fingerprint(h)
            if fp in used_fp:
                continue
            h_key = _align_key_for_match(h, with_time_band=wtb)
            if h_key == c_key:
                cand.append(h)
        if cand:
            pick = cand[int(rng.integers(0, len(cand)))]
            hk = _align_key_for_match(pick, with_time_band=wtb)
            return pick, tier_idx, c_key, hk
    # tier 2
    c_key = (c.get("channel"), c.get("fraudness"))
    cand = []
    for h in pool:
        fp = human_fingerprint(h)
        if fp in used_fp:
            continue
        if (h.get("channel"), h.get("fraudness")) == c_key:
            cand.append(h)
    if cand:
        pick = cand[int(rng.integers(0, len(cand)))]
        hk = (pick.get("channel"), pick.get("fraudness"))
        return pick, 2, c_key, hk
    return None, -1, (), ()


def build_test_claude_binary(
    *,
    base: Path | None = None,
    random_seed: int = 42,
) -> tuple[list[dict[str, Any]], BuildReport]:
    base = base or _resolve_v2_base()
    assembled = base / "data" / "interim" / "assembled"
    ds1_path = assembled / "dataset1_human.jsonl"
    claude_path = assembled / "core_test_claude_only.jsonl"
    main_paths = [
        assembled / "core_train.jsonl",
        assembled / "core_val.jsonl",
        assembled / "core_test_non_claude.jsonl",
    ]
    for p in main_paths + [ds1_path, claude_path]:
        if not p.is_file():
            raise FileNotFoundError(p)

    rng = __import__("numpy").random.default_rng(
        __import__("numpy").random.PCG64(random_seed)
    )

    main_keys = collect_main_pool_human_keys(main_paths)
    dataset1 = load_jsonl(ds1_path)
    claude_rows = load_jsonl(claude_path)

    by_sf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in claude_rows:
        by_sf[str(r["scenario_family"])].append(r)

    report = BuildReport()
    out_rows: list[dict[str, Any]] = []
    used_human_fp: set[tuple[Any, ...]] = set()

    for sf, clist in sorted(by_sf.items(), key=lambda x: x[0]):
        pool = _family_reserve_humans(dataset1, main_keys, sf)
        n_need = len(clist)
        n_res = len(pool)
        order = rng.permutation(len(clist))
        matched = 0
        tier_counts = Counter()
        shortfall_reason: str | None = None

        if n_res == 0:
            shortfall_reason = "no_human_rows_in_reserve_all_main_pool_consumed"
            report.per_family[sf] = {
                "n_claude_holdout": n_need,
                "n_human_reserve_available": 0,
                "n_pairs_written": 0,
                "shortfall_reason": shortfall_reason,
            }
            continue

        for idx in order:
            c = clist[int(idx)]
            h, tier, ck, hk = _try_match_human(c, pool, used_human_fp, rng)
            gid = c.get("gen_id")
            if h is None:
                report.match_attempts.append(
                    MatchAttempt(
                        claude_gen_id=str(gid) if gid is not None else None,
                        human_fingerprint=(),
                        scenario_family=sf,
                        tier=-1,
                        stratum_claude=ck,
                        stratum_human=hk,
                    )
                )
                continue
            hf = human_fingerprint(h)
            used_human_fp.add(hf)
            matched += 1
            tier_counts[tier] += 1
            report.match_attempts.append(
                MatchAttempt(
                    claude_gen_id=str(gid) if gid is not None else None,
                    human_fingerprint=hf,
                    scenario_family=sf,
                    tier=tier,
                    stratum_claude=ck,
                    stratum_human=hk,
                )
            )

            h_out = dict(h)
            h_out["split"] = "test"
            h_out["core_eval_slice"] = "test_claude_binary"
            h_out["claude_binary_role"] = "human_companion"
            h_out["claude_binary_match_tier"] = tier
            h_out["claude_binary_matched_gen_id"] = gid
            out_rows.append(h_out)

            c_out = dict(c)
            c_out["core_eval_slice"] = "test_claude_binary"
            c_out["claude_binary_role"] = "claude_holdout"
            c_out["claude_binary_match_tier"] = tier
            c_out["claude_binary_matched_human_fingerprint"] = list(hf)
            out_rows.append(c_out)

        if matched < n_need:
            shortfall_reason = (
                f"matched_only_{matched}_of_{n_need}_insufficient_stratum_overlap_or_reserve"
            )
        report.per_family[sf] = {
            "n_claude_holdout": n_need,
            "n_human_reserve_available": n_res,
            "n_pairs_written": matched,
            "tier_counts": dict(tier_counts),
            "shortfall_reason": shortfall_reason,
        }

    # --- checks ---
    labels = [int(r["label"]) for r in out_rows]
    report.checks.append(
        {
            "name": "claude_binary_has_both_labels",
            "ok": 0 in labels and 1 in labels,
            "n_label_0": sum(1 for x in labels if x == 0),
            "n_label_1": sum(1 for x in labels if x == 1),
        }
    )

    out_fps = {human_fingerprint(r) for r in out_rows if int(r.get("label", -1)) == 0}
    overlap = out_fps & main_keys
    report.checks.append(
        {
            "name": "claude_binary_no_overlap_with_train_val_testseen",
            "ok": len(overlap) == 0,
            "n_overlap_keys": len(overlap),
        }
    )

    fam_bal = []
    ok_bal = True
    for sf, pr in report.per_family.items():
        nc = pr["n_claude_holdout"]
        nw = pr["n_pairs_written"]
        fam_bal.append(
            {
                "scenario_family": sf,
                "n_claude": nc,
                "n_human_matched": nw,
                "balanced": nw == nc,
            }
        )
        if nw != nc:
            ok_bal = False
    report.checks.append(
        {
            "name": "claude_binary_family_balance",
            "ok": ok_bal,
            "per_family": fam_bal,
            "note": "ok=false means some families could not be fully matched (see shortfall_reason)",
        }
    )

    tier_matched = [m for m in report.match_attempts if m.tier >= 0]
    tier0_share = (
        sum(1 for m in tier_matched if m.tier == 0) / len(tier_matched)
        if tier_matched
        else 0.0
    )
    tier01_share = (
        sum(1 for m in tier_matched if m.tier in (0, 1)) / len(tier_matched)
        if tier_matched
        else 0.0
    )
    report.checks.append(
        {
            "name": "claude_binary_match_quality_check",
            "ok": tier01_share >= 0.5 or len(tier_matched) == 0,
            "tier0_share_among_matched": tier0_share,
            "tier01_share_among_matched": tier01_share,
            "n_matched": len(tier_matched),
            "note": "tier 0 = channel+fraudness+length_bin+time_band; tier 1 drops time_band; ok if ≥50% tier0∪1",
        }
    )

    return out_rows, report


def merge_manifest(
    base: Path,
    out_path: Path,
    binary_path: Path,
    n_binary: int,
    report: BuildReport,
) -> None:
    man_path = base / "data" / "interim" / "assembled" / "core_manifest.json"
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    outs = manifest.setdefault("outputs", {})
    outs["core_test_claude_binary"] = str(binary_path.resolve())
    rc = manifest.setdefault("row_counts", {})
    rc["test_claude_binary"] = n_binary
    manifest["test_claude_binary"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": 42,
        "row_count": n_binary,
        "per_family": report.per_family,
        "checks": report.checks,
    }
    val = manifest.setdefault("validation", {})
    chks = val.setdefault("checks", [])
    # drop previous claude_binary checks if re-run
    chks[:] = [c for c in chks if not str(c.get("name", "")).startswith("claude_binary_")]
    chks.extend(report.checks)
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def write_crosstabs(base: Path, rows: list[dict[str, Any]]) -> None:
    import pandas as pd

    tables = base / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    prefix = "core_crosstab_test_claude_binary"

    def save(name: str, obj: pd.DataFrame | pd.Series) -> None:
        p = tables / f"{prefix}__{name}.csv"
        obj.to_csv(p)

    save("scenario_family__label", pd.crosstab(df["scenario_family"], df["label"]))
    save("channel__label", pd.crosstab(df["channel"], df["label"]))
    save("fraudness__label", pd.crosstab(df["fraudness"], df["label"]))
    save("length_bin__label", pd.crosstab(df["length_bin"], df["label"]))
    save(
        "scenario_family__channel__label",
        pd.crosstab([df["scenario_family"], df["channel"]], df["label"]),
    )
    if "time_band" in df.columns:
        save(
            "time_band__scenario_family__label",
            pd.crosstab([df["time_band"], df["scenario_family"]], df["label"]),
        )
    if "source_family" in df.columns:
        save(
            "source_family__scenario_family__label",
            pd.crosstab([df["source_family"], df["scenario_family"]], df["label"]),
        )
    save("match_tier__label", pd.crosstab(df["claude_binary_match_tier"], df["label"]))


def append_split_diagnostics_md(base: Path, report: BuildReport, n_rows: int) -> None:
    path = base / "docs" / "core_split_diagnostics.md"
    marker_start = "<!-- AUTO:test_claude_binary:start -->\n"
    marker_end = "<!-- AUTO:test_claude_binary:end -->\n"
    text = path.read_text(encoding="utf-8")
    if marker_start in text and marker_end in text:
        pre, _, rest = text.partition(marker_start)
        _, _, post = rest.partition(marker_end)
        text = pre + post
    elif "## test_claude_binary (derived slice" in text:
        idx = text.index("## test_claude_binary (derived slice")
        text = text[:idx].rstrip() + "\n"

    block = []

    def _utc() -> str:
        return datetime.now(timezone.utc).isoformat()

    block.append("\n\n" + marker_start)
    block.append(f"## test_claude_binary (derived slice, UTC: {_utc()})\n\n")
    block.append(
        "Файл: `v2/data/interim/assembled/core_test_claude_binary.jsonl`. "
        "Human companions набраны из **резерва** `dataset1_human.jsonl` "
        "(строки **не** вошедшие в train/val/test_seen), с тирами согласования страт.\n\n"
    )
    block.append(f"- Строк в срезе: **{n_rows}**\n")
    block.append("- Проверки (см. также `core_manifest.json` → `test_claude_binary.checks`):\n")
    for c in report.checks:
        block.append(f"  - **{c['name']}**: ok=`{c.get('ok')}`\n")
    block.append("\n### Доступность vs отобранные пары по `scenario_family`\n\n")
    block.append("| scenario_family | n_claude | human_reserve | pairs_written | причина shortfall |\n")
    block.append("|---|---:|---:|---:|---|\n")
    for sf, pr in sorted(report.per_family.items()):
        reason = pr.get("shortfall_reason") or "—"
        block.append(
            f"| `{sf}` | {pr['n_claude_holdout']} | {pr['n_human_reserve_available']} | "
            f"{pr['n_pairs_written']} | {reason} |\n"
        )
    block.append("\n" + marker_end)
    path.write_text(text + "".join(block), encoding="utf-8")


def main() -> None:
    base = _resolve_v2_base()
    rows, report = build_test_claude_binary(base=base, random_seed=42)
    out_jsonl = base / "data" / "interim" / "assembled" / "core_test_claude_binary.jsonl"
    write_jsonl(out_jsonl, rows)
    print("Wrote", out_jsonl, "n=", len(rows))
    merge_manifest(base, out_jsonl, out_jsonl, len(rows), report)
    write_crosstabs(base, rows)
    append_split_diagnostics_md(base, report, len(rows))
    for c in report.checks:
        print(c)


if __name__ == "__main__":
    main()
