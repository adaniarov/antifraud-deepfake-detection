"""
v2/config.py — Shared constants for the Core dataset pipeline.

All preparation and assembly notebooks import from here to guarantee
consistent behaviour across sources.

Usage in notebooks
------------------
    import sys
    from pathlib import Path
    BASE = Path('/Users/askar/projects/antifraud-deepfake-detection/v2')
    sys.path.insert(0, str(BASE))
    from config import length_bin

Channel-level length-bin thresholds
------------------------------------
Thresholds are derived from the quantile analysis in:
    v2/notebooks/03_dataset_creation/00_length_bin_analysis.ipynb

Design rationale:
- SMS and email are incomparable in token-length scale; a universal
  threshold would either collapse all SMS into 'short' (email scale)
  or lose resolution for email (SMS scale).
- Channel is already the primary grouping axis in dataset_design_final.md;
  length stratification should follow the same granularity.
- Per-source thresholds are avoided: they break cross-source comparison
  within a channel and were the root cause of the smishtank/mendeley
  length_bin inconsistency found in the April 2026 audit.

Threshold values (token counts, exclusive upper bound for each bin):
  sms   : short  <  20  |  medium  20–59  |  long  ≥  60
  email : short  < 100  |  medium 100–399 |  long  ≥ 400
  qa    : short  <  75  |  medium  75–249 |  long  ≥ 250

Supporting data (percentiles across all channel sources, April 2026):
  SMS fraud (smishtank+mendeley): p25=25 p50=32 p75=45 p90=65 max=197
  SMS ham                       : p25= 9 p50=14 p75=24 p90=37 max=224
  Email (nazario+nigerian+enron+sa): p25=80 p50=185 p75=390 p90=650 max=308k
  QA human (HC3)                : p25=87 p50=156 p75=274 p90=426 max=1966
  QA chatgpt (HC3)              : p25=192 p50=238 p75=284 p90=328 max=629
"""

from __future__ import annotations

LENGTH_BIN_THRESHOLDS: dict[str, dict[str, int]] = {
    "sms":   {"short": 20,  "medium": 60},
    "email": {"short": 100, "medium": 400},
    "qa":    {"short": 75,  "medium": 250},
}

VALID_CHANNELS = frozenset(LENGTH_BIN_THRESHOLDS.keys())


def length_bin(token_count: int, channel: str) -> str:
    """Return 'short', 'medium', or 'long' for *token_count* given *channel*.

    Parameters
    ----------
    token_count : int
        Number of tokens in the text (whitespace/punctuation-split).
    channel : str
        One of 'sms', 'email', 'qa'.

    Raises
    ------
    KeyError
        If *channel* is not in LENGTH_BIN_THRESHOLDS.
    """
    t = LENGTH_BIN_THRESHOLDS[channel]
    if token_count < t["short"]:
        return "short"
    if token_count < t["medium"]:
        return "medium"
    return "long"
