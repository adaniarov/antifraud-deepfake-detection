"""
Nazario Phishing Corpus
------------------------
Source : http://monkey.org/~jose/phishing/
Ref    : J. Nazario, personal archive, 2005-2007.

Four mbox files: phishing0.mbox ... phishing3.mbox

All -> data/raw/human_fraud/nazario.jsonl  (content_type: phishing)
"""

import mailbox
import tempfile
from pathlib import Path

import httpx
from loguru import logger

from .common import extract_email_body, make_record, save_jsonl

BASE_URL = "http://monkey.org/~jose/phishing"
MBOX_FILES = [
    "phishing0.mbox",
    "phishing1.mbox",
    "phishing2.mbox",
    "phishing3.mbox",
]
def collect(fraud_dir) -> dict:
    fraud_dir = Path(fraud_dir)
    records = []

    for fname in MBOX_FILES:
        url = f"{BASE_URL}/{fname}"
        logger.info(f"  Fetching {fname} ...")
        try:
            resp = httpx.get(url, timeout=60, follow_redirects=True)
            resp.raise_for_status()
        except Exception as e:
            logger.warning(f"  Failed {fname}: {e}")
            continue

        # Write to temp file so mailbox can parse it
        with tempfile.NamedTemporaryFile(suffix=".mbox", delete=False) as tmp:
            tmp.write(resp.content)
            tmp_path = Path(tmp.name)

        try:
            mbox = mailbox.mbox(str(tmp_path))
            for msg in mbox:
                try:
                    raw = msg.as_string()
                except Exception:
                    continue
                text = extract_email_body(raw)
                if text:
                    records.append(make_record(text, "phishing", "nazario"))
        finally:
            tmp_path.unlink(missing_ok=True)

        logger.info(f"    Total so far: {len(records)}")

    save_jsonl(records, fraud_dir / "nazario.jsonl")
    logger.success(f"Nazario: {len(records)} records saved")
    return {"total": len(records)}


if __name__ == "__main__":
    stats = collect("data/raw/human_fraud")
    print(stats)