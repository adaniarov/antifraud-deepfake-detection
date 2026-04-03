"""
SMS Spam Collection
--------------------
Source : https://archive.ics.uci.edu/dataset/228/sms+spam+collection
Ref    : T.A. Almeida, J.M.G. Hidalgo, DocEng'11, 2011.

Tab-separated file: <label>\t<text>
Labels: "spam" / "ham"

Spam  -> data/raw/human_fraud/sms_spam.jsonl   (content_type: smishing)
Ham   -> data/raw/human_legit/sms_ham.jsonl    (content_type: legitimate)
"""

import io
import zipfile
from pathlib import Path

import httpx
from loguru import logger

from .common import make_record, save_jsonl

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def collect(fraud_dir, legit_dir) -> dict:
    fraud_dir, legit_dir = Path(fraud_dir), Path(legit_dir)

    logger.info("  Downloading SMS Spam Collection ...")
    try:
        resp = httpx.get(URL, timeout=60, follow_redirects=True)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"  Failed: {e}")
        return {"error": str(e)}

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        # The archive contains SMSSpamCollection (no extension) and a README
        data_name = next(n for n in zf.namelist() if "SMSSpamCollection" in n)
        lines = zf.read(data_name).decode("utf-8", errors="replace").splitlines()

    spam_records, ham_records = [], []

    for line in lines:
        if "\t" not in line:
            continue
        label_str, text = line.split("\t", 1)
        text = text.strip()
        if not text:
            continue

        if label_str.strip() == "spam":
            spam_records.append(make_record(text, "smishing", "sms_spam_uci"))
        elif label_str.strip() == "ham":
            ham_records.append(make_record(text, "legitimate", "sms_spam_uci"))

    save_jsonl(spam_records, fraud_dir / "sms_spam.jsonl")
    logger.success(f"SMS spam: {len(spam_records)} records saved")

    save_jsonl(ham_records, legit_dir / "sms_ham.jsonl")
    logger.success(f"SMS ham: {len(ham_records)} records saved")

    return {"spam": len(spam_records), "ham": len(ham_records)}


if __name__ == "__main__":
    stats = collect("data/raw/human_fraud", "data/raw/human_legit")
    print(stats)