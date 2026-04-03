"""
SpamAssassin Public Mail Corpus
--------------------------------
Source : https://spamassassin.apache.org/old/publiccorpus/
Ref    : Apache Software Foundation, 2003-2005.

Spam  -> data/raw/human_fraud/spamassassin_spam.jsonl  (content_type: spam)
Ham   -> data/raw/human_legit/spamassassin_ham.jsonl   (content_type: legitimate)

Archives are downloaded on demand (~2-5 MB each, no disk cache needed).
"""

import io
import tarfile
from pathlib import Path

import httpx
from loguru import logger

from .common import extract_email_body, make_record, save_jsonl

SPAM_ARCHIVES = [
    "https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2",
    "https://spamassassin.apache.org/old/publiccorpus/20030228_spam_2.tar.bz2",
    "https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2",
]

HAM_ARCHIVES = [
    "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
    "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2",
]

def _collect_archives(urls: list, content_type: str, source_id: str) -> list:
    records = []
    for url in urls:
        logger.info(f"  Downloading {url.split('/')[-1]} ...")
        try:
            resp = httpx.get(url, timeout=60, follow_redirects=True)
            resp.raise_for_status()
        except Exception as e:
            logger.warning(f"  Failed: {e}")
            continue

        with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:bz2") as tar:
            for member in tar.getmembers():
                if not member.isfile() or member.name.endswith("cmds"):
                    continue
                try:
                    raw = tar.extractfile(member).read().decode("utf-8", errors="replace")
                except Exception:
                    continue
                text = extract_email_body(raw)
                if not text:
                    continue
                records.append(make_record(text, content_type, source_id))

    return records


def collect(fraud_dir, legit_dir) -> dict:
    fraud_dir, legit_dir = Path(fraud_dir), Path(legit_dir)

    logger.info("SpamAssassin: spam -> Group A")
    spam = _collect_archives(SPAM_ARCHIVES, "spam", "spamassassin")
    save_jsonl(spam, fraud_dir / "spamassassin_spam.jsonl")
    logger.success(f"  {len(spam)} records saved")

    logger.info("SpamAssassin: ham -> Group B")
    ham = _collect_archives(HAM_ARCHIVES, "legitimate", "spamassassin")
    save_jsonl(ham, legit_dir / "spamassassin_ham.jsonl")
    logger.success(f"  {len(ham)} records saved")

    return {"spam": len(spam), "ham": len(ham)}


if __name__ == "__main__":
    stats = collect("data/raw/human_fraud", "data/raw/human_legit")
    print(stats)