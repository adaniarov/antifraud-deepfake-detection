"""
Enron-Spam Dataset
-------------------
Source : https://www2.aueb.gr/users/ion/data/enron-spam/
Ref    : V. Metsis, I. Androutsopoulos, G. Paliouras, CEAS 2006.

The AUEB site has the following structure:
  /data/enron-spam/
    Enron1/    (folder, may contain enron1.tar.gz or be downloadable as-is)
    Enron2/ ... Enron6/
    readme.txt

The preprocessed tar.gz files may be accessible as:
  http://www2.aueb.gr/users/ion/data/enron-spam/preprocessed/enron1.tar.gz
  http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron1.tar.gz

Note: https:// may fail due to SSL cert issues on the AUEB server.
      We try http:// first, then multiple URL patterns, then HuggingFace fallback.

Spam  -> data/raw/human_fraud/enron_spam.jsonl   (content_type: spam)
Ham   -> data/raw/human_legit/enron_ham.jsonl    (content_type: legitimate)
"""

import io
import tarfile
from pathlib import Path

import httpx
from loguru import logger

from .common import extract_email_body, make_record, save_jsonl

# Candidate URL patterns to try (in order).
# The AUEB server has SSL issues, so http:// variants are listed first.
URL_PATTERNS = [
    "http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/{name}.tar.gz",
    "http://www2.aueb.gr/users/ion/data/enron-spam/preprocessed/{name}.tar.gz",
    "https://www2.aueb.gr/users/ion/data/enron-spam/preprocessed/{name}.tar.gz",
]

SUBSETS = ["enron1", "enron2", "enron3", "enron4", "enron5", "enron6"]


def _try_download(name: str) -> bytes | None:
    """Try each URL pattern until one works. Returns raw bytes or None."""
    for pattern in URL_PATTERNS:
        url = pattern.format(name=name)
        try:
            resp = httpx.get(url, timeout=120, follow_redirects=True, verify=False)
            if resp.status_code == 200 and len(resp.content) > 1000:
                logger.info(f"    OK: {url}")
                return resp.content
        except Exception as e:
            logger.debug(f"    Failed {url}: {e}")
    return None


def _extract_folder(tar_bytes: bytes, folder: str) -> list:
    """Return list of raw text strings from spam/ or ham/ in the archive."""
    texts = []
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            # Paths look like: enron1/spam/0001.txt or enron1/ham/0001.txt
            parts = Path(member.name).parts
            if len(parts) >= 2 and parts[-2] == folder:
                try:
                    texts.append(
                        tar.extractfile(member).read().decode("utf-8", errors="replace")
                    )
                except Exception:
                    continue
    return texts


def _collect_from_aueb() -> tuple[list, list]:
    """Try to download all 6 subsets from AUEB. Returns (spam_records, ham_records)."""
    spam_records, ham_records = [], []

    for subset in SUBSETS:
        logger.info(f"  Downloading {subset} ...")
        data = _try_download(subset)
        if data is None:
            logger.warning(f"  Could not download {subset} from any URL")
            continue

        for raw in _extract_folder(data, "spam"):
            text = extract_email_body(raw)
            if text:
                spam_records.append(make_record(text, "spam", "enron_spam"))

        for raw in _extract_folder(data, "ham"):
            text = extract_email_body(raw)
            if text:
                ham_records.append(make_record(text, "legitimate", "enron_spam"))

        logger.info(f"    spam: {len(spam_records)} | ham: {len(ham_records)}")

    return spam_records, ham_records


def _collect_from_huggingface() -> tuple[list, list]:
    """
    Fallback: load via HuggingFace datasets (dataset: bvk/ENRON-spam).

    This dataset is a CSV version of the same Metsis et al. corpus.
    Fields: 'Subject', 'Message', 'Spam/Ham'
    """
    logger.info("  Trying HuggingFace fallback: bvk/ENRON-spam ...")
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("bvk/ENRON-spam", split="train", trust_remote_code=True)
    except Exception as e:
        logger.error(f"  HuggingFace fallback failed: {e}")
        return [], []

    spam_records, ham_records = [], []
    for item in ds:
        label = str(item.get("Spam/Ham", "")).strip().lower()
        subject = str(item.get("Subject", "")).strip()
        body = str(item.get("Message", "")).strip()
        # Combine subject + body (same as original dataset convention)
        text = f"{subject}\n{body}".strip() if subject else body
        if not text:
            continue
        if label == "spam":
            spam_records.append(make_record(text, "spam", "enron_spam_hf"))
        elif label == "ham":
            ham_records.append(make_record(text, "legitimate", "enron_spam_hf"))

    logger.info(f"  HuggingFace: spam={len(spam_records)} ham={len(ham_records)}")
    return spam_records, ham_records


def collect(fraud_dir, legit_dir) -> dict:
    fraud_dir, legit_dir = Path(fraud_dir), Path(legit_dir)

    # Strategy 1: AUEB direct download
    spam_records, ham_records = _collect_from_aueb()

    # Strategy 2: HuggingFace fallback (if AUEB failed completely or partially)
    if len(spam_records) == 0 and len(ham_records) == 0:
        logger.warning("  AUEB download failed entirely. Using HuggingFace fallback.")
        spam_records, ham_records = _collect_from_huggingface()

    if not spam_records and not ham_records:
        logger.error(
            "  Enron dataset could not be collected from any source.\n"
            "  Manual option: download enron1-6.tar.gz from:\n"
            "    http://www2.aueb.gr/users/ion/data/enron-spam/\n"
            "  and extract to data/raw/.cache/enron/, then re-run."
        )
        return {"spam": 0, "ham": 0, "status": "failed"}

    save_jsonl(spam_records, fraud_dir / "enron_spam.jsonl")
    logger.success(f"Enron spam: {len(spam_records)} records saved")

    save_jsonl(ham_records, legit_dir / "enron_ham.jsonl")
    logger.success(f"Enron ham: {len(ham_records)} records saved")

    return {"spam": len(spam_records), "ham": len(ham_records)}


if __name__ == "__main__":
    stats = collect("data/raw/human_fraud", "data/raw/human_legit")
    print(stats)