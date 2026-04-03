# """
# Phishing Corpora Collector
# ---------------------------
# Источники:
#   1. Nazario Phishing Corpus  — mbox файлы с monkey.org
#   2. Nigerian Fraud / 419 Scam — CSV с Kaggle/GitHub
#   3. IWSPA-AP 2018             — из HuggingFace datasets (зеркало)

# Все → data/raw/human_fraud/ (label=0, content_type=phishing)

# Примечание по Nazario:
#   Оригинальный сайт monkey.org периодически недоступен.
#   Скрипт пробует несколько зеркал в порядке приоритета.
# """

# import email
# import hashlib
# import json
# import mailbox
# import re
# import tempfile
# from pathlib import Path

# import httpx
# from loguru import logger
# from tenacity import retry, stop_after_attempt, wait_exponential
# from tqdm import tqdm

# # ---------------------------------------------------------------------------
# # Config
# # ---------------------------------------------------------------------------

# # Nazario — прямые ссылки на mbox-файлы (несколько зеркал)
# NAZARIO_SOURCES = [
#     # Первичный
#     "http://monkey.org/~jose/phishing/phishing0.mbox",
#     "http://monkey.org/~jose/phishing/phishing1.mbox",
#     "http://monkey.org/~jose/phishing/phishing2.mbox",
#     "http://monkey.org/~jose/phishing/phishing3.mbox",
#     # GitHub зеркало (агрегация)
#     "https://raw.githubusercontent.com/rf-peixoto/phishing_pot/main/email/emails.json",
# ]

# # HuggingFace phishing email dataset (replaces broken Nigerian CSV)
# HF_PHISHING_DATASET = "doantumy/email-phishing"  # fields: text, label (1=phishing)

# MIN_CHARS = 30
# MAX_CHARS = 4000


# # ---------------------------------------------------------------------------
# # Helpers
# # ---------------------------------------------------------------------------


# @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
# def download(url: str, timeout: int = 60) -> bytes:
#     logger.info(f"Fetching: {url}")
#     resp = httpx.get(url, follow_redirects=True, timeout=timeout)
#     resp.raise_for_status()
#     return resp.content


# def sha256(text: str) -> str:
#     return hashlib.sha256(text.encode()).hexdigest()


# def clean_email_text(text: str) -> str:
#     """Universal email text cleaner."""
#     # Remove HTML tags
#     text = re.sub(r"<[^>]+>", " ", text)
#     # Remove email headers artifacts
#     text = re.sub(
#         r"^(From|To|CC|Subject|Date|Received|Return-Path|X-[^:]+|MIME-Version|Content-Type):.*$",
#         "",
#         text,
#         flags=re.MULTILINE | re.IGNORECASE,
#     )
#     # Remove quoted replies
#     text = re.sub(r"^>.*$", "", text, flags=re.MULTILINE)
#     # Remove URLs (keep text around them)
#     text = re.sub(r"https?://\S+", "[URL]", text)
#     # Normalize whitespace
#     text = re.sub(r"\s+", " ", text).strip()
#     return text[:MAX_CHARS]


# def extract_mbox_body(msg) -> str | None:
#     """Extract plain text body from mailbox.Message."""
#     body_parts = []
#     if msg.is_multipart():
#         for part in msg.walk():
#             if part.get_content_type() == "text/plain":
#                 try:
#                     payload = part.get_payload(decode=True)
#                     charset = part.get_content_charset() or "utf-8"
#                     body_parts.append(payload.decode(charset, errors="replace"))
#                 except Exception:
#                     pass
#     else:
#         try:
#             payload = msg.get_payload(decode=True)
#             if payload:
#                 charset = msg.get_content_charset() or "utf-8"
#                 body_parts.append(payload.decode(charset, errors="replace"))
#             else:
#                 raw = msg.get_payload()
#                 if isinstance(raw, str):
#                     body_parts.append(raw)
#         except Exception:
#             pass

#     return "\n".join(body_parts).strip() or None


# # ---------------------------------------------------------------------------
# # Source-specific parsers
# # ---------------------------------------------------------------------------


# def collect_nazario_mbox(url: str, seen: set) -> list[dict]:
#     """Parse a single Nazario .mbox file."""
#     records = []
#     try:
#         data = download(url)
#     except Exception as e:
#         logger.warning(f"Nazario mbox unavailable: {url} — {e}")
#         return records

#     # Write to temp file because mailbox needs file path
#     with tempfile.NamedTemporaryFile(suffix=".mbox", delete=False) as tmp:
#         tmp.write(data)
#         tmp_path = tmp.name

#     try:
#         mbox = mailbox.mbox(tmp_path)
#         for msg in tqdm(mbox, desc=f"Nazario: {url.split('/')[-1]}", leave=False):

#             body = extract_mbox_body(msg)
#             if not body:
#                 continue

#             text = clean_email_text(body)
#             if len(text) < MIN_CHARS:
#                 continue

#             h = sha256(text)
#             if h in seen:
#                 continue
#             seen.add(h)

#             records.append({
#                 "text": text,
#                 "label": 0,
#                 "label_str": "human",
#                 "origin_model": None,
#                 "content_type": "phishing",
#                 "dataset_source": "nazario",
#                 "prompt_type": None,
#                 "prompt_id": None,
#                 "char_length": len(text),
#                 "split": None,
#             })
#     except Exception as e:
#         logger.error(f"Error parsing mbox {url}: {e}")
#     finally:
#         Path(tmp_path).unlink(missing_ok=True)

#     return records


# def collect_huggingface_phishing(seen: set) -> list[dict]:
#     """
#     Load phishing emails from doantumy/email-phishing on HuggingFace.
#     Fields: 'text' (email body), 'label' (1 = phishing, take only those).
#     """
#     records = []
#     try:
#         from datasets import load_dataset  # type: ignore

#         logger.info(f"Loading {HF_PHISHING_DATASET} from HuggingFace...")
#         ds = load_dataset(HF_PHISHING_DATASET, split="train")

#         for item in tqdm(ds, desc="HF phishing (doantumy)", leave=False):
#             # Keep only phishing rows (label == 1)
#             if item.get("label") != 1:
#                 continue

#             text = clean_email_text(str(item.get("text", "")))
#             if len(text) < MIN_CHARS:
#                 continue

#             h = sha256(text)
#             if h in seen:
#                 continue
#             seen.add(h)

#             records.append({
#                 "text": text,
#                 "label": 0,
#                 "label_str": "human",
#                 "origin_model": None,
#                 "content_type": "scam",
#                 "dataset_source": "hf_phishing",
#                 "prompt_type": None,
#                 "prompt_id": None,
#                 "char_length": len(text),
#                 "split": None,
#             })

#     except Exception as e:
#         logger.warning(f"HuggingFace phishing dataset failed: {e}")

#     return records


# # ---------------------------------------------------------------------------
# # Main collector
# # ---------------------------------------------------------------------------


# def collect(output_dir_fraud: Path) -> dict:
#     """
#     Collect all phishing/fraud human texts.

#     Sources:
#     1. Nazario mbox files
#     2. HuggingFace phishing dataset (complement)
#     """
#     output_dir_fraud.mkdir(parents=True, exist_ok=True)

#     seen_hashes: set[str] = set()
#     all_records: list[dict] = []
#     stats = {"nazario": 0, "hf_phishing": 0}

#     # 1. Nazario
#     logger.info("=== Collecting Nazario Phishing Corpus ===")
#     for url in NAZARIO_SOURCES[:4]:  # .mbox files only
#         recs = collect_nazario_mbox(url, seen_hashes)
#         all_records.extend(recs)
#         stats["nazario"] += len(recs)
#         logger.info(f"Nazario so far: {stats['nazario']}")

#     # 2. HuggingFace phishing dataset
#     logger.info("=== Collecting HF phishing dataset ===")
#     recs = collect_huggingface_phishing(seen_hashes)
#     all_records.extend(recs)
#     stats["hf_phishing"] += len(recs)

#     # Save all
#     out_file = output_dir_fraud / "phishing_corpora.jsonl"
#     with open(out_file, "w", encoding="utf-8") as fp:
#         for r in all_records:
#             fp.write(json.dumps(r, ensure_ascii=False) + "\n")

#     total = len(all_records)
#     logger.success(
#         f"Phishing corpora: saved {total} records → {out_file}\n"
#         f"  Nazario: {stats['nazario']}, HF phishing: {stats['hf_phishing']}"
#     )
#     return {**stats, "total": total}


# # ---------------------------------------------------------------------------
# # CLI
# # ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     import typer

#     def main(fraud_dir: Path = Path("data/raw/human_fraud")):
#         stats = collect(fraud_dir)
#         logger.info(f"Final stats: {stats}")

#     typer.run(main)