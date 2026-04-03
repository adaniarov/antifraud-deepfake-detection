"""
common.py — utilities shared by all collectors.

Responsibility: format parsing only.
  - Extract plain text from email/mbox/zip/JSONL
  - Build raw records
  - Save JSONL

No research decisions here:
  - No length filtering
  - No URL masking
  - No deduplication
  - No language detection
"""

import hashlib
import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Email body extraction
# ---------------------------------------------------------------------------

def extract_email_body(raw: str) -> str:
    """
    Parse a raw RFC 2822 email string and return plain-text body.

    Steps:
      1. Parse MIME structure with stdlib email
      2. Prefer text/plain parts; fall back to text/html with tags stripped
      3. Decode Content-Transfer-Encoding (base64, quoted-printable)
      4. Strip residual HTML tags from any HTML part
    """
    from email import message_from_string
    from email.errors import MessageError

    try:
        msg = message_from_string(raw)
    except (MessageError, Exception):
        # Fallback: treat entire string as body
        return _strip_html(_basic_header_strip(raw))

    parts = []

    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct not in ("text/plain", "text/html"):
                continue
            try:
                payload = part.get_payload(decode=True) or b""
                charset = part.get_content_charset() or "utf-8"
                text = payload.decode(charset, errors="replace")
            except Exception:
                text = str(part.get_payload() or "")
            if ct == "text/html":
                text = _strip_html(text)
            parts.append(text)
    else:
        try:
            payload = msg.get_payload(decode=True) or b""
            charset = msg.get_content_charset() or "utf-8"
            body = payload.decode(charset, errors="replace")
        except Exception:
            body = str(msg.get_payload() or "")
        if msg.get_content_type() == "text/html":
            body = _strip_html(body)
        parts.append(body)

    return " ".join(parts).strip()


def _strip_html(text: str) -> str:
    """Remove HTML tags, keeping text nodes."""
    return re.sub(r"<[^>]+>", " ", text)


def _basic_header_strip(text: str) -> str:
    """Fallback: remove lines that look like email headers."""
    lines = text.splitlines()
    body_lines = []
    in_body = False
    for line in lines:
        if not in_body and re.match(r"^[A-Za-z\-]+:\s", line):
            continue
        if not in_body and line.strip() == "":
            in_body = True
            continue
        body_lines.append(line)
    return "\n".join(body_lines)


# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------

def make_record(text: str, content_type: str, dataset_source: str) -> dict:
    """
    Build a raw human-written record.

    'text' is the extracted body — not further cleaned.
    All cleaning decisions happen in preprocessing/.
    """
    return {
        "text": text,
        "label": 0,
        "label_str": "human",
        "origin_model": None,
        "content_type": content_type,
        "dataset_source": dataset_source,
        "char_length": len(text),
        "split": None,
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_jsonl(records: list, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            line = json.dumps(r, ensure_ascii=True) + "\n"
            f.write(line)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()