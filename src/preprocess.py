"""Prétraitement NLP multilingue (normalisation légère, nettoyage)."""

from __future__ import annotations

import re
import unicodedata


# URLs, mentions (style commentaires / réseaux)
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_MULTI_SPACE = re.compile(r"\s+")
# Caractères de contrôle / directionnels RTL inutiles pour le modèle
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def normalize_unicode(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFC", text)


def clean_text(text: str, strip_urls: bool = True, strip_mentions: bool = True) -> str:
    """Nettoyage minimal pour commentaires web (multilingue)."""
    text = normalize_unicode(text)
    text = _CONTROL.sub("", text)
    if strip_urls:
        text = _URL_RE.sub(" ", text)
    if strip_mentions:
        text = _MENTION_RE.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


def is_valid_sample(text: str, min_chars: int = 5, max_chars: int = 5000) -> bool:
    if not text:
        return False
    n = len(text.strip())
    return min_chars <= n <= max_chars
