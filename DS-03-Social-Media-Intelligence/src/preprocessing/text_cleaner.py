"""Text Cleaner Module: removes noise, normalizes whitespace, preserves semantics."""

import html
import re
import unicodedata
from typing import Dict, List, Tuple

from .emoji_handler import EmojiHandler


class TextCleaner:
    """Comprehensive social media text normalizer."""

    URL_REGEX = re.compile(r"https?://\S+|www\.\S+")
    MENTION_REGEX = re.compile(r"@[\w_]+")
    HASHTAG_REGEX = re.compile(r"#([\w_]+)")
    HTML_TAG_REGEX = re.compile(r"<[^>]+>")
    MULTI_SPACE_REGEX = re.compile(r"\s+")
    PUNCT_REPEAT_REGEX = re.compile(r"([!?.]){2,}")

    @classmethod
    def clean(cls, text: str, preserve_emojis: bool = True) -> str:
        """Clean raw social text."""
        if not text:
            return ""

        # 1. Unescape HTML entities (e.g. &amp; -> &)
        text = html.unescape(text)

        # 2. Normalize Unicode forms (NFKC)
        text = unicodedata.normalize("NFKC", text)

        # 3. Strip HTML tags
        text = cls.HTML_TAG_REGEX.sub(" ", text)

        # 4. Strip URLs
        text = cls.URL_REGEX.sub(" ", text)

        # 5. Handle emojis: either demojize to retain semantic tokens or keep as is
        if preserve_emojis:
            text = EmojiHandler.demojize(text)

        # 6. Normalize punctuation repetition (e.g. "!!!!!" -> "!")
        text = cls.PUNCT_REPEAT_REGEX.sub(r"\1", text)

        # 7. Collapse excessive whitespace
        text = cls.MULTI_SPACE_REGEX.sub(" ", text).strip()

        return text

    @classmethod
    def extract_mentions(cls, text: str) -> List[str]:
        """Extract user handles mentioned in text."""
        if not text:
            return []
        return [m.lstrip("@") for m in cls.MENTION_REGEX.findall(text)]

    @classmethod
    def extract_hashtags(cls, text: str) -> List[str]:
        """Extract hashtags."""
        if not text:
            return []
        return cls.HASHTAG_REGEX.findall(text)
