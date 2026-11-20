"""Emoji Handler Module: extraction, demojizing, and sentiment hinting."""

import logging
from typing import List, Tuple

try:
    import emoji
    HAS_EMOJI_LIB = True
except ImportError:
    HAS_EMOJI_LIB = False

logger = logging.getLogger(__name__)


class EmojiHandler:
    """Handles emoji extraction, demojizing, and sentiment hinting."""

    EMOJI_SENTIMENT_MAP = {
        "🔥": 0.8, "🚀": 0.9, "👏": 0.7, "❤️": 0.9, "👍": 0.6,
        "🎉": 0.8, "☀️": 0.5, "🤖": 0.2, "💻": 0.1, "✨": 0.5,
        "💔": -0.8, "🚨": -0.5, "😱": -0.7, "⚠️": -0.6, "💥": -0.4,
        "🛑": -0.6, "☣️": -0.8, "👎": -0.6, "😡": -0.9, "😭": -0.8
    }

    @classmethod
    def extract_emojis(cls, text: str) -> List[str]:
        """Extract all emojis present in the text."""
        if not text:
            return []
        if HAS_EMOJI_LIB:
            return [c["emoji"] for c in emoji.emoji_list(text)]
        # Fallback Unicode character inspection
        return [c for c in text if c in cls.EMOJI_SENTIMENT_MAP]

    @classmethod
    def demojize(cls, text: str) -> str:
        """Convert Unicode emojis into text tags, e.g. 🚀 -> :rocket:"""
        if not text:
            return ""
        if HAS_EMOJI_LIB:
            return emoji.demojize(text, delimiters=(" :", ": "))
        return text

    @classmethod
    def calculate_emoji_polarity(cls, text: str) -> float:
        """Calculate aggregate polarity contribution from emojis."""
        emojis = cls.extract_emojis(text)
        if not emojis:
            return 0.0
        scores = [cls.EMOJI_SENTIMENT_MAP.get(e, 0.0) for e in emojis]
        return sum(scores) / len(scores)
