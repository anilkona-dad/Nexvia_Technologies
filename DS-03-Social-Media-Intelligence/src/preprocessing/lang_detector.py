"""Language Identification Module."""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from langdetect import DetectorFactory, detect
    DetectorFactory.seed = 42
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False


class LanguageDetector:
    """Detects primary language of text with fallback heuristic for multilingual posts."""

    COMMON_WORDS = {
        "en": {"the", "and", "is", "in", "to", "with", "this", "for", "are", "on", "that"},
        "es": {"el", "la", "de", "que", "y", "en", "un", "por", "con", "para", "los"},
        "fr": {"le", "la", "de", "des", "un", "une", "est", "et", "dans", "en", "pour"},
        "de": {"der", "die", "das", "und", "in", "den", "von", "zu", "mit", "sich", "auf"}
    }

    @classmethod
    def detect_language(cls, text: str) -> str:
        """Detect language code, returning standard ISO code (e.g. 'en', 'es', 'fr', 'de')."""
        if not text or len(text.strip()) < 4:
            return "unknown"

        # Clean text for language detection
        clean_for_lang = re.sub(r"https?://\S+|@\w+|#\w+|[:_]", " ", text)
        clean_for_lang = clean_for_lang.strip()

        if HAS_LANGDETECT and len(clean_for_lang) > 10:
            try:
                lang = detect(clean_for_lang)
                return str(lang)
            except Exception:
                pass

        # Lexical frequency fallback
        words = set(re.findall(r"\b[a-zA-Z\u00C0-\u017F]+\b", clean_for_lang.lower()))
        best_lang = "en"
        best_overlap = 0

        for lang_code, vocab in cls.COMMON_WORDS.items():
            overlap = len(words.intersection(vocab))
            if overlap > best_overlap:
                best_overlap = overlap
                best_lang = lang_code

        return best_lang if best_overlap > 0 else "en"
