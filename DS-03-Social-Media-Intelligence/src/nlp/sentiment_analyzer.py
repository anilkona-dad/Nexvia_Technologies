"""Multilingual Transformer-based Sentiment Analysis Module."""

import logging
import math
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MultilingualSentimentAnalyzer:
    """Multilingual Sentiment Analyzer supporting HuggingFace Transformers and heuristic fallback."""

    # Lexicon for fast multilingual fallback across en, es, fr, de
    MULTILINGUAL_VALENCE = {
        # Positive
        "breakthrough": 0.8, "revolutionary": 0.85, "hope": 0.7, "clean": 0.5,
        "investment": 0.4, "game changer": 0.9, "dexterity": 0.5, "surpassing": 0.6,
        "pioneer": 0.6, "success": 0.8, "excellent": 0.85, "good": 0.5, "innovative": 0.75,
        "revolucionario": 0.85, "esperanza": 0.7, "avance": 0.6, "éxito": 0.8,
        "ambitieux": 0.6, "nouveaux": 0.3, "massifs": 0.4, "progrès": 0.7,
        "fortschritt": 0.7, "erfolg": 0.8, "hervorragend": 0.85,
        # Negative
        "tyranny": -0.85, "scam": -0.9, "devastating": -0.85, "vulnerability": -0.6,
        "ransomware": -0.8, "emergency": -0.7, "banned": -0.6, "attack": -0.75,
        "seize": -0.7, "danger": -0.7, "fake": -0.8, "warnung": -0.6, "silenced": -0.6,
        "gefahr": -0.7, "dubiose": -0.6, "attaque": -0.75, "urgence": -0.7,
        "peligro": -0.7, "amenaza": -0.75, "estafa": -0.9
    }

    def __init__(self, model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment", use_gpu: bool = False):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.pipeline = None

        try:
            from transformers import pipeline
            device = 0 if use_gpu else -1
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=device,
                truncation=True,
                max_length=256
            )
            logger.info(f"Loaded Hugging Face sentiment model: {model_name}")
        except Exception as e:
            logger.info(f"Transformer pipeline initialized in fast multilingual fallback mode ({e}).")

    def analyze_text(self, text: str, emoji_hint: float = 0.0) -> Dict[str, Any]:
        """Analyze sentiment of a single piece of text."""
        if not text:
            return {"label": "neutral", "polarity": 0.0, "confidence": 0.5}

        if self.pipeline:
            try:
                res = self.pipeline(text[:512])[0]
                label_str = res["label"].lower()
                score = res["score"]

                # Map labels to polarity [-1.0, 1.0]
                if "pos" in label_str or "4" in label_str or "5" in label_str:
                    label = "positive"
                    polarity = score
                elif "neg" in label_str or "1" in label_str or "2" in label_str:
                    label = "negative"
                    polarity = -score
                else:
                    label = "neutral"
                    polarity = 0.0

                # Blend emoji hint if present
                if abs(emoji_hint) > 0.01:
                    polarity = 0.8 * polarity + 0.2 * emoji_hint

                return {
                    "label": label,
                    "polarity": round(float(polarity), 3),
                    "confidence": round(float(score), 3)
                }
            except Exception as e:
                logger.debug(f"HF pipeline inference error: {e}. Falling back to lexical.")

        return self._lexical_sentiment(text, emoji_hint)

    def _lexical_sentiment(self, text: str, emoji_hint: float = 0.0) -> Dict[str, Any]:
        """Fast multilingual lexical scoring."""
        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return {"label": "neutral", "polarity": 0.0, "confidence": 0.5}

        valence_sum = 0.0
        hit_count = 0

        # Check unigrams and bigrams
        for i in range(len(words)):
            unigram = words[i]
            if unigram in self.MULTILINGUAL_VALENCE:
                valence_sum += self.MULTILINGUAL_VALENCE[unigram]
                hit_count += 1
            if i < len(words) - 1:
                bigram = f"{words[i]} {words[i+1]}"
                if bigram in self.MULTILINGUAL_VALENCE:
                    valence_sum += self.MULTILINGUAL_VALENCE[bigram]
                    hit_count += 1

        if hit_count > 0:
            avg_valence = valence_sum / hit_count
        else:
            avg_valence = 0.0

        # Factor in emoji hint
        combined = 0.7 * avg_valence + 0.3 * emoji_hint if abs(emoji_hint) > 0.01 else avg_valence

        # Clamp between -1.0 and 1.0
        polarity = max(-1.0, min(1.0, combined))

        if polarity > 0.15:
            label = "positive"
            conf = 0.5 + abs(polarity) * 0.45
        elif polarity < -0.15:
            label = "negative"
            conf = 0.5 + abs(polarity) * 0.45
        else:
            label = "neutral"
            conf = 0.65

        return {
            "label": label,
            "polarity": round(float(polarity), 3),
            "confidence": round(float(conf), 3)
        }

    def analyze_batch(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add sentiment scores to a batch of posts."""
        for post in posts:
            text = post.get("cleaned_text") or post.get("content") or post.get("title", "")
            emoji_hint = post.get("emoji_sentiment_hint", 0.0)
            post["sentiment"] = self.analyze_text(text, emoji_hint=emoji_hint)
        return posts
