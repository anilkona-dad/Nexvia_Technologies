"""Misinformation and Fake News Classifier trained on LIAR benchmark."""

import logging
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from .liar_loader import LIARDatasetLoader

logger = logging.getLogger(__name__)


class MisinformationClassifier:
    """Classifies social media posts for misinformation risk."""

    ALARMIST_KEYWORDS = [
        "secret", "banned", "tyranny", "miracle", "cures", "silenced",
        "conspiracy", "microchips", "seize", "emergency", "warning",
        "urgent", "shocking", "hiding", "leaked", "confidential"
    ]

    def __init__(self, model_dir: str = "data/models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_file = self.model_dir / "misinfo_classifier.pkl"
        self.vectorizer_file = self.model_dir / "misinfo_vectorizer.pkl"

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.classifier: Optional[LogisticRegression] = None
        self.is_trained = False

        self._load_if_exists()

    def _load_if_exists(self):
        """Load serialized model weights if previously trained."""
        if self.model_file.exists() and self.vectorizer_file.exists():
            try:
                with open(self.model_file, "rb") as f:
                    self.classifier = pickle.load(f)
                with open(self.vectorizer_file, "rb") as f:
                    self.vectorizer = pickle.load(f)
                self.is_trained = True
                logger.info("Loaded pre-trained Misinformation Classifier from disk.")
            except Exception as e:
                logger.warning(f"Could not load cached model: {e}")

    def train_and_evaluate(self, test_size: float = 0.25, random_state: int = 42) -> Dict[str, Any]:
        """Train classifier on LIAR benchmark dataset and return comprehensive performance metrics."""
        texts, labels_binary, _ = LIARDatasetLoader.load_data()

        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels_binary, test_size=test_size, random_state=random_state, stratify=labels_binary
        )

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 1),
            stop_words=None,
            max_features=3000,
            sublinear_tf=True
        )

        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        self.classifier = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            random_state=random_state,
            max_iter=500
        )
        self.classifier.fit(X_train_vec, y_train)
        self.is_trained = True

        # Save model
        with open(self.model_file, "wb") as f:
            pickle.dump(self.classifier, f)
        with open(self.vectorizer_file, "wb") as f:
            pickle.dump(self.vectorizer, f)

        # Evaluation
        y_pred = self.classifier.predict(X_test_vec)
        y_prob = self.classifier.predict_proba(X_test_vec)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()

        metrics = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": cm,
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }
        logger.info(f"Model Training Complete. Metrics: {metrics}")
        return metrics

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict misinformation probability and risk category for text."""
        if not self.is_trained:
            self.train_and_evaluate()

        # Linguistic features
        clean = text.lower()
        exclamation_count = text.count("!")
        caps_words = len([w for w in text.split() if w.isupper() and len(w) > 1])
        alarmist_hits = sum(1 for kw in self.ALARMIST_KEYWORDS if kw in clean)

        # Heuristic boost for sensationalist phrasing
        linguistic_boost = min(0.35, (exclamation_count * 0.05) + (caps_words * 0.04) + (alarmist_hits * 0.08))

        # Model probability: class 0 = Fake, class 1 = Credible
        vec = self.vectorizer.transform([text])
        prob_credible = float(self.classifier.predict_proba(vec)[0][1])
        prob_fake = 1.0 - prob_credible

        # Combine model probability with additive linguistic cues
        combined_fake_prob = min(0.99, max(0.01, prob_fake + linguistic_boost))

        if combined_fake_prob >= 0.70:
            risk = "CRITICAL"
            label = "Misinformation / Fake"
        elif combined_fake_prob >= 0.50:
            risk = "HIGH"
            label = "Questionable / Unverified"
        elif combined_fake_prob >= 0.30:
            risk = "MEDIUM"
            label = "Partially Reliable"
        else:
            risk = "LOW"
            label = "Credible / Fact-Consistent"

        return {
            "label": label,
            "fake_probability": round(float(combined_fake_prob), 3),
            "risk_level": risk,
            "linguistic_cues": {
                "exclamation_count": exclamation_count,
                "all_caps_words": caps_words,
                "alarmist_matches": alarmist_hits
            }
        }

    def enrich_posts(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Append misinformation predictions to each post."""
        for p in posts:
            text = f"{p.get('title', '')} {p.get('content', '')}".strip()
            p["misinformation"] = self.predict(text)
        return posts
