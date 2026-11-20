"""Data Preprocessing Pipeline orchestrator."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .emoji_handler import EmojiHandler
from .lang_detector import LanguageDetector
from .text_cleaner import TextCleaner

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """End-to-end preprocessing pipeline for social media posts."""

    def __init__(self, preserve_emojis: bool = True):
        self.cleaner = TextCleaner()
        self.emoji_handler = EmojiHandler()
        self.lang_detector = LanguageDetector()
        self.preserve_emojis = preserve_emojis

    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Process an individual post record."""
        raw_title = record.get("title", "")
        raw_content = record.get("content", "")
        combined_raw = f"{raw_title} {raw_content}".strip()

        # Extract emojis and mentions before stripping
        emojis = self.emoji_handler.extract_emojis(combined_raw)
        mentions = self.cleaner.extract_mentions(combined_raw)
        hashtags = self.cleaner.extract_hashtags(combined_raw)

        # Clean text
        cleaned_title = self.cleaner.clean(raw_title, preserve_emojis=self.preserve_emojis)
        cleaned_content = self.cleaner.clean(raw_content, preserve_emojis=self.preserve_emojis)
        cleaned_text = f"{cleaned_title} {cleaned_content}".strip()

        # Language Detection
        detected_lang = self.lang_detector.detect_language(cleaned_text)

        # Assemble enriched record
        enriched = dict(record)
        enriched["cleaned_text"] = cleaned_text
        enriched["cleaned_title"] = cleaned_title
        enriched["cleaned_content"] = cleaned_content
        enriched["emojis"] = emojis
        enriched["mentions"] = mentions
        enriched["hashtags"] = hashtags
        enriched["language"] = detected_lang
        enriched["emoji_sentiment_hint"] = self.emoji_handler.calculate_emoji_polarity(combined_raw)

        return enriched

    def process_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of in-memory records."""
        return [self.process_record(r) for r in records]

    def process_directory(self, raw_dir: str, output_path: str) -> List[Dict[str, Any]]:
        """Read all JSON files from raw_dir, preprocess them, and write JSONL to output_path."""
        raw_path = Path(raw_dir)
        json_files = list(raw_path.rglob("*.json"))
        logger.info(f"Found {len(json_files)} raw JSON files in {raw_dir}")

        all_processed: List[Dict[str, Any]] = []
        for file in json_files:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    batch = json.load(f)
                    if isinstance(batch, list):
                        processed = self.process_batch(batch)
                        all_processed.extend(processed)
            except Exception as e:
                logger.error(f"Error reading {file}: {e}")

        # Save to output JSONL
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            for item in all_processed:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(f"Processed {len(all_processed)} posts. Saved to {output_path}")
        return all_processed


class SparkNLPPipeline:
    """Spark NLP Pipeline integration for distributed large-scale preprocessing."""

    @staticmethod
    def is_spark_available() -> bool:
        try:
            import pyspark
            return True
        except ImportError:
            return False

    @classmethod
    def create_spark_session(cls, app_name: str = "SocialMediaSparkNLP"):
        """Create a PySpark session configured for text processing."""
        if not cls.is_spark_available():
            raise RuntimeError("PySpark is not installed in the environment.")
        from pyspark.sql import SparkSession
        return SparkSession.builder \
            .appName(app_name) \
            .config("spark.driver.memory", "4g") \
            .getOrCreate()
