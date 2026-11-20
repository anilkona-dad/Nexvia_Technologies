"""Base Collector for Social Media Ingestion."""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SocialPost:
    """Standardized representation of a social media post/comment."""
    post_id: str
    platform: str  # "reddit" | "youtube" | "synthetic"
    author: str
    created_at: str  # ISO 8601 string
    title: str
    content: str
    url: str = ""
    parent_id: Optional[str] = None
    target_author: Optional[str] = None  # user being replied to
    metrics: Dict[str, Any] = field(default_factory=lambda: {
        "score": 0,
        "upvote_ratio": 1.0,
        "num_comments": 0,
        "view_count": 0
    })
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseCollector(ABC):
    """Abstract Base Class for social media data collectors."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def collect(self, limit: int = 50) -> List[SocialPost]:
        """Collect data from the upstream platform."""
        pass

    def save_raw_json(self, posts: List[SocialPost], filename_prefix: str = "batch") -> Path:
        """Persist raw extracted posts to partitioned JSON storage."""
        now = datetime.now(timezone.utc)
        date_folder = self.output_dir / now.strftime("%Y-%m-%d")
        date_folder.mkdir(parents=True, exist_ok=True)

        timestamp_str = now.strftime("%H%M%S")
        file_path = date_folder / f"{filename_prefix}_{timestamp_str}.json"

        data = [p.to_dict() for p in posts]
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(posts)} posts to {file_path}")
        return file_path
