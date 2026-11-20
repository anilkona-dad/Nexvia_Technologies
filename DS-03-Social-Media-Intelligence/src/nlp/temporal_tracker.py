"""Sentiment Temporal Tracking Module: analyzes sentiment trends over time."""

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List


class TemporalSentimentTracker:
    """Tracks and aggregates sentiment dynamics across time windows."""

    @classmethod
    def track_sentiment_over_time(
        cls,
        posts: List[Dict[str, Any]],
        interval: str = "day"  # "day" or "hour"
    ) -> List[Dict[str, Any]]:
        """Group posts by time interval and calculate temporal sentiment metrics."""
        buckets = defaultdict(lambda: {
            "total_count": 0,
            "positive_count": 0,
            "neutral_count": 0,
            "negative_count": 0,
            "polarity_sum": 0.0,
            "confidences": []
        })

        for p in posts:
            created_str = p.get("created_at")
            if not created_str:
                continue

            try:
                # Handle ISO timestamps with Z or offsets
                clean_time = created_str.replace("Z", "+00:00")
                dt = datetime.fromisoformat(clean_time)
            except Exception:
                continue

            if interval == "hour":
                bucket_key = dt.strftime("%Y-%m-%d %H:00")
            else:
                bucket_key = dt.strftime("%Y-%m-%d")

            sent = p.get("sentiment", {})
            label = sent.get("label", "neutral")
            polarity = sent.get("polarity", 0.0)
            conf = sent.get("confidence", 0.5)

            b = buckets[bucket_key]
            b["total_count"] += 1
            if label == "positive":
                b["positive_count"] += 1
            elif label == "negative":
                b["negative_count"] += 1
            else:
                b["neutral_count"] += 1

            b["polarity_sum"] += polarity
            b["confidences"].append(conf)

        # Build chronological timeline
        timeline: List[Dict[str, Any]] = []
        prev_avg_polarity = None

        for bucket_key in sorted(buckets.keys()):
            b = buckets[bucket_key]
            count = b["total_count"]
            avg_polarity = round(b["polarity_sum"] / count, 3) if count > 0 else 0.0
            avg_conf = round(sum(b["confidences"]) / count, 3) if count > 0 else 0.0

            # Sentiment velocity (delta from previous window)
            velocity = 0.0
            if prev_avg_polarity is not None:
                velocity = round(avg_polarity - prev_avg_polarity, 3)
            prev_avg_polarity = avg_polarity

            timeline.append({
                "timestamp_window": bucket_key,
                "post_volume": count,
                "mean_polarity": avg_polarity,
                "sentiment_velocity": velocity,
                "positive_count": b["positive_count"],
                "neutral_count": b["neutral_count"],
                "negative_count": b["negative_count"],
                "avg_confidence": avg_conf
            })

        return timeline
