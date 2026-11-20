"""Kibana Dashboard and Visualizations Generator."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class KibanaDashboardExporter:
    """Generates importable Kibana Saved Objects (Dashboards, Visualizations, and Alerts)."""

    INDEX_PATTERN_ID = "social-pulse-index-pattern"
    DASHBOARD_ID = "social-pulse-executive-dashboard"

    @classmethod
    def export_saved_objects(cls, output_path: str = "data/kibana_saved_objects.ndjson"):
        """Generate NDJSON file containing Kibana Dashboard and Visualizations."""
        objects = []

        # 1. Index Pattern
        objects.append({
            "type": "index-pattern",
            "id": cls.INDEX_PATTERN_ID,
            "attributes": {
                "title": "social-pulse-posts*",
                "timeFieldName": "created_at",
                "fields": "[]"
            }
        })

        # 2. Visualization: Sentiment Trends Over Time
        objects.append({
            "type": "visualization",
            "id": "viz-sentiment-timeline",
            "attributes": {
                "title": "Sentiment Trajectory Over Time",
                "visState": json.dumps({
                    "title": "Sentiment Trajectory Over Time",
                    "type": "line",
                    "params": {
                        "type": "line",
                        "grid": {"categoryLines": False},
                        "categoryAxes": [{"id": "CategoryAxis-1", "type": "category", "position": "bottom"}]
                    },
                    "aggs": [
                        {"id": "1", "enabled": True, "type": "avg", "schema": "metric", "params": {"field": "sentiment.polarity"}},
                        {"id": "2", "enabled": True, "type": "date_histogram", "schema": "segment", "params": {"field": "created_at", "interval": "d"}}
                    ]
                }),
                "kibanaSavedObjectMeta": {
                    "searchSourceJSON": json.dumps({"indexRefName": "kibanaSavedObjectMeta.searchSourceJSON.index"})
                }
            },
            "references": [{"name": "kibanaSavedObjectMeta.searchSourceJSON.index", "type": "index-pattern", "id": cls.INDEX_PATTERN_ID}]
        })

        # 3. Visualization: BERTopic Dynamic Distribution
        objects.append({
            "type": "visualization",
            "id": "viz-topic-distribution",
            "attributes": {
                "title": "Dynamic Topic Rise & Fall",
                "visState": json.dumps({
                    "title": "Dynamic Topic Rise & Fall",
                    "type": "area",
                    "params": {"type": "area", "mode": "stacked"},
                    "aggs": [
                        {"id": "1", "enabled": True, "type": "count", "schema": "metric"},
                        {"id": "2", "enabled": True, "type": "terms", "schema": "group", "params": {"field": "topic.topic_name.keyword", "size": 5}},
                        {"id": "3", "enabled": True, "type": "date_histogram", "schema": "segment", "params": {"field": "created_at", "interval": "w"}}
                    ]
                }),
                "kibanaSavedObjectMeta": {
                    "searchSourceJSON": json.dumps({"indexRefName": "kibanaSavedObjectMeta.searchSourceJSON.index"})
                }
            },
            "references": [{"name": "kibanaSavedObjectMeta.searchSourceJSON.index", "type": "index-pattern", "id": cls.INDEX_PATTERN_ID}]
        })

        # 4. Visualization: Misinformation Radar Table
        objects.append({
            "type": "visualization",
            "id": "viz-misinformation-alerts",
            "attributes": {
                "title": "Misinformation & Fake News Alert Radar",
                "visState": json.dumps({
                    "title": "Misinformation & Fake News Alert Radar",
                    "type": "table",
                    "aggs": [
                        {"id": "1", "enabled": True, "type": "terms", "schema": "bucket", "params": {"field": "misinformation.risk_level.keyword", "size": 5}},
                        {"id": "2", "enabled": True, "type": "avg", "schema": "metric", "params": {"field": "misinformation.fake_probability"}}
                    ]
                }),
                "kibanaSavedObjectMeta": {
                    "searchSourceJSON": json.dumps({"indexRefName": "kibanaSavedObjectMeta.searchSourceJSON.index"})
                }
            },
            "references": [{"name": "kibanaSavedObjectMeta.searchSourceJSON.index", "type": "index-pattern", "id": cls.INDEX_PATTERN_ID}]
        })

        # 5. Visualization: Influencer PageRank Table
        objects.append({
            "type": "visualization",
            "id": "viz-influencer-pagerank",
            "attributes": {
                "title": "Top Influencer PageRank Leaderboard",
                "visState": json.dumps({
                    "title": "Top Influencer PageRank Leaderboard",
                    "type": "table",
                    "aggs": [
                        {"id": "1", "enabled": True, "type": "terms", "schema": "bucket", "params": {"field": "author.keyword", "size": 10}},
                        {"id": "2", "enabled": True, "type": "max", "schema": "metric", "params": {"field": "graph_metrics.author_pagerank"}}
                    ]
                }),
                "kibanaSavedObjectMeta": {
                    "searchSourceJSON": json.dumps({"indexRefName": "kibanaSavedObjectMeta.searchSourceJSON.index"})
                }
            },
            "references": [{"name": "kibanaSavedObjectMeta.searchSourceJSON.index", "type": "index-pattern", "id": cls.INDEX_PATTERN_ID}]
        })

        # 6. Master Dashboard
        objects.append({
            "type": "dashboard",
            "id": cls.DASHBOARD_ID,
            "attributes": {
                "title": "Social Media Intelligence Executive Dashboard",
                "description": "Unified dashboard tracking sentiment, BERTopic dynamics, PageRank influencers, and fake news.",
                "panelsJSON": json.dumps([
                    {"panelIndex": "1", "gridData": {"x": 0, "y": 0, "w": 24, "h": 12, "i": "1"}, "panelRefName": "panel_1"},
                    {"panelIndex": "2", "gridData": {"x": 24, "y": 0, "w": 24, "h": 12, "i": "2"}, "panelRefName": "panel_2"},
                    {"panelIndex": "3", "gridData": {"x": 0, "y": 12, "w": 24, "h": 12, "i": "3"}, "panelRefName": "panel_3"},
                    {"panelIndex": "4", "gridData": {"x": 24, "y": 12, "w": 24, "h": 12, "i": "4"}, "panelRefName": "panel_4"}
                ]),
                "optionsJSON": json.dumps({"useMargins": True, "hidePanelTitles": False})
            },
            "references": [
                {"name": "panel_1", "type": "visualization", "id": "viz-sentiment-timeline"},
                {"name": "panel_2", "type": "visualization", "id": "viz-topic-distribution"},
                {"name": "panel_3", "type": "visualization", "id": "viz-influencer-pagerank"},
                {"name": "panel_4", "type": "visualization", "id": "viz-misinformation-alerts"}
            ]
        })

        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            for obj in objects:
                f.write(json.dumps(obj) + "\n")

        logger.info(f"Exported Kibana Dashboard Saved Objects to {output_path}")
        return output_path
