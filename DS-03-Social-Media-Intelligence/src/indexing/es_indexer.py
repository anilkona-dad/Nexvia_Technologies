"""Elasticsearch Ingestion and Search Module."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

logger = logging.getLogger(__name__)

try:
    from elasticsearch import Elasticsearch, helpers
    HAS_ELASTICSEARCH = True
except ImportError:
    HAS_ELASTICSEARCH = False


class ElasticsearchIndexer:
    """Manages index templates, bulk document ingestion, and queries in Elasticsearch."""

    def __init__(
        self,
        hosts: Optional[List[str]] = None,
        index_name: str = "social-pulse-posts",
        mapping_file: str = "docker/elasticsearch/mappings.json"
    ):
        self.hosts = hosts or ["http://localhost:9200"]
        self.index_name = index_name
        self.mapping_file = Path(mapping_file)
        self.es = None

        if HAS_ELASTICSEARCH:
            try:
                self.es = Elasticsearch(self.hosts, request_timeout=10)
                if self.es.ping():
                    logger.info(f"Connected to Elasticsearch cluster at {self.hosts}")
                else:
                    self.es = None
            except Exception as e:
                logger.info(f"Elasticsearch live cluster not reachable ({e}). Bulk export mode enabled.")
                self.es = None

    def initialize_index(self):
        """Create Elasticsearch index with mappings if not already present."""
        if not self.es:
            return

        try:
            if not self.es.indices.exists(index=self.index_name):
                mappings = {}
                if self.mapping_file.exists():
                    with open(self.mapping_file, "r", encoding="utf-8") as f:
                        mappings = json.load(f)
                self.es.indices.create(index=self.index_name, body=mappings)
                logger.info(f"Created index '{self.index_name}' with custom mappings.")
        except Exception as e:
            logger.error(f"Failed to create index '{self.index_name}': {e}")

    def bulk_index_posts(
        self,
        posts: List[Dict[str, Any]],
        export_file_path: Optional[str] = "data/elasticsearch_bulk.jsonl"
    ) -> int:
        """Index posts into Elasticsearch in bulk and save bulk NDJSON file."""
        if not posts:
            return 0

        # Prepare NDJSON bulk payload
        ndjson_lines = []
        actions = []

        for p in posts:
            doc_id = p.get("post_id", "")
            action_meta = {"index": {"_index": self.index_name, "_id": doc_id}}
            ndjson_lines.append(json.dumps(action_meta))
            ndjson_lines.append(json.dumps(p, ensure_ascii=False))

            actions.append({
                "_index": self.index_name,
                "_id": doc_id,
                "_source": p
            })

        # Persist bulk file
        if export_file_path:
            dump_file = Path(export_file_path)
            dump_file.parent.mkdir(parents=True, exist_ok=True)
            with open(dump_file, "w", encoding="utf-8") as f:
                f.write("\n".join(ndjson_lines) + "\n")
            logger.info(f"Saved {len(posts)} documents to bulk NDJSON at {export_file_path}")

        # Send to live cluster if available
        indexed_count = 0
        if self.es:
            try:
                success, errors = helpers.bulk(self.es, actions, raise_on_error=False)
                indexed_count = success
                logger.info(f"Successfully bulk indexed {success} documents into Elasticsearch.")
            except Exception as e:
                logger.warning(f"Error during live bulk indexing: {e}")
        else:
            # Fallback HTTP request attempt
            try:
                bulk_url = f"{self.hosts[0]}/_bulk"
                payload = "\n".join(ndjson_lines) + "\n"
                resp = requests.post(
                    bulk_url,
                    data=payload,
                    headers={"Content-Type": "application/x-ndjson"},
                    timeout=5
                )
                if resp.status_code in [200, 201]:
                    indexed_count = len(posts)
                    logger.info(f"Bulk indexed {len(posts)} docs via REST API.")
            except Exception:
                pass

        return indexed_count

    def search(
        self,
        query: str = "",
        sentiment_label: Optional[str] = None,
        risk_level: Optional[str] = None,
        size: int = 10
    ) -> List[Dict[str, Any]]:
        """Search posts by text query and filters."""
        if not self.es:
            logger.warning("Elasticsearch instance not connected.")
            return []

        must_clauses = []
        if query:
            must_clauses.append({"multi_match": {"query": query, "fields": ["title", "content", "cleaned_text"]}})
        if sentiment_label:
            must_clauses.append({"term": {"sentiment.label": sentiment_label}})
        if risk_level:
            must_clauses.append({"term": {"misinformation.risk_level": risk_level}})

        body = {
            "query": {"bool": {"must": must_clauses}} if must_clauses else {"match_all": {}},
            "size": size,
            "sort": [{"created_at": {"order": "desc"}}]
        }

        try:
            res = self.es.search(index=self.index_name, body=body)
            hits = [h["_source"] for h in res["hits"]["hits"]]
            return hits
        except Exception as e:
            logger.error(f"Search query failed: {e}")
            return []
