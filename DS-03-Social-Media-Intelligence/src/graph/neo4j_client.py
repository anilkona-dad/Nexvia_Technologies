"""Neo4j Graph Database Driver and Cypher Ingestion Module."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import networkx as nx

logger = logging.getLogger(__name__)

try:
    from neo4j import GraphDatabase
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False


class Neo4jGraphManager:
    """Manages graph persistence and Cypher queries in Neo4j."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "socialpulse2026"
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None

        if HAS_NEO4J:
            try:
                self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
                # Quick connectivity verification
                with self.driver.session() as session:
                    session.run("RETURN 1")
                logger.info(f"Connected to Neo4j instance at {self.uri}")
            except Exception as e:
                logger.info(f"Neo4j live connection not reachable ({e}). Offline Cypher export enabled.")
                self.driver = None

    def close(self):
        """Close Neo4j driver connection."""
        if self.driver:
            self.driver.close()

    def sync_networkx_graph(
        self,
        G: nx.DiGraph,
        user_metrics: Dict[str, Dict[str, Any]],
        export_cypher_path: Optional[str] = "data/neo4j_ingest.cypher"
    ):
        """Sync interaction graph to live Neo4j and generate standalone Cypher script."""
        cypher_lines = [
            "// --- Schema Constraints ---",
            "CREATE CONSTRAINT user_username_unique IF NOT EXISTS FOR (u:User) REQUIRE u.username IS UNIQUE;",
            "",
            "// --- User Nodes Ingestion ---"
        ]

        # 1. User nodes
        for username, metrics in user_metrics.items():
            line = (
                f"MERGE (u:User {{username: '{username}'}}) "
                f"SET u.pagerank = {metrics.get('pagerank', 0.0)}, "
                f"u.community_id = {metrics.get('community_id', 0)}, "
                f"u.in_degree = {metrics.get('in_degree', 0)}, "
                f"u.posts_count = {metrics.get('posts_count', 0)};"
            )
            cypher_lines.append(line)

        cypher_lines.append("\n// --- Interaction Relationships ---")

        # 2. Edges
        for u, v, data in G.edges(data=True):
            weight = data.get("weight", 1.0)
            replies = data.get("reply_count", 0)
            line = (
                f"MATCH (source:User {{username: '{u}'}}), (target:User {{username: '{v}'}}) "
                f"MERGE (source)-[r:INTERACTS_WITH]->(target) "
                f"SET r.weight = {weight}, r.reply_count = {replies};"
            )
            cypher_lines.append(line)

        # Write Cypher dump file
        if export_cypher_path:
            dump_file = Path(export_cypher_path)
            dump_file.parent.mkdir(parents=True, exist_ok=True)
            with open(dump_file, "w", encoding="utf-8") as f:
                f.write("\n".join(cypher_lines))
            logger.info(f"Exported {len(cypher_lines)} Cypher statements to {export_cypher_path}")

        # If live connection is active, execute statements
        if self.driver:
            try:
                with self.driver.session() as session:
                    for line in cypher_lines:
                        if line.strip() and not line.startswith("//"):
                            session.run(line)
                logger.info("Successfully synced interaction graph into live Neo4j database.")
            except Exception as e:
                logger.warning(f"Error executing Cypher queries in live Neo4j: {e}")
