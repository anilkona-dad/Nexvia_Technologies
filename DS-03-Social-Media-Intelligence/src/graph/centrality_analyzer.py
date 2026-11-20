"""Centrality, PageRank, and Community Detection Module."""

import logging
from typing import Any, Dict, List, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


class CentralityAnalyzer:
    """Computes PageRank influencer scores and community clustering on user interaction graphs."""

    @classmethod
    def analyze_graph(
        cls,
        G: nx.DiGraph,
        damping_factor: float = 0.85
    ) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
        """Compute PageRank, degree centrality, betweenness, and community partitions.
        Returns:
            user_metrics: mapping of username -> metrics
            communities: list of community summaries
        """
        if G.number_of_nodes() == 0:
            return {}, []

        # 1. PageRank algorithm
        try:
            pagerank = nx.pagerank(G, alpha=damping_factor, weight="weight")
        except Exception as e:
            logger.warning(f"PageRank computation defaulted: {e}")
            pagerank = {node: 1.0 / len(G) for node in G.nodes()}

        # 2. In-degree and Out-degree
        in_degree = dict(G.in_degree())
        out_degree = dict(G.out_degree())

        # 3. Betweenness Centrality
        try:
            betweenness = nx.betweenness_centrality(G, weight="weight")
        except Exception:
            betweenness = {node: 0.0 for node in G.nodes()}

        # 4. Community Detection (convert to undirected for modularity clustering)
        undirected_G = G.to_undirected()
        try:
            from networkx.algorithms.community import louvain_communities
            raw_communities = louvain_communities(undirected_G, weight="weight", seed=42)
        except (ImportError, AttributeError):
            try:
                from networkx.algorithms.community import greedy_modularity_communities
                raw_communities = list(greedy_modularity_communities(undirected_G, weight="weight"))
            except Exception:
                raw_communities = [set(G.nodes())]

        # Map node to community ID
        node_to_comm: Dict[str, int] = {}
        community_summaries: List[Dict[str, Any]] = []

        for cid, comm_set in enumerate(raw_communities):
            members = list(comm_set)
            # Find top leader in this community by PageRank
            leader = max(members, key=lambda m: pagerank.get(m, 0.0))
            for member in members:
                node_to_comm[member] = cid

            community_summaries.append({
                "community_id": cid,
                "size": len(members),
                "leader": leader,
                "leader_pagerank": round(float(pagerank.get(leader, 0.0)), 5),
                "members": members[:10]  # top 10 preview
            })

        # Assemble per-user metrics
        user_metrics: Dict[str, Dict[str, Any]] = {}
        for node in G.nodes():
            user_metrics[node] = {
                "username": node,
                "pagerank": round(float(pagerank.get(node, 0.0)), 5),
                "in_degree": in_degree.get(node, 0),
                "out_degree": out_degree.get(node, 0),
                "betweenness": round(float(betweenness.get(node, 0.0)), 5),
                "community_id": node_to_comm.get(node, 0),
                "posts_count": G.nodes[node].get("posts_count", 0),
                "cumulative_score": G.nodes[node].get("cumulative_score", 0)
            }

        return user_metrics, community_summaries

    @classmethod
    def get_top_influencers(cls, user_metrics: Dict[str, Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        """Return leaderboard of top influencers sorted by PageRank."""
        sorted_users = sorted(user_metrics.values(), key=lambda u: u["pagerank"], reverse=True)
        return sorted_users[:top_k]

    @classmethod
    def enrich_posts_with_graph(
        cls,
        posts: List[Dict[str, Any]],
        user_metrics: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Enrich posts with author PageRank and community membership."""
        for p in posts:
            author = p.get("author")
            metrics = user_metrics.get(author, {})
            p["graph_metrics"] = {
                "author_pagerank": metrics.get("pagerank", 0.0),
                "author_in_degree": metrics.get("in_degree", 0),
                "community_id": metrics.get("community_id", -1)
            }
        return posts
