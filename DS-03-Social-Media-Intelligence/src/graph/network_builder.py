"""NetworkX Interaction Graph Construction Module."""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


class InteractionGraphBuilder:
    """Constructs directed user-interaction graphs from social media posts and reply threads."""

    @classmethod
    def build_graph(cls, posts: List[Dict[str, Any]]) -> nx.DiGraph:
        """Construct directed graph where edges represent replies and mentions."""
        G = nx.DiGraph()

        # Track user stats
        user_posts = defaultdict(int)
        user_scores = defaultdict(int)
        user_platforms = defaultdict(set)

        for p in posts:
            author = p.get("author")
            if not author or author in ["[deleted]", "Anonymous", "Unknown Channel"]:
                continue

            platform = p.get("platform", "unknown")
            score = p.get("metrics", {}).get("score", 0)

            user_posts[author] += 1
            user_scores[author] += score
            user_platforms[author].add(platform)

            # 1. Edge: Reply interaction (author -> target_author)
            target = p.get("target_author")
            if target and target != author and target not in ["[deleted]", "Anonymous"]:
                if G.has_edge(author, target):
                    G[author][target]["weight"] += 1.0
                    G[author][target]["reply_count"] += 1
                else:
                    G.add_edge(author, target, weight=1.0, reply_count=1, mention_count=0)

            # 2. Edge: Mention interaction (author -> mentioned_user)
            mentions = p.get("mentions", [])
            for m in mentions:
                if m != author:
                    if G.has_edge(author, m):
                        G[author][m]["weight"] += 0.5
                        G[author][m]["mention_count"] += 1
                    else:
                        G.add_edge(author, m, weight=0.5, reply_count=0, mention_count=1)

        # Attach node attributes
        for user in G.nodes():
            G.nodes[user]["posts_count"] = user_posts[user]
            G.nodes[user]["cumulative_score"] = user_scores[user]
            G.nodes[user]["platforms"] = list(user_platforms[user])

        logger.info(f"Built interaction graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
        return G
