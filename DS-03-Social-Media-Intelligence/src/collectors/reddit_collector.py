"""Reddit Ingestion Module using PRAW with fallback to public JSON endpoints."""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional
import requests

from .base_collector import BaseCollector, SocialPost

logger = logging.getLogger(__name__)


class RedditCollector(BaseCollector):
    """Collects posts and comment interaction trees from Reddit."""

    def __init__(
        self,
        output_dir: str = "data/raw/reddit",
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None,
        subreddits: Optional[List[str]] = None
    ):
        super().__init__(output_dir)
        self.client_id = client_id or os.getenv("REDDIT_CLIENT_ID", "")
        self.client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET", "")
        self.user_agent = user_agent or os.getenv("REDDIT_USER_AGENT", "SocialPulse/1.0")
        self.subreddits = subreddits or ["technology", "worldnews", "science", "fakenews"]
        self.reddit = None

        if self.client_id and self.client_secret:
            try:
                import praw
                self.reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent
                )
                logger.info("Reddit PRAW client initialized in read-only mode.")
            except Exception as e:
                logger.warning(f"Could not initialize PRAW client: {e}. Will use public JSON fallback.")

    def collect(self, limit: int = 25) -> List[SocialPost]:
        """Collect hot posts and top comments from target subreddits."""
        all_posts: List[SocialPost] = []

        for sub in self.subreddits:
            logger.info(f"Collecting posts from r/{sub}...")
            if self.reddit:
                posts = self._collect_with_praw(sub, limit=limit)
            else:
                posts = self._collect_with_public_json(sub, limit=limit)
            all_posts.extend(posts)

        if all_posts:
            self.save_raw_json(all_posts, filename_prefix="reddit_batch")
        return all_posts

    def _collect_with_praw(self, subreddit_name: str, limit: int) -> List[SocialPost]:
        """Fetch posts and comments using PRAW."""
        posts = []
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            for submission in subreddit.hot(limit=limit):
                created_iso = datetime.fromtimestamp(submission.created_utc, timezone.utc).isoformat()
                author_name = str(submission.author) if submission.author else "[deleted]"

                post = SocialPost(
                    post_id=f"reddit_{submission.id}",
                    platform="reddit",
                    author=author_name,
                    created_at=created_iso,
                    title=submission.title or "",
                    content=submission.selftext or "",
                    url=f"https://reddit.com{submission.permalink}",
                    metrics={
                        "score": submission.score,
                        "upvote_ratio": getattr(submission, "upvote_ratio", 1.0),
                        "num_comments": submission.num_comments,
                        "subreddit": subreddit_name
                    },
                    raw_metadata={
                        "subreddit": subreddit_name,
                        "over_18": submission.over_18,
                        "is_original_content": submission.is_original_content
                    }
                )
                posts.append(post)

                # Fetch top comments for user interaction graph
                try:
                    submission.comments.replace_more(limit=0)
                    for comment in submission.comments[:5]:
                        comment_author = str(comment.author) if comment.author else "[deleted]"
                        comment_iso = datetime.fromtimestamp(comment.created_utc, timezone.utc).isoformat()
                        comment_post = SocialPost(
                            post_id=f"reddit_comment_{comment.id}",
                            platform="reddit",
                            author=comment_author,
                            created_at=comment_iso,
                            title=f"Re: {submission.title[:60]}",
                            content=comment.body or "",
                            parent_id=f"reddit_{submission.id}",
                            target_author=author_name,
                            url=f"https://reddit.com{comment.permalink}",
                            metrics={
                                "score": comment.score,
                                "upvote_ratio": 1.0,
                                "num_comments": 0,
                                "subreddit": subreddit_name
                            }
                        )
                        posts.append(comment_post)
                except Exception as c_err:
                    logger.debug(f"Could not parse comments for submission {submission.id}: {c_err}")

        except Exception as e:
            logger.error(f"Error collecting from r/{subreddit_name} via PRAW: {e}")
        return posts

    def _collect_with_public_json(self, subreddit_name: str, limit: int) -> List[SocialPost]:
        """Fetch posts via public reddit JSON endpoint with rate limits."""
        posts = []
        url = f"https://www.reddit.com/r/{subreddit_name}/hot.json?limit={limit}"
        headers = {"User-Agent": self.user_agent}

        try:
            time.sleep(1.0)  # Politeness delay
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                children = data.get("data", {}).get("children", [])
                for item in children:
                    d = item.get("data", {})
                    created_iso = datetime.fromtimestamp(d.get("created_utc", time.time()), timezone.utc).isoformat()
                    author_name = d.get("author", "[deleted]")

                    post = SocialPost(
                        post_id=f"reddit_{d.get('id', '')}",
                        platform="reddit",
                        author=author_name,
                        created_at=created_iso,
                        title=d.get("title", ""),
                        content=d.get("selftext", ""),
                        url=f"https://reddit.com{d.get('permalink', '')}",
                        metrics={
                            "score": d.get("score", 0),
                            "upvote_ratio": d.get("upvote_ratio", 1.0),
                            "num_comments": d.get("num_comments", 0),
                            "subreddit": subreddit_name
                        },
                        raw_metadata={
                            "subreddit": subreddit_name,
                            "domain": d.get("domain", "")
                        }
                    )
                    posts.append(post)
            else:
                logger.warning(f"Reddit public API returned status {resp.status_code} for r/{subreddit_name}")
        except Exception as e:
            logger.error(f"Failed to fetch public JSON for r/{subreddit_name}: {e}")

        return posts
