"""YouTube Data API v3 Ingestion Module."""

import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .base_collector import BaseCollector, SocialPost

logger = logging.getLogger(__name__)


class YouTubeCollector(BaseCollector):
    """Collects video descriptions and comment interaction threads from YouTube."""

    def __init__(
        self,
        output_dir: str = "data/raw/youtube",
        api_key: Optional[str] = None,
        queries: Optional[List[str]] = None
    ):
        super().__init__(output_dir)
        self.api_key = api_key or os.getenv("YOUTUBE_API_KEY", "")
        self.queries = queries or ["technology trends", "artificial intelligence", "breaking science"]
        self.youtube = None

        if self.api_key:
            try:
                from googleapiclient.discovery import build
                self.youtube = build("youtube", "v3", developerKey=self.api_key)
                logger.info("YouTube Data API v3 client successfully initialized.")
            except Exception as e:
                logger.warning(f"Failed to initialize YouTube client: {e}")

    def collect(self, limit: int = 15) -> List[SocialPost]:
        """Search videos and extract video metadata + top comments."""
        if not self.youtube:
            logger.warning("No YouTube API client available (missing YOUTUBE_API_KEY). Skipping live collection.")
            return []

        all_posts: List[SocialPost] = []
        for query in self.queries:
            logger.info(f"Querying YouTube for '{query}'...")
            try:
                search_response = self.youtube.search().list(
                    q=query,
                    part="id,snippet",
                    maxResults=min(limit, 25),
                    type="video"
                ).execute()

                for item in search_response.get("items", []):
                    video_id = item["id"]["videoId"]
                    snippet = item["snippet"]

                    post = SocialPost(
                        post_id=f"yt_video_{video_id}",
                        platform="youtube",
                        author=snippet.get("channelTitle", "Unknown Channel"),
                        created_at=snippet.get("publishedAt", datetime.now(timezone.utc).isoformat()),
                        title=snippet.get("title", ""),
                        content=snippet.get("description", ""),
                        url=f"https://www.youtube.com/watch?v={video_id}",
                        metrics={
                            "query": query,
                            "channel_id": snippet.get("channelId", "")
                        },
                        raw_metadata={"kind": "video", "query": query}
                    )
                    all_posts.append(post)

                    # Extract top comment threads
                    comments = self._fetch_comments(video_id, max_comments=10)
                    all_posts.extend(comments)

            except Exception as e:
                logger.error(f"Error fetching YouTube data for query '{query}': {e}")

        if all_posts:
            self.save_raw_json(all_posts, filename_prefix="youtube_batch")
        return all_posts

    def _fetch_comments(self, video_id: str, max_comments: int = 10) -> List[SocialPost]:
        """Fetch comment threads for a video."""
        comments: List[SocialPost] = []
        try:
            thread_response = self.youtube.commentThreads().list(
                videoId=video_id,
                part="snippet",
                maxResults=max_comments,
                textFormat="plainText"
            ).execute()

            for item in thread_response.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                author = snippet.get("authorDisplayName", "Anonymous")
                comment_post = SocialPost(
                    post_id=f"yt_comment_{item['id']}",
                    platform="youtube",
                    author=author,
                    created_at=snippet.get("publishedAt", datetime.now(timezone.utc).isoformat()),
                    title=f"Comment on video {video_id}",
                    content=snippet.get("textDisplay", ""),
                    parent_id=f"yt_video_{video_id}",
                    url=f"https://www.youtube.com/watch?v={video_id}",
                    metrics={
                        "score": snippet.get("likeCount", 0),
                        "num_replies": item["snippet"].get("totalReplyCount", 0)
                    }
                )
                comments.append(comment_post)
        except Exception as e:
            logger.debug(f"Could not load comments for video {video_id}: {e}")

        return comments
