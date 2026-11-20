"""Realistic Synthetic Data Generator for Offline Pipeline Development & Demonstrations."""

import logging
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from .base_collector import BaseCollector, SocialPost

logger = logging.getLogger(__name__)


class MockSocialDataGenerator(BaseCollector):
    """Generates rich synthetic social media posts and reply networks across multiple topics."""

    INFLUENCER_USERS = [
        "TechPioneer_Alpha", "QuantumLeap_AI", "TruthSeeker_99",
        "GlobalNews_Watcher", "ClimateObserver", "CyberSentinel"
    ]

    COMMUNITY_USERS = [
        "dev_enthusiast", "data_wizard", "curious_cat", "sceptic_voice",
        "future_tech_fan", "elena_es", "pierre_fr", "hans_de",
        "code_ninja", "pixel_artist", "science_nerd", "policy_analyst",
        "skeptical_user42", "optimist_prime", "eco_warrior", "digital_nomad"
    ]

    SAMPLE_CORPUS = [
        # AI & Tech (Positive / Neutral)
        {
            "topic": "AI & Robotics",
            "lang": "en",
            "title": "Breakthrough in autonomous robotics achieves human-level dexterity 🤖🚀",
            "content": "Researchers have revealed a new foundation model for manipulation that learns dexterity in zero-shot environments. This is a game changer! 🔥",
            "sentiment_hint": "positive",
            "fake_news": False
        },
        {
            "topic": "AI & Robotics",
            "lang": "en",
            "title": "Is open-source AI surpassing proprietary models? Deep dive analysis 💻",
            "content": "Benchmarking the latest 70B open weights against enterprise APIs. The cost-to-performance ratio is shifting dramatically in favor of self-hosting.",
            "sentiment_hint": "neutral",
            "fake_news": False
        },
        {
            "topic": "AI & Robotics",
            "lang": "es",
            "title": "Avance revolucionario en inteligencia artificial para diagnóstico médico 🩺👏",
            "content": "El nuevo modelo de visión computacional detecta anomalías tempranas con 98% de precisión en hospitales públicos. ¡Una gran esperanza!",
            "sentiment_hint": "positive",
            "fake_news": False
        },
        # Climate & Renewable Energy
        {
            "topic": "Climate & Energy",
            "lang": "en",
            "title": "Solar capacity surpasses coal generation in major power grid milestones ☀️⚡",
            "content": "Record investment in utility-scale battery storage has enabled renewable power to dominate peak energy hours this summer. Clean energy future is here!",
            "sentiment_hint": "positive",
            "fake_news": False
        },
        {
            "topic": "Climate & Energy",
            "lang": "fr",
            "title": "Transition énergétique: de nouveaux investissements massifs dans l'éolien offshore 🌊💨",
            "content": "L'Union Européenne annonce un consortium pour tripler les capacités éoliennes en mer d'ici 2030. Un projet ambitieux pour le climat.",
            "sentiment_hint": "positive",
            "fake_news": False
        },
        # Misinformation / Fake News items (Negative / Alarmist)
        {
            "topic": "Misinformation / Conspiracies",
            "lang": "en",
            "title": "URGENT: Government secret directive bans all cash transactions by next month! 🚨😱",
            "content": "Leaked confidential memos prove central banks will seize all physical currency next Tuesday. Share this before they delete it! Absolute tyranny! 💥",
            "sentiment_hint": "negative",
            "fake_news": True
        },
        {
            "topic": "Misinformation / Conspiracies",
            "lang": "en",
            "title": "Secret miracle mineral cures 100% of degenerative diseases instantly! ⚠️💊",
            "content": "Big pharma is hiding the simple salt crystalline cure found in backyard dirt. Doctors are being silenced! Buy the extract now at miracle-cure-scam.xyz",
            "sentiment_hint": "negative",
            "fake_news": True
        },
        {
            "topic": "Misinformation / Conspiracies",
            "lang": "de",
            "title": "Achtung: Trinkwasser angeblich mit Gedankenkontroll-Chemikalien versetzt! 🛑☣️",
            "content": "Geheime Dokumente enthüllen angeblich dubiose Zusätze im Leitungswasser großer Städte. Niemand berichtet darüber! Teilt diese Warnung!",
            "sentiment_hint": "negative",
            "fake_news": True
        },
        # Cybersecurity & Policy
        {
            "topic": "Cybersecurity",
            "lang": "en",
            "title": "Critical zero-day vulnerability patched in popular network appliances 🛡️🔒",
            "content": "Security agencies urge immediate firmware update across edge routers following widespread automated scanning activity.",
            "sentiment_hint": "neutral",
            "fake_news": False
        },
        {
            "topic": "Cybersecurity",
            "lang": "en",
            "title": "Ransomware group targets municipal hospital system in devastating attack 💔🚑",
            "content": "Emergency rooms diverted patients as critical databases were encrypted. Cyber incident response teams are working round the clock.",
            "sentiment_hint": "negative",
            "fake_news": False
        }
    ]

    REPLY_TEMPLATES = [
        "Completely agree with @{target}! The evidence is undeniable. 👏",
        "Interesting point @{target}, but have you considered the counterarguments?",
        "This looks like total fake news @{target}. Do you have any verified peer-reviewed sources?",
        "Thanks for sharing this @{target}! Forwarding to our entire engineering team. 🚀",
        "I am skeptical about these numbers @{target}. Correlation is not causation.",
        "Spot on! We are witnessing history unfold right in front of our eyes. 🔥"
    ]

    def __init__(self, output_dir: str = "data/raw"):
        super().__init__(output_dir)

    def collect(self, limit: int = 150) -> List[SocialPost]:
        """Generate a realistic dataset of interconnected social posts and replies."""
        posts: List[SocialPost] = []
        random.seed(42)

        now = datetime.now(timezone.utc)
        all_authors = self.INFLUENCER_USERS + self.COMMUNITY_USERS

        # 1. Create root posts spread over the past 30 days
        root_posts: List[SocialPost] = []
        num_roots = max(10, int(limit * 0.4))

        for i in range(num_roots):
            item = random.choice(self.SAMPLE_CORPUS)
            # Pick influencer with 60% probability for roots
            author = random.choice(self.INFLUENCER_USERS if random.random() < 0.6 else self.COMMUNITY_USERS)
            platform = "reddit" if random.random() < 0.55 else "youtube"
            
            # Spread timestamps over past 30 days
            days_ago = random.uniform(0, 30)
            created_at = (now - timedelta(days=days_ago)).isoformat()
            post_id = f"{platform}_post_{1000 + i}"

            root_post = SocialPost(
                post_id=post_id,
                platform=platform,
                author=author,
                created_at=created_at,
                title=item["title"],
                content=item["content"],
                url=f"https://{platform}.com/posts/{1000 + i}",
                metrics={
                    "score": random.randint(10, 2500) if author in self.INFLUENCER_USERS else random.randint(1, 150),
                    "upvote_ratio": round(random.uniform(0.65, 0.99), 2),
                    "num_comments": 0,
                    "view_count": random.randint(500, 50000)
                },
                raw_metadata={
                    "topic_category": item["topic"],
                    "language": item["lang"],
                    "is_synthetic": True,
                    "ground_truth_fake": item["fake_news"]
                }
            )
            root_posts.append(root_post)

        posts.extend(root_posts)

        # 2. Create reply trees connecting authors (crucial for PageRank and Neo4j graph)
        replies_needed = limit - len(root_posts)
        for i in range(replies_needed):
            parent = random.choice(root_posts)
            reply_author = random.choice([u for u in all_authors if u != parent.author])
            parent_dt = datetime.fromisoformat(parent.created_at)
            reply_dt = parent_dt + timedelta(hours=random.uniform(0.5, 48))
            if reply_dt > now:
                reply_dt = now

            reply_text = random.choice(self.REPLY_TEMPLATES).format(target=parent.author)
            reply_id = f"{parent.platform}_reply_{2000 + i}"

            reply_post = SocialPost(
                post_id=reply_id,
                platform=parent.platform,
                author=reply_author,
                created_at=reply_dt.isoformat(),
                title=f"Re: {parent.title[:40]}...",
                content=reply_text,
                parent_id=parent.post_id,
                target_author=parent.author,
                url=f"{parent.url}#reply_{2000 + i}",
                metrics={
                    "score": random.randint(0, 80),
                    "upvote_ratio": round(random.uniform(0.5, 1.0), 2)
                },
                raw_metadata={
                    "topic_category": parent.raw_metadata.get("topic_category", "General"),
                    "is_reply": True,
                    "ground_truth_fake": False
                }
            )
            posts.append(reply_post)

        # Save to disk
        self.save_raw_json(posts, filename_prefix="synthetic_corpus")
        logger.info(f"Generated {len(posts)} realistic synthetic posts and replies.")
        return posts
