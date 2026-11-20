"""Social Media Intelligence Platform - Master CLI Orchestration Pipeline."""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("SocialPulse")

# Core imports
from src.collectors import MockSocialDataGenerator, RedditCollector, YouTubeCollector
from src.preprocessing import PreprocessingPipeline
from src.nlp import (
    MultilingualSentimentAnalyzer,
    TemporalSentimentTracker,
    TopicModeler,
    MisinformationClassifier
)
from src.graph import InteractionGraphBuilder, CentralityAnalyzer, Neo4jGraphManager
from src.indexing import ElasticsearchIndexer, KibanaDashboardExporter


class SocialMediaPipeline:
    """Master orchestrator executing the 7-milestone intelligence pipeline."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.models_dir = self.data_dir / "models"
        
        self.preprocessor = PreprocessingPipeline()
        self.sentiment_analyzer = MultilingualSentimentAnalyzer()
        self.topic_modeler = TopicModeler(nr_topics=5)
        self.misinfo_classifier = MisinformationClassifier(model_dir=str(self.models_dir))
        self.graph_manager = Neo4jGraphManager()
        self.es_indexer = ElasticsearchIndexer()

    def run_all(self, source: str = "mock", limit: int = 150):
        """Execute full end-to-end pipeline across all 7 milestones."""
        print("\n" + "="*70)
        print("  SOCIAL MEDIA INTELLIGENCE & MISINFORMATION PLATFORM")
        print("="*70)

        # -------------------------------------------------------------------------
        # Milestone 1: Data Collection & Raw Storage
        # -------------------------------------------------------------------------
        print("\n[*] Milestone 1: Ingesting Data & Saving Raw JSON...")
        if source == "reddit":
            collector = RedditCollector(output_dir=str(self.raw_dir / "reddit"))
        elif source == "youtube":
            collector = YouTubeCollector(output_dir=str(self.raw_dir / "youtube"))
        else:
            collector = MockSocialDataGenerator(output_dir=str(self.raw_dir / "synthetic"))

        raw_posts = collector.collect(limit=limit)
        print(f"    -> Ingested {len(raw_posts)} posts from source '{source}'.")

        # -------------------------------------------------------------------------
        # Milestone 2: Preprocessing, Text Cleaning, Emojis, Language Detection
        # -------------------------------------------------------------------------
        print("\n[*] Milestone 2: Cleaning Text, Handling Emojis & Detecting Language...")
        raw_dicts = [p.to_dict() for p in raw_posts]
        cleaned_posts = self.preprocessor.process_batch(raw_dicts)
        
        langs = set(p["language"] for p in cleaned_posts)
        emoji_count = sum(len(p.get("emojis", [])) for p in cleaned_posts)
        print(f"    -> Preprocessed {len(cleaned_posts)} posts.")
        print(f"    -> Languages detected: {', '.join(langs)}")
        print(f"    -> Total emojis extracted: {emoji_count}")

        # -------------------------------------------------------------------------
        # Milestone 3: Multilingual Sentiment Analysis & Temporal Tracking
        # -------------------------------------------------------------------------
        print("\n[*] Milestone 3: Analyzing Multilingual Sentiment & Temporal Dynamics...")
        sentiment_posts = self.sentiment_analyzer.analyze_batch(cleaned_posts)
        sentiment_timeline = TemporalSentimentTracker.track_sentiment_over_time(sentiment_posts, interval="day")
        
        pos_count = sum(1 for p in sentiment_posts if p["sentiment"]["label"] == "positive")
        neg_count = sum(1 for p in sentiment_posts if p["sentiment"]["label"] == "negative")
        neu_count = sum(1 for p in sentiment_posts if p["sentiment"]["label"] == "neutral")
        print(f"    -> Sentiment Distribution: Positive={pos_count}, Neutral={neu_count}, Negative={neg_count}")
        print(f"    -> Tracked sentiment across {len(sentiment_timeline)} temporal windows.")

        # -------------------------------------------------------------------------
        # Milestone 4: BERTopic Topic Modeling & Dynamic Evolution
        # -------------------------------------------------------------------------
        print("\n[*] Milestone 4: Running BERTopic Modeling & Dynamic Topics Over Time...")
        topic_posts = self.topic_modeler.enrich_posts(sentiment_posts)
        
        docs = [p["cleaned_text"] for p in topic_posts]
        timestamps = [p["created_at"] for p in topic_posts]
        topics_over_time = self.topic_modeler.compute_topics_over_time(docs, timestamps, interval="week")
        
        print("    -> Discovered Topic Clusters:")
        for tid, name in self.topic_modeler.topic_names.items():
            top_words = ", ".join(self.topic_modeler.topic_words.get(tid, [])[:4])
            print(f"       Topic {tid}: {name} (Keywords: {top_words})")

        # -------------------------------------------------------------------------
        # Milestone 5: Graph Analytics & Influencer Mapping (NetworkX + Neo4j)
        # -------------------------------------------------------------------------
        print("\n[*] Milestone 5: Constructing Interaction Graph & Computing PageRank...")
        G = InteractionGraphBuilder.build_graph(topic_posts)
        user_metrics, communities = CentralityAnalyzer.analyze_graph(G)
        enriched_posts = CentralityAnalyzer.enrich_posts_with_graph(topic_posts, user_metrics)
        
        top_influencers = CentralityAnalyzer.get_top_influencers(user_metrics, top_k=5)
        print("    -> Top Influencers Ranked by PageRank:")
        for idx, inf in enumerate(top_influencers, 1):
            print(f"       {idx}. @{inf['username']} (PageRank: {inf['pagerank']}, In-Degree: {inf['in_degree']}, Comm: {inf['community_id']})")
        print(f"    -> Identified {len(communities)} distinct interaction communities.")

        # Sync to Neo4j / export Cypher
        cypher_path = str(self.data_dir / "neo4j_graph.cypher")
        self.graph_manager.sync_networkx_graph(G, user_metrics, export_cypher_path=cypher_path)

        # -------------------------------------------------------------------------
        # Milestone 6: Misinformation / Fake News Classifier (LIAR)
        # -------------------------------------------------------------------------
        print("\n[*] Milestone 6: Training & Evaluating Misinformation Classifier on LIAR Dataset...")
        eval_metrics = self.misinfo_classifier.train_and_evaluate()
        final_posts = self.misinfo_classifier.enrich_posts(enriched_posts)
        
        print(f"    -> LIAR Benchmark Evaluation:")
        print(f"       Accuracy:  {eval_metrics['accuracy'] * 100:.2f}%")
        print(f"       Precision: {eval_metrics['precision'] * 100:.2f}%")
        print(f"       Recall:    {eval_metrics['recall'] * 100:.2f}%")
        print(f"       F1-Score:  {eval_metrics['f1_score'] * 100:.2f}%")

        high_risk_count = sum(1 for p in final_posts if p["misinformation"]["risk_level"] in ["HIGH", "CRITICAL"])
        print(f"    -> Flagged {high_risk_count} posts with elevated misinformation risk.")

        # -------------------------------------------------------------------------
        # Milestone 7: Elasticsearch & Kibana Indexing
        # -------------------------------------------------------------------------
        print("\n[*] Milestone 7: Indexing into Elasticsearch & Generating Kibana Dashboards...")
        self.es_indexer.initialize_index()
        bulk_file = str(self.data_dir / "elasticsearch_bulk.jsonl")
        indexed_count = self.es_indexer.bulk_index_posts(final_posts, export_file_path=bulk_file)
        
        kibana_file = str(self.data_dir / "kibana_saved_objects.ndjson")
        KibanaDashboardExporter.export_saved_objects(output_path=kibana_file)
        print(f"    -> Generated Elasticsearch bulk file ({len(final_posts)} docs).")
        print(f"    -> Exported Kibana Saved Objects to: {kibana_file}")

        # Save complete enriched dataset
        processed_file = self.processed_dir / "posts_enriched.jsonl"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        with open(processed_file, "w", encoding="utf-8") as f:
            for p in final_posts:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"    -> Saved enriched master dataset to: {processed_file}")

        print("\n" + "="*70)
        print("  END-TO-END PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70 + "\n")

        return {
            "posts_processed": len(final_posts),
            "topics": len(self.topic_modeler.topic_names),
            "influencers": len(top_influencers),
            "misinfo_metrics": eval_metrics,
            "sentiment_distribution": {"pos": pos_count, "neu": neu_count, "neg": neg_count}
        }


def main():
    parser = argparse.ArgumentParser(description="Social Media Intelligence Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command")

    # run-all
    run_all_cmd = subparsers.add_parser("run-all", help="Execute complete 7-milestone pipeline")
    run_all_cmd.add_argument("--source", choices=["mock", "reddit", "youtube"], default="mock", help="Data source")
    run_all_cmd.add_argument("--limit", type=int, default=150, help="Number of records to ingest")

    # collect
    collect_cmd = subparsers.add_parser("collect", help="Ingest raw posts (Milestone 1)")
    collect_cmd.add_argument("--source", choices=["mock", "reddit", "youtube"], default="mock")
    collect_cmd.add_argument("--limit", type=int, default=100)

    # train-misinfo
    subparsers.add_parser("train-misinfo", help="Train and test LIAR misinformation classifier (Milestone 6)")

    # index-kibana
    subparsers.add_parser("index-kibana", help="Export Elasticsearch and Kibana configs (Milestone 7)")

    args = parser.parse_args()
    pipeline = SocialMediaPipeline()

    if args.command == "run-all" or args.command is None:
        src = getattr(args, "source", "mock")
        lim = getattr(args, "limit", 150)
        pipeline.run_all(source=src, limit=lim)
    elif args.command == "collect":
        if args.source == "reddit":
            c = RedditCollector()
        elif args.source == "youtube":
            c = YouTubeCollector()
        else:
            c = MockSocialDataGenerator()
        c.collect(limit=args.limit)
    elif args.command == "train-misinfo":
        clf = MisinformationClassifier()
        metrics = clf.train_and_evaluate()
        print(json.dumps(metrics, indent=2))
    elif args.command == "index-kibana":
        KibanaDashboardExporter.export_saved_objects()
        print("Exported Kibana saved objects.")


if __name__ == "__main__":
    main()
