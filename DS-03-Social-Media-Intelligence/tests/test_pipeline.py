"""Unit and Integration Tests for Social Media Intelligence Platform."""

import os
from pathlib import Path

from src.collectors import MockSocialDataGenerator, SocialPost
from src.preprocessing import PreprocessingPipeline, TextCleaner, EmojiHandler, LanguageDetector
from src.nlp import MultilingualSentimentAnalyzer, TopicModeler, MisinformationClassifier
from src.graph import InteractionGraphBuilder, CentralityAnalyzer
from src.indexing import KibanaDashboardExporter


def test_mock_collector():
    collector = MockSocialDataGenerator(output_dir="data/raw/test")
    posts = collector.collect(limit=30)
    assert len(posts) == 30
    assert any(p.target_author is not None for p in posts), "Should generate reply interaction edges"
    assert all(p.post_id and p.author for p in posts)


def test_text_cleaning_and_emojis():
    raw_text = "Breaking News! Check this link https://example.com/breaking 🚀🔥 &amp; tell @someone"
    cleaned = TextCleaner.clean(raw_text, preserve_emojis=True)
    assert "https://" not in cleaned
    assert "&amp;" not in cleaned
    assert ":rocket:" in cleaned or "🚀" in cleaned

    emojis = EmojiHandler.extract_emojis(raw_text)
    assert len(emojis) >= 2


def test_language_detection():
    en_text = "The quick brown fox jumps over the lazy dog"
    es_text = "El avance revolucionario en inteligencia artificial para diagnóstico médico"
    assert LanguageDetector.detect_language(en_text) == "en"
    assert LanguageDetector.detect_language(es_text) == "es"


def test_sentiment_analysis():
    analyzer = MultilingualSentimentAnalyzer()
    res_pos = analyzer.analyze_text("This breakthrough innovation is an extraordinary success! 🚀")
    assert res_pos["label"] == "positive"
    assert res_pos["polarity"] > 0

    res_neg = analyzer.analyze_text("Devastating ransomware cyber attack creates critical emergency 💔")
    assert res_neg["label"] == "negative"
    assert res_neg["polarity"] < 0


def test_topic_modeling():
    docs = [
        "Robotics and artificial intelligence foundation models with dexterity",
        "Open source neural networks and GPU transformer architectures",
        "Offshore wind energy turbine capacity expansion in Europe",
        "Solar photovoltaic power generation records broken this summer",
        "Ransomware encryption strikes municipal database servers",
        "Cybersecurity zero-day vulnerability patched in enterprise edge firewall"
    ]
    modeler = TopicModeler(nr_topics=3)
    topics, confs = modeler.fit_transform(docs)
    assert len(topics) == len(docs)
    assert len(modeler.topic_names) >= 2


def test_graph_and_pagerank():
    posts = [
        {
            "post_id": "p1", "author": "Alice", "target_author": None,
            "mentions": [], "metrics": {"score": 100}
        },
        {
            "post_id": "p2", "author": "Bob", "target_author": "Alice",
            "mentions": [], "metrics": {"score": 10}
        },
        {
            "post_id": "p3", "author": "Charlie", "target_author": "Alice",
            "mentions": ["Bob"], "metrics": {"score": 5}
        }
    ]
    G = InteractionGraphBuilder.build_graph(posts)
    assert G.number_of_nodes() >= 3
    assert G.has_edge("Bob", "Alice")

    metrics, communities = CentralityAnalyzer.analyze_graph(G)
    assert "Alice" in metrics
    # Alice has incoming replies from Bob and Charlie, so her PageRank should be high
    assert metrics["Alice"]["pagerank"] > metrics["Charlie"]["pagerank"]


def test_misinformation_classifier():
    classifier = MisinformationClassifier(model_dir="data/models/test")
    eval_metrics = classifier.train_and_evaluate()
    assert eval_metrics["accuracy"] >= 0.70
    assert eval_metrics["f1_score"] >= 0.70

    fake_res = classifier.predict("URGENT: Government secret directive bans all cash next Tuesday! 🚨")
    assert fake_res["risk_level"] in ["HIGH", "CRITICAL"]

    real_res = classifier.predict("Renewable energy generation from solar and wind reached record global highs.")
    assert real_res["risk_level"] in ["LOW", "MEDIUM"]


def test_kibana_export():
    out_file = "data/test_kibana.ndjson"
    KibanaDashboardExporter.export_saved_objects(output_path=out_file)
    assert Path(out_file).exists()
    assert Path(out_file).stat().st_size > 500
    Path(out_file).unlink()
