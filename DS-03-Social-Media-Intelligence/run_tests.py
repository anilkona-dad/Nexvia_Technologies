"""Test runner for testing all milestones without external test frameworks."""

import sys
from tests.test_pipeline import (
    test_mock_collector,
    test_text_cleaning_and_emojis,
    test_language_detection,
    test_sentiment_analysis,
    test_topic_modeling,
    test_graph_and_pagerank,
    test_misinformation_classifier,
    test_kibana_export
)

TESTS = [
    ("Milestone 1: Ingestion & Mock Collector", test_mock_collector),
    ("Milestone 2: Text Cleaning & Emojis", test_text_cleaning_and_emojis),
    ("Milestone 2: Multilingual Detection", test_language_detection),
    ("Milestone 3: Sentiment Analysis", test_sentiment_analysis),
    ("Milestone 4: Topic Modeling", test_topic_modeling),
    ("Milestone 5: Graph & PageRank", test_graph_and_pagerank),
    ("Milestone 6: Misinformation Classifier", test_misinformation_classifier),
    ("Milestone 7: Kibana Export", test_kibana_export)
]

def run():
    passed = 0
    failed = 0
    print("\n" + "="*60)
    print("  RUNNING PIPELINE UNIT & INTEGRATION TEST SUITE")
    print("="*60)

    for name, test_fn in TESTS:
        try:
            test_fn()
            print(f" [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f" [FAIL] {name}: {e}")
            failed += 1

    print("="*60)
    print(f"Test Results: {passed} Passed, {failed} Failed")
    print("="*60 + "\n")
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    run()
