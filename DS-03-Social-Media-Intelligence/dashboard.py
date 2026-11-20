"""Interactive Visual Web Dashboard for Social Media Intelligence Platform."""

import json
from pathlib import Path
import pandas as pd

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


def load_processed_data():
    """Load enriched JSONL dataset."""
    path = Path("data/processed/posts_enriched.jsonl")
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main_streamlit():
    st.set_page_config(
        page_title="Social Pulse | Intelligence Platform",
        page_icon="🌐",
        layout="wide"
    )

    st.title("🌐 Social Media Intelligence & Misinformation Platform")
    st.markdown(
        "**Multilingual Sentiment Tracking • Dynamic BERTopic Discovery • "
        "PageRank Influencer Mapping • Misinformation Detection (LIAR)**"
    )

    posts = load_processed_data()
    if not posts:
        st.warning("No processed data found. Please run `python main.py run-all` first to generate pipeline data.")
        return

    # Top KPI Metrics
    c1, c2, c3, c4 = st.columns(4)
    total_posts = len(posts)
    pos_count = sum(1 for p in posts if p.get("sentiment", {}).get("label") == "positive")
    high_risk_count = sum(1 for p in posts if p.get("misinformation", {}).get("risk_level") in ["HIGH", "CRITICAL"])
    unique_authors = len(set(p.get("author") for p in posts if p.get("author")))

    c1.metric("Total Ingested Posts", total_posts)
    c2.metric("Unique Contributors", unique_authors)
    c3.metric("Positive Sentiment", f"{(pos_count/total_posts)*100:.1f}%")
    c4.metric("Misinformation Alerts", high_risk_count, delta_color="inverse")

    st.divider()

    # Tabs for Milestones
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Sentiment & Trends",
        "🧠 BERTopic Dynamics",
        "🕸️ Influencer Graph & PageRank",
        "🛡️ Misinformation Detector",
        "🔍 Data Explorer"
    ])

    with tab1:
        st.subheader("Multilingual Sentiment Trajectory Over Time")
        rows = []
        for p in posts:
            dt = p.get("created_at", "")[:10]
            s = p.get("sentiment", {})
            rows.append({
                "date": dt,
                "polarity": s.get("polarity", 0.0),
                "label": s.get("label", "neutral"),
                "language": p.get("language", "unknown")
            })
        df_sent = pd.DataFrame(rows)
        if not df_sent.empty:
            avg_by_date = df_sent.groupby("date")["polarity"].mean().reset_index()
            st.line_chart(avg_by_date.set_index("date"))

            st.write("##### Sentiment by Detected Language")
            lang_dist = df_sent.groupby(["language", "label"]).size().unstack().fillna(0)
            st.bar_chart(lang_dist)

    with tab2:
        st.subheader("BERTopic Dynamic Topic Evolution")
        t_rows = []
        for p in posts:
            top = p.get("topic", {})
            t_rows.append({
                "topic_name": top.get("topic_name", "Unassigned"),
                "confidence": top.get("confidence", 0.5),
                "date": p.get("created_at", "")[:7]  # Month/Window
            })
        df_topic = pd.DataFrame(t_rows)
        if not df_topic.empty:
            st.write("##### Topic Frequency Breakdown")
            st.bar_chart(df_topic["topic_name"].value_counts())

    with tab3:
        st.subheader("Top Influencer PageRank Leaderboard")
        authors = {}
        for p in posts:
            auth = p.get("author")
            gm = p.get("graph_metrics", {})
            if auth and auth not in authors:
                authors[auth] = {
                    "Author": auth,
                    "PageRank": gm.get("author_pagerank", 0.0),
                    "In-Degree": gm.get("author_in_degree", 0),
                    "Community ID": gm.get("community_id", 0)
                }
        df_authors = pd.DataFrame(list(authors.values())).sort_values(by="PageRank", ascending=False)
        st.dataframe(df_authors.head(15), use_container_width=True)

    with tab4:
        st.subheader("Misinformation Detection & Risk Scoring")
        st.markdown("Test the classifier in real-time or inspect flagged content:")
        
        user_input = st.text_input(
            "Enter statement to verify:",
            "Secret government memo bans all cash and seizes private savings next Tuesday!"
        )
        if user_input:
            from src.nlp import MisinformationClassifier
            clf = MisinformationClassifier()
            res = clf.predict(user_input)
            
            col_a, col_b = st.columns(2)
            col_a.metric("Predicted Label", res["label"])
            col_b.metric("Fake Probability", f"{res['fake_probability'] * 100:.1f}%")
            
            if res["risk_level"] in ["HIGH", "CRITICAL"]:
                st.error(f"⚠️ Risk Level: {res['risk_level']}")
            else:
                st.success(f"✅ Risk Level: {res['risk_level']}")

        st.write("##### Flagged Posts from Corpus")
        flagged = [
            {
                "Title": p.get("title", ""),
                "Author": p.get("author", ""),
                "Risk": p.get("misinformation", {}).get("risk_level", "LOW"),
                "Fake Prob": p.get("misinformation", {}).get("fake_probability", 0.0)
            }
            for p in posts if p.get("misinformation", {}).get("risk_level") in ["HIGH", "CRITICAL"]
        ]
        st.dataframe(pd.DataFrame(flagged), use_container_width=True)

    with tab5:
        st.subheader("Master Corpus Explorer")
        st.dataframe(pd.DataFrame(posts)[["post_id", "platform", "author", "created_at", "title", "language"]], use_container_width=True)


if __name__ == "__main__":
    if HAS_STREAMLIT:
        main_streamlit()
    else:
        print("Streamlit not installed. Run `pip install streamlit` or use CLI `python main.py run-all`.")
