# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import sys
from typing import Optional, Dict, List
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Add project directory to path
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.append(project_dir)

from whatsapp_analyzer.data_loader import load_and_parse_chat
from whatsapp_analyzer.text_processor import preprocess_dataframe, get_user_statistics
from whatsapp_analyzer.analyzer import (
    analyze_sentiment,
    analyze_sentiment_batch,
    analyze_personality_texts,
    configure_hf_token,
    has_hf_token,
    perform_topic_modeling,
    analyze_response_times,
    temporal_analysis,
    analyze_engagement,
    analyze_user_engagement,
)
from whatsapp_analyzer.visualizer import (
    plot_sentiment,
    plot_temporal,
    plot_participants,
    plot_personality,
    plot_topics_bertopic,
    plot_user_engagement,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def configure_hf_token_from_secrets():
    """Expose Streamlit's HF token secret to the analyzer module."""
    try:
        hf_token = st.secrets.get("HF_TOKEN")
    except Exception:
        hf_token = None

    if hf_token:
        configure_hf_token(hf_token)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

configure_hf_token_from_secrets()

st.title("WhatsApp Chat Analyzer 💬")
st.write("Upload your exported WhatsApp chat to analyse conversation patterns and insights.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Upload Chat")
uploaded_file = st.sidebar.file_uploader(
    "Choose a WhatsApp chat export file (.txt)", type=["txt"]
)

st.sidebar.header("Language")
chat_language = st.sidebar.selectbox(
    "Chat language",
    options=["Auto-detect", "English", "Italian", "Other"],
    index=0,
    help=(
        "Choose 'Auto-detect' to let the app figure it out. "
        "Selecting the correct language improves sentiment accuracy."
    ),
)
USE_MULTILINGUAL = chat_language != "English"

st.sidebar.header("Analysis Options")
run_sentiment   = st.sidebar.checkbox("Sentiment Analysis",   value=True)
run_personality = st.sidebar.checkbox("Personality Analysis", value=True)
run_topics      = st.sidebar.checkbox(
    "Topic Modeling", value=True,
    help="May be slow for large chats."
)
run_emoji       = st.sidebar.checkbox("Emoji Analysis",       value=True)
run_engagement  = st.sidebar.checkbox("Engagement Analysis",  value=True)

hf_token_available = has_hf_token()
sentiment_backend = st.sidebar.selectbox(
    "Sentiment backend",
    options=["Hugging Face API", "Local fallback"],
    index=0 if hf_token_available else 1,
    help="Hugging Face uses your Streamlit HF_TOKEN secret. Local fallback uses VADER/TextBlob.",
)
sentiment_backend_key = "hf" if sentiment_backend == "Hugging Face API" else "local"
if sentiment_backend_key == "hf":
    if hf_token_available:
        st.sidebar.success("HF_TOKEN loaded from Streamlit secrets.")
    else:
        st.sidebar.error("HF_TOKEN was not found in Streamlit secrets.")

# ── Cached helpers ────────────────────────────────────────────────────────────

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        df = load_and_parse_chat(uploaded_file)
        if df is None or df.empty:
            st.error("Could not parse chat data from the uploaded file. "
                     "Make sure it is an unmodified WhatsApp export (.txt).")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        logger.exception("Exception while loading data")
        return None


@st.cache_data
def process_data(df):
    if df is None:
        return None
    try:
        processed = preprocess_dataframe(df)
        if processed is None:
            st.error("Failed to preprocess data.")
        return processed
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        logger.exception("Exception in process_data")
        return None


@st.cache_data
def run_analysis(
    df, do_sentiment, do_personality, do_topics, multilingual,
    sentiment_backend, hf_token_available
):
    if df is None:
        return {}

    results = {}
    texts = df['Text'].fillna('').astype(str).tolist()

    if do_sentiment and texts:
        with st.spinner("Running sentiment analysis…"):
            try:
                sentiment_results = analyze_sentiment_batch(
                    texts,
                    use_multilingual=multilingual,
                    backend=sentiment_backend,
                )
                for col in sentiment_results.columns:
                    df[col] = sentiment_results[col].values
                results['sentiment'] = sentiment_results
            except Exception as e:
                logger.exception("Sentiment analysis failed")
                if sentiment_backend == "hf":
                    st.error(
                        "Hugging Face sentiment analysis failed. "
                        "Check that `HF_TOKEN` is present in Streamlit secrets and has access "
                        f"to the inference API. Details: {e}"
                    )
                else:
                    st.warning(f"Sentiment analysis failed: {e}")

    if do_personality and texts:
        with st.spinner("Running personality analysis…"):
            try:
                traits = [
                    'joy', 'anger', 'sadness', 'family',
                    'friend', 'social', 'positive_emotion', 'negative_emotion',
                ]
                personality_results = analyze_personality_texts(texts, traits)
                for col in personality_results.columns:
                    df[col] = personality_results[col].values
                results['personality'] = personality_results
            except Exception as e:
                logger.exception("Personality analysis failed")
                st.warning(f"Personality analysis failed: {e}")

    if do_topics and texts:
        with st.spinner("Running topic modeling (this may take a while)…"):
            try:
                content_df = df[df['Text'].str.len() > 10].copy()
                if not content_df.empty:
                    topic_model, topics, topic_info = perform_topic_modeling(
                        content_df['Text'].tolist()
                    )
                    if topic_model is not None and topics is not None:
                        content_df['topic'] = topics
                        results['topics'] = {
                            'model': topic_model,
                            'topics': topics,
                            'info': topic_info,
                            'df': content_df,
                        }
                        logger.info(f"Generated {len(set(topics))} topics")
                    else:
                        st.warning("Topic modeling returned no results.")
                else:
                    st.warning("Not enough text content for topic modeling.")
            except Exception as e:
                logger.exception("Topic modeling failed")
                st.warning(f"Topic modeling failed: {e}")

    return results


# ── Topic display helper ──────────────────────────────────────────────────────

def display_topic_modeling_results(results):
    if 'topics' not in results:
        st.warning("No topic modeling results available.")
        return

    topic_model = results['topics']['model']
    topics      = results['topics']['topics']
    topic_info  = results['topics']['info']
    topic_df    = results['topics']['df']

    st.subheader("Topic Overview")
    try:
        st.dataframe(topic_info.drop('Representative_Docs', axis=1, errors='ignore'))

        st.subheader("Topic Visualizations")
        topic_figures = plot_topics_bertopic(topic_model, topics, topic_df)

        if not topic_figures:
            st.warning("Could not generate topic visualizations.")
            return

        st.pyplot(topic_figures[0])
        if len(topic_figures) > 1:
            st.pyplot(topic_figures[1])

        st.subheader("Top Topics – Word Clouds")
        num_wc   = min(len(topic_figures) - 2, 6)
        num_cols = min(2, num_wc) if num_wc > 0 else 1
        cols     = st.columns(num_cols)
        for i, fig in enumerate(topic_figures[2: 2 + num_wc]):
            cols[i % num_cols].pyplot(fig)

        st.subheader("Explore a Specific Topic")
        topic_unique = sorted(set(topics))
        if topic_unique:
            selected = st.selectbox(
                "Select topic",
                options=topic_unique,
                format_func=lambda x: (
                    f"Topic {x}: "
                    + (topic_info[topic_info['Topic'] == x]['Name'].values[0]
                       if x in topic_info['Topic'].values else 'Unknown')
                ),
            )
            docs = topic_df[topic_df['topic'] == selected]['Text'].head(10)
            st.write("**Representative messages:**")
            for i, doc in enumerate(docs, 1):
                st.write(f"{i}. {doc}")
    except Exception as e:
        st.error(f"Error displaying topic results: {e}")
        logger.exception("Exception in display_topic_modeling_results")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df = load_data(uploaded_file)

    if df is None:
        st.info("Please upload a WhatsApp chat export file (.txt) to begin analysis.")
        st.markdown("""
        ### How to export your WhatsApp chat

        **English / iOS**
        1. Open the chat → tap the contact/group name at the top
        2. Scroll down → **Export Chat** → **Without Media**
        3. Upload the `.txt` file here

        **Italian / Android**
        1. Apri la chat → tocca ⋮ in alto a destra
        2. **Altro** → **Esporta chat** → **Senza media**
        3. Carica il file `.txt` qui

        ### What you can analyse
        - **Participant Activity** – who messages most and when
        - **Temporal Patterns** – hourly / daily activity heatmaps
        - **Sentiment Analysis** – emotional tone (works in Italian too!)
        - **Content Analysis** – word usage and personality traits
        - **Topic Modeling** – main themes discussed
        - **Engagement Analysis** – media, emoji, and word stats
        """)
        return

    st.success(f"✅ Loaded **{len(df):,}** messages from **{df['Name'].nunique()}** participants")

    with st.expander("📋 Data Preview"):
        st.dataframe(df.head(10))
        col1, col2, col3 = st.columns(3)
        col1.metric("Date Range",
                    f"{df['Datetime'].min().date()} → {df['Datetime'].max().date()}")
        col2.metric("Total Messages", f"{len(df):,}")
        col3.metric("Participants", ", ".join(df['Name'].unique()))

    processed_df = process_data(df)
    if processed_df is None:
        st.error("Failed to process the data. Please check the log.")
        return

    analysis_results = run_analysis(
        processed_df,
        do_sentiment=run_sentiment,
        do_personality=run_personality,
        do_topics=run_topics,
        multilingual=USE_MULTILINGUAL,
        sentiment_backend=sentiment_backend_key,
        hf_token_available=hf_token_available,
    )

    st.header("📊 Chat Analysis")

    tabs = st.tabs([
        "👥 Participants",
        "🕐 Temporal",
        "💬 Sentiment",
        "📝 Content",
        "🔍 Topics",
        "📈 Engagement",
    ])

    # ── Tab 1: Participants ───────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Participant Activity")
        fig = plot_participants(processed_df)
        if fig:
            st.pyplot(fig)

        user_stats = get_user_statistics(processed_df)
        if user_stats:
            st.subheader("Per-participant Statistics")
            for name, stats in user_stats.items():
                st.markdown(f"**{name}**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Messages",         stats['message_count'])
                c2.metric("Total Words",       stats.get('word_count', 0))
                c3.metric("Avg Words/Message", f"{stats.get('avg_words_per_message', 0):.1f}")
                if run_emoji and stats.get('favorite_emojis'):
                    st.write("Top emojis:", ''.join(stats['favorite_emojis'].keys()))

    # ── Tab 2: Temporal ───────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Temporal Patterns")

        # Ensure lowercase column aliases expected by plot_temporal
        if 'hour' not in processed_df.columns and 'Hour' in processed_df.columns:
            processed_df['hour'] = processed_df['Hour']
        if 'day_of_week' not in processed_df.columns and 'DayOfWeek' in processed_df.columns:
            processed_df['day_of_week'] = processed_df['DayOfWeek']

        fig = plot_temporal(processed_df)
        if fig:
            st.pyplot(fig)
        else:
            st.info("Generating fallback temporal plots…")
            if 'Hour' in processed_df.columns:
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.countplot(x='Hour', data=processed_df, ax=ax)
                ax.set_title("Messages by Hour of Day")
                st.pyplot(fig)
            if 'DayOfWeek' in processed_df.columns:
                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.countplot(x='DayOfWeek', data=processed_df, order=days, ax=ax)
                ax.set_title("Messages by Day of Week")
                plt.xticks(rotation=45)
                st.pyplot(fig)

    # ── Tab 3: Sentiment ──────────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Sentiment Analysis")
        if run_sentiment and 'sentiment' in processed_df.columns:
            if 'sentiment_backend' in processed_df.columns:
                backend_counts = processed_df['sentiment_backend'].value_counts()
                st.caption(
                    "Backend used: "
                    + ", ".join(
                        f"{backend} ({count:,})"
                        for backend, count in backend_counts.items()
                    )
                )

            fig = plot_sentiment(processed_df)
            if fig:
                st.pyplot(fig)

            st.write("**Sentiment Distribution**")
            sentiment_counts = processed_df['sentiment'].value_counts()
            cols = st.columns(len(sentiment_counts))
            for i, (sent, count) in enumerate(sentiment_counts.items()):
                pct = 100 * count / len(processed_df)
                cols[i].metric(sent.capitalize(), f"{count} ({pct:.1f}%)")

            st.subheader("Sentiment by Participant")
            pivot = (
                pd.crosstab(processed_df['Name'], processed_df['sentiment'], normalize='index')
                * 100
            )
            fig, ax = plt.subplots(figsize=(10, 5))
            pivot.plot(kind='bar', stacked=True, ax=ax,
                       color=['#4caf50', '#ffeb3b', '#f44336'])
            ax.set_title("Sentiment Distribution by Participant (%)")
            ax.set_ylabel("Percentage")
            ax.set_xlabel("Participant")
            ax.legend(title="Sentiment")
            plt.xticks(rotation=45)
            st.pyplot(fig)

            if chat_language != "English":
                st.info(
                    "Sentiment was analysed using the selected multilingual pipeline."
                )
        else:
            st.info("Enable **Sentiment Analysis** in the sidebar to see results.")

    # ── Tab 4: Content ────────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Content Analysis")

        if 'WordCount' in processed_df.columns:
            st.write("**Word Count Distribution**")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.histplot(data=processed_df, x='WordCount', bins=20, kde=True, ax=ax)
            ax.set_title("Words per Message")
            ax.set_xlabel("Word Count")
            st.pyplot(fig)

        if run_personality and 'personality' in analysis_results:
            st.subheader("Personality Traits")
            fig = plot_personality(analysis_results['personality'])
            if fig:
                st.pyplot(fig)

            st.write("**Traits by Participant**")
            p_df = analysis_results['personality'].copy()
            traits_cols = p_df.columns.tolist()
            p_df['Name'] = processed_df['Name'].values
            by_participant = p_df.groupby('Name')[traits_cols].mean()

            fig, ax = plt.subplots(figsize=(12, 5))
            by_participant.T.plot(kind='bar', ax=ax)
            ax.set_title("Personality Traits by Participant")
            ax.set_ylabel("Score")
            ax.set_xlabel("Trait")
            plt.xticks(rotation=45)
            ax.legend(title="Participant")
            st.pyplot(fig)

    # ── Tab 5: Topics ─────────────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("Topic Modeling")
        if run_topics:
            if 'topics' in analysis_results:
                display_topic_modeling_results(analysis_results)
            else:
                st.info(
                    "Topic modeling was enabled but no results were generated. "
                    "This can happen if there are too few messages or an error occurred."
                )
        else:
            st.info("Enable **Topic Modeling** in the sidebar to see results.")

    # ── Tab 6: Engagement ─────────────────────────────────────────────────────
    with tabs[5]:
        st.subheader("Engagement Analysis")
        if run_engagement:
            with st.spinner("Calculating engagement metrics…"):
                try:
                    engagement = analyze_user_engagement(processed_df)
                    if engagement:
                        # Use plot_user_engagement which now returns figures
                        engagement_figs = plot_user_engagement(engagement)
                        for fig in engagement_figs:
                            st.pyplot(fig)

                        # Also show raw data tables
                        if 'media_analysis' in engagement and 'data' in engagement['media_analysis']:
                            with st.expander("Media data"):
                                st.dataframe(engagement['media_analysis']['data'])
                        if 'emoji_analysis' in engagement:
                            with st.expander("Emoji data"):
                                st.dataframe(engagement['emoji_analysis'])
                        if 'word_analysis' in engagement:
                            with st.expander("Word data"):
                                st.dataframe(engagement['word_analysis'])
                    else:
                        st.info("No engagement data available.")
                except Exception as e:
                    st.error(f"Engagement analysis failed: {e}")
                    logger.exception("Engagement analysis error")
        else:
            st.info("Enable **Engagement Analysis** in the sidebar to see results.")


# Streamlit runs at module scope — call main() directly (not under __name__ guard)
main()
