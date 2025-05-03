# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import datetime
import os
import sys
from typing import Optional, Dict, List

# Add project directory to path
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.append(project_dir)

# Import modules - adjust imports based on your project structure
from whatsapp_analyzer.data_loader import load_and_parse_chat
from whatsapp_analyzer.text_processor import preprocess_dataframe, get_user_statistics
from whatsapp_analyzer.analyzer import (
    analyze_sentiment,
    analyze_sentiment_batch,
    analyze_personality_texts,
    perform_topic_modeling,
    analyze_response_times,
    temporal_analysis,
    analyze_engagement,
    analyze_user_engagement
)
from whatsapp_analyzer.visualizer import (
    plot_sentiment,
    plot_temporal,
    plot_participants,
    plot_personality,
    plot_topics_bertopic,
    plot_user_engagement
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App title and description
st.title("WhatsApp Chat Analyzer")
st.write("Upload your exported WhatsApp chat to analyze conversation patterns and insights.")

# Sidebar
st.sidebar.header("Upload Chat")
uploaded_file = st.sidebar.file_uploader("Choose a WhatsApp chat export file (.txt)", type=["txt"])

# Analysis options
st.sidebar.header("Analysis Options")
run_sentiment = st.sidebar.checkbox("Sentiment Analysis", value=True)
run_personality = st.sidebar.checkbox("Personality Analysis", value=True)
run_topics = st.sidebar.checkbox("Topic Modeling", value=False, help="May be slow for large chats")
run_emoji = st.sidebar.checkbox("Emoji Analysis", value=True)

# Helper functions
@st.cache_data
def load_data(uploaded_file):
    """Load data from uploaded file with caching"""
    if uploaded_file is not None:
        try:
            # Load and parse chat data
            df = load_and_parse_chat(uploaded_file)
            if df is None or df.empty:
                st.error("Error: Could not parse chat data from the uploaded file")
                logger.error("Failed to parse chat data - returned None or empty DataFrame")
                return None
                
            return df
        except Exception as e:
            st.error(f"Error: {str(e)}")
            logger.exception("Exception while loading data")
            return None
    return None

@st.cache_data
def process_data(df):
    """Process data with caching"""
    if df is None:
        logger.error("Cannot process None DataFrame in process_data")
        return None
        
    try:
        # Preprocess the dataframe
        processed_df = preprocess_dataframe(df)
        if processed_df is None:
            st.error("Error: Failed to preprocess data")
            logger.error("preprocess_dataframe returned None")
            return None
            
        return processed_df
    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        logger.exception("Exception in process_data")
        return None

@st.cache_data
def run_analysis(df, run_sentiment=True, run_personality=True):
    """Run analysis with caching"""
    if df is None:
        logger.error("Cannot analyze None DataFrame in run_analysis")
        return {}
        
    results = {}
    
    try:
        # Extract text for analysis
        texts = df['Text'].fillna('').astype(str).tolist()
        
        # Run sentiment analysis
        if run_sentiment and texts:
            with st.spinner("Running sentiment analysis..."):
                sentiment_results = analyze_sentiment_batch(texts)
                
                # Add results to dataframe
                for col in sentiment_results.columns:
                    df[col] = sentiment_results[col].values
                    
                results['sentiment'] = sentiment_results
                
        # Run personality analysis
        if run_personality and texts:
            with st.spinner("Running personality analysis..."):
                traits = ['joy', 'anger', 'sadness', 'family', 'friend', 'social', 
                        'positive_emotion', 'negative_emotion']
                personality_results = analyze_personality_texts(texts, traits)
                
                # Add results to dataframe
                for col in personality_results.columns:
                    df[col] = personality_results[col].values
                    
                results['personality'] = personality_results
                
        return results
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        logger.exception("Exception in run_analysis")
        return {}

def main():
    # Load the chat data
    df = load_data(uploaded_file)
    
    # If data is loaded successfully
    if df is not None:
        st.success(f"Successfully loaded chat with {len(df)} messages")
        
        # Show data info
        with st.expander("Data Preview"):
            st.dataframe(df.head(5))
            
            # Show basic info
            st.write(f"**Date Range**: {df['Datetime'].min().date()} to {df['Datetime'].max().date()}")
            st.write(f"**Participants**: {', '.join(df['Name'].unique())}")
            st.write(f"**Total Messages**: {len(df)}")
        
        # Process the data
        processed_df = process_data(df)
        
        if processed_df is not None:
            # Run analysis if data is processed successfully
            analysis_results = run_analysis(
                processed_df, 
                run_sentiment=run_sentiment,
                run_personality=run_personality
            )
            
            # Start visualization
            st.header("Chat Analysis")
            
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4 = st.tabs([
                "Participant Analysis", 
                "Temporal Analysis", 
                "Sentiment Analysis", 
                "Content Analysis"
            ])
            
            with tab1:
                st.subheader("Participant Activity")
                
                # Plot message counts
                fig = plot_participants(processed_df)
                if fig:
                    st.pyplot(fig)
                
                # Calculate user statistics
                user_stats = get_user_statistics(processed_df)
                
                # Show participant stats
                if user_stats:
                    st.subheader("Participant Statistics")
                    for name, stats in user_stats.items():
                        st.write(f"**{name}**")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Messages", stats['message_count'])
                        col2.metric("Words", stats['word_count'])
                        col3.metric("Avg Words/Message", f"{stats['avg_words_per_message']:.1f}")
                        
                        # Show media count if available
                        if 'media_count' in stats:
                            st.metric("Media Messages", stats['media_count'])
                        
                        # Show emoji stats if available
                        if run_emoji and 'favorite_emojis' in stats and stats['favorite_emojis']:
                            st.write("**Top Emojis:**", ''.join(stats['favorite_emojis'].keys()))
            
            with tab2:
                st.subheader("Temporal Patterns")
                
                # Add temporal columns if needed
                if 'hour' not in processed_df.columns and 'Hour' in processed_df.columns:
                    processed_df['hour'] = processed_df['Hour']
                    
                if 'day_of_week' not in processed_df.columns and 'DayOfWeek' in processed_df.columns:
                    processed_df['day_of_week'] = processed_df['DayOfWeek']
                
                # Plot temporal patterns
                fig = plot_temporal(processed_df)
                if fig:
                    st.pyplot(fig)
                else:
                    # Create custom temporal plots if visualizer couldn't create them
                    st.write("Creating custom temporal plots...")
                    
                    # Messages by hour
                    if 'Hour' in processed_df.columns:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        sns.countplot(x='Hour', data=processed_df, ax=ax)
                        ax.set_title("Messages by Hour of Day")
                        ax.set_xlabel("Hour")
                        ax.set_ylabel("Message Count")
                        st.pyplot(fig)
                    
                    # Messages by day of week
                    if 'DayOfWeek' in processed_df.columns:
                        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                        fig, ax = plt.subplots(figsize=(10, 5))
                        sns.countplot(
                            x='DayOfWeek', 
                            data=processed_df, 
                            order=days_order,
                            ax=ax
                        )
                        ax.set_title("Messages by Day of Week")
                        ax.set_xlabel("Day")
                        ax.set_ylabel("Message Count")
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
            
            with tab3:
                st.subheader("Sentiment Analysis")
                
                if run_sentiment and 'sentiment' in processed_df.columns:
                    # Plot sentiment distribution
                    fig = plot_sentiment(processed_df)
                    if fig:
                        st.pyplot(fig)
                    
                    # Show sentiment stats
                    st.write("**Sentiment Distribution:**")
                    sentiment_counts = processed_df['sentiment'].value_counts()
                    
                    # Create metrics in columns
                    cols = st.columns(3)
                    for i, (sent, count) in enumerate(sentiment_counts.items()):
                        pct = 100 * count / len(processed_df)
                        cols[i].metric(
                            sent.capitalize(), 
                            f"{count} ({pct:.1f}%)"
                        )
                    
                    # Sentiment by participant
                    st.subheader("Sentiment by Participant")
                    
                    # Create a pivot table of sentiment by participant
                    pivot = pd.crosstab(
                        processed_df['Name'], 
                        processed_df['sentiment'],
                        normalize='index'
                    ) * 100
                    
                    # Plot the pivot table
                    fig, ax = plt.subplots(figsize=(10, 5))
                    pivot.plot(
                        kind='bar', 
                        stacked=True, 
                        ax=ax,
                        color=['#4caf50', '#ffeb3b', '#f44336']  # green, yellow, red
                    )
                    ax.set_title("Sentiment Distribution by Participant (%)")
                    ax.set_ylabel("Percentage")
                    ax.set_xlabel("Participant")
                    ax.legend(title="Sentiment")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                else:
                    st.info("Sentiment analysis not selected or failed to run.")
            
            with tab4:
                st.subheader("Content Analysis")
                
                # Word count analysis
                if 'WordCount' in processed_df.columns:
                    st.write("**Word Count Distribution**")
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.histplot(
                        data=processed_df, 
                        x='WordCount',
                        bins=20,
                        kde=True,
                        ax=ax
                    )
                    ax.set_title("Word Count Distribution")
                    ax.set_xlabel("Words per Message")
                    ax.set_ylabel("Frequency")
                    st.pyplot(fig)
                
                # Personality traits
                if run_personality and 'personality' in analysis_results:
                    st.subheader("Personality Traits")
                    
                    # Plot personality traits
                    fig = plot_personality(analysis_results['personality'])
                    if fig:
                        st.pyplot(fig)
                    
                    # Show trait comparison by participant
                    st.write("**Trait Comparison by Participant**")
                    
                    personality_df = analysis_results['personality']
                    traits = personality_df.columns.tolist()
                    
                    # Add participant column
                    personality_df['Name'] = processed_df['Name'].values
                    
                    # Calculate average trait values by participant
                    trait_by_participant = personality_df.groupby('Name')[traits].mean()
                    
                    # Plot traits by participant
                    fig, ax = plt.subplots(figsize=(12, 6))
                    trait_by_participant.T.plot(kind='bar', ax=ax)
                    ax.set_title("Personality Traits by Participant")
                    ax.set_ylabel("Score")
                    ax.set_xlabel("Trait")
                    plt.xticks(rotation=45)
                    ax.legend(title="Participant")
                    st.pyplot(fig)
        else:
            st.error("Failed to process the data. Please check the log for details.")
    else:
        # Show instructions when no file is uploaded
        st.info("Please upload a WhatsApp chat export file (.txt) to begin analysis.")
        
        st.markdown("""
        ### How to export your WhatsApp chat:
        
        1. Open the WhatsApp chat you want to analyze
        2. Tap the three dots (⋮) in the top right corner
        3. Select "More" > "Export chat"
        4. Choose "Without media"
        5. Save the .txt file and upload it here
        
        ### What you can analyze:
        
        - **Participant Activity**: Who sends the most messages and when
        - **Temporal Patterns**: When the chat is most active
        - **Sentiment Analysis**: The emotional tone of messages
        - **Content Analysis**: Word usage, emoji patterns, and topics
        """)

if __name__ == "__main__":
    main()