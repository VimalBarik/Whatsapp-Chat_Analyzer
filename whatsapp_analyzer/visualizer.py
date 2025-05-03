# visualizer.py

import os
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
from datetime import datetime

# Color configuration
COLOR_PALETTE = {
    "positive": "#4caf50",
    "neutral": "#ffeb3b",
    "negative": "#f44336",
    "highlight": "#2196f3",
    "person1": "#8e44ad",
    "person2": "#e67e22"
}

def plot_sentiment(sentiment_df: pd.DataFrame) -> Optional[plt.Figure]:
    if sentiment_df.empty or 'sentiment' not in sentiment_df.columns:
        st.warning("No sentiment data to visualize")
        return None

    sentiment_df['sentiment'] = pd.Categorical(
        sentiment_df['sentiment'],
        categories=['positive', 'neutral', 'negative'],
        ordered=True
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    plot = sns.countplot(
        x='sentiment',
        data=sentiment_df,
        order=['positive', 'neutral', 'negative'],
        palette=[COLOR_PALETTE['positive'], COLOR_PALETTE['neutral'], COLOR_PALETTE['negative']],
        ax=ax
    )
    
    total = len(sentiment_df)
    for p in plot.patches:
        height = p.get_height()
        ax.text(
            p.get_x() + p.get_width()/2.,
            height + 0.5,
            f'{100 * height/total:.1f}%',
            ha='center'
        )
    
    ax.set_title("Sentiment Distribution (Total: {})".format(total))
    return fig

def plot_temporal(df: pd.DataFrame) -> Optional[plt.Figure]:
    if not {'hour', 'day_of_week'}.issubset(df.columns):
        return None

    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    sns.countplot(x='hour', data=df, ax=axs[0])
    axs[0].set_title("Messages by Hour")

    sns.countplot(
        x='day_of_week',
        data=df,
        order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        ax=axs[1]
    )
    axs[1].set_title("Messages by Day of Week")

    plt.tight_layout()
    return fig

def plot_participants(df: pd.DataFrame) -> Optional[plt.Figure]:
    if df.empty or 'Name' not in df.columns:
        return None

    if 'sentiment' in df.columns:
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        
        sns.countplot(
            y='Name',
            data=df,
            order=df['Name'].value_counts().index,
            palette=[COLOR_PALETTE['person1'], COLOR_PALETTE['person2']],
            ax=axs[0]
        )
        axs[0].set_title("Message Count by Participant")

        sns.countplot(
            x='Name',
            hue='sentiment',
            data=df,
            hue_order=['positive', 'neutral', 'negative'],
            palette=[COLOR_PALETTE['positive'], COLOR_PALETTE['neutral'], COLOR_PALETTE['negative']],
            ax=axs[1]
        )
        axs[1].set_title("Sentiment by Participant")
        axs[1].legend(title='Sentiment')
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(
            y='Name',
            data=df,
            order=df['Name'].value_counts().index,
            palette=[COLOR_PALETTE['person1'], COLOR_PALETTE['person2']],
            ax=ax
        )
        ax.set_title("Message Count by Participant")

    plt.tight_layout()
    return fig

def plot_personality(personality_df: pd.DataFrame) -> Optional[plt.Figure]:
    if personality_df.empty:
        return None

    numeric_df = personality_df.select_dtypes(include='number')
    if numeric_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    numeric_df.mean().sort_values().plot(
        kind='barh',
        color=COLOR_PALETTE['highlight'],
        ax=ax
    )
    ax.set_title("Average Personality Traits")
    return fig

def plot_topics_bertopic(bertopic_model, topics: List[int], df: pd.DataFrame) -> List[plt.Figure]:
    if bertopic_model is None or not topics:
        return []

    figures = []
    
    topic_counts = pd.Series(topics).value_counts().sort_index()
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    topic_counts.plot(kind='bar', ax=ax1, color=COLOR_PALETTE['highlight'])
    ax1.set_title("Topic Frequency")
    figures.append(fig1)

    if 'Name' in df.columns:
        topic_df = df.copy()
        topic_df['topic'] = topics
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        sns.countplot(
            x='topic',
            hue='Name',
            data=topic_df,
            palette=[COLOR_PALETTE['person1'], COLOR_PALETTE['person2']],
            ax=ax2
        )
        ax2.set_title("Topic Distribution by Participant")
        figures.append(fig2)

    for topic in topic_counts.index[:5]:
        words_scores = dict(bertopic_model.get_topic(topic))
        wc = WordCloud(width=800, height=400, background_color='white')
        wc.generate_from_frequencies(words_scores)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        ax.set_title(f"Topic {topic}", fontsize=16)
        figures.append(fig)

    return figures

def plot_user_engagement(results: Dict) -> None:
    if not results:
        print("No engagement data to visualize")
        return
    
    if 'media_analysis' in results and 'data' in results['media_analysis']:
        media_data = results['media_analysis']['data']
        if not media_data.empty:
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(
                x='Name', 
                y='Media_Count', 
                data=media_data,
                palette=[COLOR_PALETTE['person1'], COLOR_PALETTE['person2']]
            )
            plt.title("Media Shared per User")
            
            for p in ax.patches:
                ax.annotate(
                    f"{int(p.get_height())}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', 
                    va='center', 
                    xytext=(0, 5), 
                    textcoords='offset points'
                )
            
            if 'percentage' in results['media_analysis']:
                percentages = results['media_analysis']['percentage']
                for i, name in enumerate(media_data['Name']):
                    pct = percentages.get(name, 0)
                    ax.text(
                        i, 
                        media_data.loc[i, 'Media_Count'] + 0.5, 
                        f"{pct}%", 
                        ha='center'
                    )
            
            plt.tight_layout()
            plt.show()
    
    if 'emoji_analysis' in results and not results['emoji_analysis'].empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        sns.barplot(
            x='Name', 
            y='Total_Emojis', 
            data=results['emoji_analysis'],
            palette=[COLOR_PALETTE['person1'], COLOR_PALETTE['person2']],
            ax=ax1
        )
        ax1.set_title("Total Emojis Used")
        
        sns.barplot(
            x='Name', 
            y='Avg_Emojis_Per_Message', 
            data=results['emoji_analysis'],
            palette=[COLOR_PALETTE['person1'], COLOR_PALETTE['person2']],
            ax=ax2
        )
        ax2.set_title("Average Emojis per Message")
        
        plt.tight_layout()
        plt.show()
    
    if 'word_analysis' in results and not results['word_analysis'].empty:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        metrics = ['Total_Words', 'Avg_Words_Per_Message', 'Median_Words_Per_Message']
        titles = ["Total Words", "Average Words per Message", "Median Words per Message"]
        
        for ax, metric, title in zip(axes, metrics, titles):
            sns.barplot(
                x='Name', 
                y=metric, 
                data=results['word_analysis'],
                palette=[COLOR_PALETTE['person1'], COLOR_PALETTE['person2']],
                ax=ax
            )
            ax.set_title(title)
            ax.set_ylabel("Count" if metric == 'Total_Words' else "Words")
        
        plt.tight_layout()
        plt.show()