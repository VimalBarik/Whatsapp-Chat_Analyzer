# analyzer.py

import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from textblob import TextBlob
from empath import Empath
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime
import emoji

logger = logging.getLogger(__name__)

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text: str) -> Dict[str, Union[float, str]]:
    if not isinstance(text, str) or not text.strip():
        return {'polarity': 0.0, 'subjectivity': 0.0, 'compound': 0.0, 'sentiment': 'neutral'}

    clean_text = re.sub(r'[^\w\s]', '', text.lower())
    
    blob = TextBlob(clean_text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    vader_scores = analyzer.polarity_scores(clean_text)
    compound = vader_scores['compound']

    if compound >= 0.25 or polarity >= 0.25:
        sentiment = 'positive'
    elif compound <= -0.25 or polarity <= -0.25:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'

    return {
        'polarity': polarity,
        'subjectivity': subjectivity,
        'compound': compound,
        'sentiment': sentiment
    }

def analyze_sentiment_batch(texts: List[str]) -> pd.DataFrame:
    results = [analyze_sentiment(text) for text in tqdm(texts, desc="Analyzing sentiment")]
    return pd.DataFrame(results)

default_personality_traits = [
    'joy', 'anger', 'sadness', 'family', 
    'friend', 'social', 'positive_emotion', 
    'negative_emotion'
]

def analyze_personality_texts(texts: List[str], traits: List[str] = None) -> pd.DataFrame:
    if traits is None:
        traits = default_personality_traits

    lexicon = Empath()
    results = []
    
    for text in tqdm(texts, desc="Analyzing personality"):
        if not isinstance(text, str) or not text.strip():
            results.append({trait: 0 for trait in traits})
            continue
            
        analysis = lexicon.analyze(text, categories=traits, normalize=True)
        results.append({trait: analysis.get(trait, 0) for trait in traits})
        
    return pd.DataFrame(results)

def perform_topic_modeling(texts: List[str]) -> Tuple:
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer

    if not texts or len(texts) < 10:
        logger.warning("Insufficient texts for topic modeling")
        return None, None, None

    topic_model = BERTopic(verbose=False)
    topics, probs = topic_model.fit_transform(texts)

    topic_info = topic_model.get_topic_info()
    return topic_model, topics, topic_info

def analyze_response_times(df: pd.DataFrame) -> pd.DataFrame:
    if 'Datetime' not in df.columns or 'Name' not in df.columns:
        raise ValueError("DataFrame must contain 'Datetime' and 'Name' columns")

    df = df.sort_values('Datetime')
    results = []
    
    for i in range(1, len(df)):
        current = df.iloc[i]
        previous = df.iloc[i-1]
        
        if current['Name'] != previous['Name']:
            response_time = (current['Datetime'] - previous['Datetime']).total_seconds() / 60
            results.append({
                'responder': current['Name'],
                'response_time_min': response_time,
                'previous_sender': previous['Name']
            })
    
    return pd.DataFrame(results)

def temporal_analysis(df: pd.DataFrame) -> Dict:
    if 'Datetime' not in df.columns:
        raise ValueError("DataFrame must contain 'Datetime' column")

    df['hour'] = df['Datetime'].dt.hour
    df['day_of_week'] = df['Datetime'].dt.day_name()
    df['month'] = df['Datetime'].dt.month_name()

    return {
        'hourly': df.groupby('hour').size(),
        'daily': df.groupby('day_of_week').size(),
        'monthly': df.groupby('month').size()
    }

def analyze_engagement(df: pd.DataFrame) -> Dict:
    if 'Name' not in df.columns:
        raise ValueError("DataFrame must contain 'Name' column")

    results = {}
    name_counts = df['Name'].value_counts()
    results['message_counts'] = name_counts
    
    if 'IsMedia' in df.columns:
        results['media_counts'] = df[df['IsMedia']]['Name'].value_counts()
    
    if 'Text' in df.columns:
        df['word_count'] = df['Text'].apply(lambda x: len(str(x).split()))
        results['word_stats'] = df.groupby('Name')['word_count'].agg(['mean', 'median', 'sum'])
    
    return results

def analyze_user_engagement(df: pd.DataFrame) -> Dict:
    """Analyze user engagement metrics."""
    results = {}
    df_analysis = df.copy()
    media_pattern = re.compile(r'<Media.*?>', re.IGNORECASE)
    
    # Media analysis
    df_analysis['IsMedia'] = df_analysis['Text'].apply(lambda x: bool(media_pattern.search(str(x))))
    
    if 'Name' in df_analysis.columns:
        media_counts = df_analysis[df_analysis['IsMedia']].groupby('Name').size().reset_index(name='Media_Count')
        total_media = media_counts['Media_Count'].sum()
        media_counts['Percentage'] = (media_counts['Media_Count'] / total_media * 100).round(1) if total_media > 0 else 0.0
        
        results['media_analysis'] = {
            'data': media_counts, 
            'counts': total_media,
            'percentage': media_counts.set_index('Name')['Percentage'].to_dict()
        }
    
    # Emoji analysis
    if 'Text' in df_analysis.columns and 'Name' in df_analysis.columns:
        emoji_pattern = re.compile(
    r'['
    r'\U0001F600-\U0001F64F'  # Emoticons
    r'\U0001F300-\U0001F5FF'  # Symbols & pictographs
    r'\U0001F680-\U0001F6FF'  # Transport & map symbols
    r'\U0001F1E0-\U0001F1FF'  # Flags
    r'\U00002500-\U00002BEF'  # Chinese/Japanese/Korean characters
    r'\U00002702-\U000027B0'  # Dingbats
    r'\U00002702-\U000027B0'  # Dingbats
    r'\U000024C2-\U0001F251'  # Enclosed characters
    r'\U0001f926-\U0001f937'  # Person gestures
    r'\U00010000-\U0010ffff'  # Supplementary private use area
    r'\u2640-\u2642'  # Gender symbols
    r'\u2600-\u2B55'  # Misc symbols
    r'\u200d'  # Zero-width joiner
    r'\u23cf'  # Control symbol
    r'\u23e9'  # Play button
    r'\u231a'  # Watch symbol
    r'\ufe0f'  # Variation selector
    r'\u3030'  # Japanese wave dash
    r']|:[a-zA-Z0-9_]+:'  # Emojis in text form like :smile:
    )
        
        def count_emojis(text):
            if not isinstance(text, str) or media_pattern.search(text):
                return 0
            return len(emoji_pattern.findall(text))
        
        df_analysis['Emoji_Count'] = df_analysis['Text'].apply(count_emojis)
        
        emoji_counts = df_analysis.groupby('Name')['Emoji_Count'].agg(['sum', 'mean']).reset_index()
        emoji_counts.columns = ['Name', 'Total_Emojis', 'Avg_Emojis_Per_Message']
        results['emoji_analysis'] = emoji_counts
    
    # Word count analysis
    if 'Text' in df_analysis.columns and 'Name' in df_analysis.columns:
        def count_words(text):
            if not isinstance(text, str) or media_pattern.search(text):
                return 0
            
            clean_text = emoji.replace_emoji(text, replace=' ')
            words = re.sub(r'[^\w\s]', '', clean_text).split()
            return len(words)
        
        df_analysis['Word_Count'] = df_analysis['Text'].apply(count_words)
        
        word_stats = df_analysis.groupby('Name')['Word_Count'].agg(['sum', 'mean', 'median']).reset_index()
        word_stats.columns = ['Name', 'Total_Words', 'Avg_Words_Per_Message', 'Median_Words_Per_Message']
        results['word_analysis'] = word_stats
    
    return results