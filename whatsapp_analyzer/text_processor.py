# text_processor.py

import re
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, List, Union
import emoji
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('sentiments/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

def count_emojis(text: str) -> int:
    """Count the number of emojis in text"""
    if not isinstance(text, str):
        return 0
    return sum(1 for char in text if char in emoji.EMOJI_DATA)

def extract_emojis(text: str) -> str:
    """Extract all emojis from text as a string"""
    if not isinstance(text, str):
        return ""
    return ''.join([char for char in text if char in emoji.EMOJI_DATA])

def clean_text(text: str) -> str:
    """Clean text by removing URLs, special characters, etc."""
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove special characters but keep emojis
    text = ''.join([c for c in text if c.isalnum() or c.isspace() or c in emoji.EMOJI_DATA])
    
    return text.strip()

def preprocess_dataframe(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Preprocess the DataFrame for analysis
    
    Args:
        df: Input DataFrame with chat data
        
    Returns:
        Preprocessed DataFrame or None if input is invalid
    """
    # Check if df is None or empty
    if df is None:
        logger.error("Cannot preprocess: DataFrame is None")
        return None
    
    if df.empty:
        logger.error("Cannot preprocess: DataFrame is empty")
        return None
    
    try:
        # Make a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Filter out system messages if Name column exists
        if 'Name' in processed_df.columns:
            # First check if Name column has any null values and handle them
            null_names = processed_df['Name'].isnull().sum()
            if null_names > 0:
                logger.warning(f"Found {null_names} rows with null names. Filling with 'Unknown'.")
                processed_df['Name'] = processed_df['Name'].fillna('Unknown')
            
            # Now filter out system messages
            initial_count = len(processed_df)
            processed_df = processed_df[~processed_df['Name'].str.lower().str.contains('system', na=False)]
            filtered_count = initial_count - len(processed_df)
            if filtered_count > 0:
                logger.info(f"Filtered out {filtered_count} system messages")
                
        # Handle null Text values
        if 'Text' in processed_df.columns:
            null_texts = processed_df['Text'].isnull().sum()
            if null_texts > 0:
                logger.warning(f"Found {null_texts} rows with null text. Filling with empty string.")
                processed_df['Text'] = processed_df['Text'].fillna('')
                
            # Add text-based features
            processed_df['CleanText'] = processed_df['Text'].apply(clean_text)
            processed_df['WordCount'] = processed_df['Text'].astype(str).apply(lambda x: len(str(x).split()))
            processed_df['CharCount'] = processed_df['Text'].astype(str).apply(len)
            processed_df['EmojiCount'] = processed_df['Text'].astype(str).apply(count_emojis)
            processed_df['Emojis'] = processed_df['Text'].astype(str).apply(extract_emojis)
            
            # Flag media messages
            if 'IsMedia' not in processed_df.columns:
                processed_df['IsMedia'] = processed_df['Text'].astype(str).str.contains('<Media omitted>', case=False)
                
        logger.info(f"Preprocessing complete. DataFrame shape: {processed_df.shape}")
        return processed_df
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        return None

def generate_word_counts(df: pd.DataFrame, min_count: int = 5) -> Dict[str, int]:
    """Generate word frequency counts from messages"""
    if df is None or 'Text' not in df.columns:
        return {}
        
    try:
        # Combine all text
        all_text = ' '.join(df['Text'].astype(str).tolist())
        
        # Tokenize and clean
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(all_text.lower())
        
        # Remove stopwords, punctuation, and short words
        clean_tokens = [
            word for word in tokens 
            if word not in stop_words
            and word not in string.punctuation
            and len(word) > 2
        ]
        
        # Count words
        word_counts = {}
        for word in clean_tokens:
            word_counts[word] = word_counts.get(word, 0) + 1
            
        # Filter by minimum count
        filtered_counts = {word: count for word, count in word_counts.items() if count >= min_count}
        
        # Sort by frequency (descending)
        return dict(sorted(filtered_counts.items(), key=lambda item: item[1], reverse=True))
        
    except Exception as e:
        logger.error(f"Error generating word counts: {str(e)}")
        return {}

def get_user_statistics(df: pd.DataFrame) -> Dict[str, Dict]:
    """Generate per-user statistics"""
    if df is None or 'Name' not in df.columns:
        return {}
        
    try:
        stats = {}
        
        for name, group in df.groupby('Name'):
            user_stats = {
                'message_count': len(group),
                'word_count': group.get('WordCount', 0).sum(),
                'char_count': group.get('CharCount', 0).sum(),
                'emoji_count': group.get('EmojiCount', 0).sum(),
                'media_count': group.get('IsMedia', False).sum(),
                'avg_words_per_message': group.get('WordCount', 0).mean(),
                'favorite_emojis': get_top_emojis(group, 5)
            }
            
            # Add hourly activity if datetime info available
            if 'Hour' in group.columns:
                hours = group['Hour'].value_counts()
                user_stats['peak_hour'] = hours.idxmax() if not hours.empty else None
                
            stats[name] = user_stats
            
        return stats
        
    except Exception as e:
        logger.error(f"Error generating user statistics: {str(e)}")
        return {}

def get_top_emojis(df: pd.DataFrame, top_n: int = 5) -> Dict[str, int]:
    """Get top N emojis from a DataFrame"""
    if 'Emojis' not in df.columns:
        return {}
        
    all_emojis = ''.join(df['Emojis'].astype(str).tolist())
    emoji_counts = {}
    
    for char in all_emojis:
        emoji_counts[char] = emoji_counts.get(char, 0) + 1
        
    # Sort and return top N
    top_emojis = dict(sorted(emoji_counts.items(), key=lambda x: x[1], reverse=True)[:top_n])
    return top_emojis
