# text_processor.py

import re
import pandas as pd
import logging
from typing import Optional, Dict
import emoji
import string
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure NLTK resources are available (including Italian stopwords)
for resource in ('tokenizers/punkt', 'corpora/stopwords', 'tokenizers/punkt_tab'):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split('/')[-1], quiet=True)

# Combined English + Italian stopwords
_STOP_WORDS = (
    set(stopwords.words('english')) |
    set(stopwords.words('italian'))
)


def count_emojis(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return sum(1 for char in text if char in emoji.EMOJI_DATA)


def extract_emojis(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return ''.join(char for char in text if char in emoji.EMOJI_DATA)


def clean_text(text: str) -> str:
    """Remove URLs and special characters while preserving emojis."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+', '', text)
    text = ''.join(c for c in text if c.isalnum() or c.isspace() or c in emoji.EMOJI_DATA)
    return text.strip()


def preprocess_dataframe(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        logger.error("Cannot preprocess: DataFrame is None or empty")
        return None

    try:
        processed_df = df.copy()

        if 'Name' in processed_df.columns:
            null_names = processed_df['Name'].isnull().sum()
            if null_names > 0:
                logger.warning(f"Filling {null_names} null names with 'Unknown'")
                processed_df['Name'] = processed_df['Name'].fillna('Unknown')
            initial = len(processed_df)
            processed_df = processed_df[
                ~processed_df['Name'].str.lower().str.contains('system', na=False)
            ]
            filtered = initial - len(processed_df)
            if filtered:
                logger.info(f"Filtered {filtered} system messages")

        if 'Text' in processed_df.columns:
            processed_df['Text'] = processed_df['Text'].fillna('')
            processed_df['CleanText'] = processed_df['Text'].apply(clean_text)
            processed_df['WordCount'] = processed_df['Text'].astype(str).apply(
                lambda x: len(x.split())
            )
            processed_df['CharCount'] = processed_df['Text'].astype(str).apply(len)
            processed_df['EmojiCount'] = processed_df['Text'].astype(str).apply(count_emojis)
            processed_df['Emojis'] = processed_df['Text'].astype(str).apply(extract_emojis)
            if 'IsMedia' not in processed_df.columns:
                processed_df['IsMedia'] = processed_df['Text'].astype(str).str.contains(
                    r'<Media omitted>|<.+omesso>', case=False, regex=True
                )

        logger.info(f"Preprocessing complete. Shape: {processed_df.shape}")
        return processed_df

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return None


def get_user_statistics(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Return per-participant statistics including message count, word count,
    average words per message, and top emojis.
    """
    if df is None or df.empty or 'Name' not in df.columns:
        return {}

    stats = {}
    for name, group in df.groupby('Name'):
        message_count = len(group)
        word_count = int(group['WordCount'].sum()) if 'WordCount' in group.columns else 0
        avg_words = round(group['WordCount'].mean(), 1) if 'WordCount' in group.columns else 0.0

        # Top emojis
        all_emojis = ''.join(
            group['Emojis'].dropna().astype(str).tolist()
            if 'Emojis' in group.columns
            else []
        )
        emoji_counter = Counter(all_emojis)
        top_emojis = dict(emoji_counter.most_common(5)) if emoji_counter else {}

        stats[name] = {
            'message_count': message_count,
            'word_count': word_count,
            'avg_words_per_message': avg_words,
            'favorite_emojis': top_emojis,
        }

    return stats


def generate_word_counts(df: pd.DataFrame, min_count: int = 5) -> Dict[str, int]:
    """Generate word-frequency counts using combined EN+IT stopwords."""
    if df is None or 'Text' not in df.columns:
        return {}
    try:
        all_text = ' '.join(df['Text'].astype(str).tolist())
        tokens = word_tokenize(all_text.lower())
        clean_tokens = [
            w for w in tokens
            if w not in _STOP_WORDS
            and w not in string.punctuation
            and len(w) > 2
        ]
        word_counts: Dict[str, int] = {}
        for w in clean_tokens:
            word_counts[w] = word_counts.get(w, 0) + 1
        filtered = {w: c for w, c in word_counts.items() if c >= min_count}
        return dict(sorted(filtered.items(), key=lambda x: x[1], reverse=True))
    except Exception as e:
        logger.error(f"Error generating word counts: {e}")
        return {}
