# config.py

import matplotlib.pyplot as plt
import seaborn as sns

# ── File paths ────────────────────────────────────────────────────────────────
INPUT_CHAT_FILE       = "chat.txt"
RAW_CSV_OUTPUT        = "whatsapp_chat.csv"
CLEANED_CSV_OUTPUT    = "cleaned_chat.csv"
SENTIMENT_CSV_OUTPUT  = "sentiment_chat.csv"
PERSONALITY_CSV_OUTPUT= "personality_chat.csv"
TOPIC_CSV_OUTPUT      = "topic_chat.csv"

# ── Analysis thresholds ───────────────────────────────────────────────────────
POSITIVE_THRESHOLD  = 0.05
NEGATIVE_THRESHOLD  = -0.05
NUM_TOPICS          = 5
TOP_N_WORDS         = 15
MIN_MESSAGE_LENGTH  = 3
SAMPLE_SIZE         = 5000

# ── Visualisation ─────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.dpi']   = 100
plt.rcParams['savefig.dpi']  = 300
sns.set_palette('viridis')

COLOR_PALETTE = {
    'positive':  '#4CAF50',   # green
    'negative':  '#F44336',   # red
    'neutral':   '#9E9E9E',   # grey
    'highlight': '#2196F3',   # blue
    'person1':   '#FF9F43',   # orange
    'person2':   '#26C6DA',   # cyan
    'system':    '#9575CD',   # purple
}

# ── BERTopic ──────────────────────────────────────────────────────────────────
# Uses a multilingual sentence-transformer so Italian (and other languages) work.
# Set BERTOPIC_EMBEDDING_MODEL to None to use BERTopic's default (English-only).
BERTOPIC_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

BERTOPIC_PARAMS = {
    'language':               'multilingual',   # changed from 'english'
    'calculate_probabilities': False,
    'verbose':                 True,
    'min_topic_size':          10,
    'nr_topics':               NUM_TOPICS,
    'low_memory':              True,
}

VECTORIZER_PARAMS = {
    'stop_words': None,      # None = no stopword removal at vectorizer level;
                             # language-specific stopwords are handled in text_processor.py
    'ngram_range': (1, 1),
    'min_df':      2,
    'max_df':      0.95,
    'dtype':       'float32',
}

# ── Personality traits ────────────────────────────────────────────────────────
PERSONALITY_TRAITS = [
    'joy', 'anger', 'sadness',
    'family', 'friend', 'social',
]

# ── Sentiment weights (for future weighted ensemble) ─────────────────────────
SENTIMENT_WEIGHTS = {
    'transformer': 0.6,   # XLM-RoBERTa (multilingual)
    'vader':       0.3,
    'textblob':    0.1,
}

# ── Time bins ─────────────────────────────────────────────────────────────────
TIME_BINS = {
    'morning':   (6,  12),
    'afternoon': (12, 18),
    'evening':   (18, 23),
    'night':     (23,  6),
}

# ── CPU optimisation ──────────────────────────────────────────────────────────
CPU_OPTIMIZATION = {
    'batch_size':               256,
    'max_text_length':          384,
    'min_batch_cpu':             50,
    'max_batch_cpu':            500,
    'sentiment_batch_size':     128,
    'personality_batch_size':    64,
    'topic_batch_size':          32,
    'num_processes':              6,   # leave 2 cores free for system
    'chunk_size':               100,
}

# ── Multilingual settings ─────────────────────────────────────────────────────
# Supported sentiment backends (in priority order):
#   1. 'transformer'  – cardiffnlp/twitter-xlm-roberta-base-sentiment
#                       (pip install transformers torch)
#   2. 'translate'    – langdetect + deep-translator → VADER + TextBlob
#                       (pip install langdetect deep-translator)
#   3. 'vader'        – English-only fallback (always available)
SENTIMENT_BACKEND_PRIORITY = ['transformer', 'translate', 'vader']

# Translation provider used when transformer model is unavailable.
# 'google' requires internet access; 'libre' can run locally.
TRANSLATION_PROVIDER = 'google'   # options: 'google', 'libre'
