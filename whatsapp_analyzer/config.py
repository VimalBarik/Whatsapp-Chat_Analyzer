import matplotlib.pyplot as plt
import seaborn as sns

# File paths and output configurations
INPUT_CHAT_FILE = "chat.txt"
RAW_CSV_OUTPUT = "whatsapp_chat.csv" 
CLEANED_CSV_OUTPUT = "cleaned_chat.csv"
SENTIMENT_CSV_OUTPUT = "sentiment_chat.csv"
PERSONALITY_CSV_OUTPUT = "personality_chat.csv"
TOPIC_CSV_OUTPUT = "topic_chat.csv"

# Analysis thresholds and parameters
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05
NUM_TOPICS = 5
TOP_N_WORDS = 15
MIN_MESSAGE_LENGTH = 3
SAMPLE_SIZE = 5000

# Visualization settings
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
sns.set_palette('viridis')

# Color palette for visualizations
COLOR_PALETTE = {
    'positive': '#4CAF50',        # Green
    'negative': '#F44336',       # Red
    'neutral': '#9E9E9E',        # Gray
    'highlight': '#2196F3',      # Blue
    'person1': '#FF9F43',        # Orange
    'person2': '#26C6DA',        # Cyan
    'system': '#9575CD'          # Purple
}

# BERTopic model parameters
BERTOPIC_PARAMS = {
    'language': 'english',
    'calculate_probabilities': False,
    'verbose': True,
    'min_topic_size': 10,
    'nr_topics': NUM_TOPICS,
    'low_memory': True  # Important for M1 optimization
}

# CountVectorizer parameters
VECTORIZER_PARAMS = {
    'stop_words': 'english',
    'ngram_range': (1, 1),
    'min_df': 2,
    'max_df': 0.95,
    'dtype': 'float32'
}

# Personality analysis traits
PERSONALITY_TRAITS = [
    'joy', 
    'anger', 
    'sadness', 
    'family', 
    'friend', 
    'social'
]

# Sentiment analysis weights
SENTIMENT_WEIGHTS = {
    'vader': 0.6,
    'textblob': 0.3,
    'emoji': 0.1
}

# Time analysis settings
TIME_BINS = {
    'morning': (6, 12),
    'afternoon': (12, 18),
    'evening': (18, 23),
    'night': (23, 6)
}

# CPU Optimization Parameters for M1
CPU_OPTIMIZATION = {
    'batch_size': 256,  # Reduced for better memory management
    'max_text_length': 384,
    'min_batch_cpu': 50,
    'max_batch_cpu': 500,
    'sentiment_batch_size': 128,
    'personality_batch_size': 64,
    'topic_batch_size': 32,
    'num_processes': 6,  # Leave 2 cores free for system
    'chunk_size': 100
}