import platform
import os
import torch as _torch
import logging
import multiprocessing

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CPU configuration for M1
os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(multiprocessing.cpu_count())
os.environ['OPENBLAS_NUM_THREADS'] = str(multiprocessing.cpu_count())

# Initialize global CPU flags
DEVICE = _torch.device("cpu")
logger.info(f"Running on Apple M1 with {multiprocessing.cpu_count()} CPU cores")

# Import all submodules
from .data_loader import load_and_parse_chat
from .text_processor import count_emojis, clean_text, preprocess_dataframe
from .analyzer import (
    analyze_sentiment,
    analyze_sentiment_batch,
    analyze_personality_texts,
    perform_topic_modeling,
    analyze_response_times,
    temporal_analysis,
    analyze_engagement,
    analyze_user_engagement
)
from .visualizer import (
    plot_sentiment,
    plot_temporal,
    plot_participants,
    plot_personality,
    plot_topics_bertopic,
    plot_user_engagement
)

from .parallel_utils import (
    set_multiprocessing_context,
    parallel_process,
    _process_sentiment_batch,
    parallel_sentiment_analysis,
    _process_personality_batch,
    parallel_personality_analysis,
    parallel_topic_modeling,
    parallel_user_engagement,
    parallel_response_times,
    parallel_temporal_analysis
)
from .config import (
    INPUT_CHAT_FILE,
    RAW_CSV_OUTPUT,
    CLEANED_CSV_OUTPUT,
    SENTIMENT_CSV_OUTPUT,
    PERSONALITY_CSV_OUTPUT,
    TOPIC_CSV_OUTPUT,
    POSITIVE_THRESHOLD,
    NEGATIVE_THRESHOLD,
    NUM_TOPICS,
    TOP_N_WORDS,
    MIN_MESSAGE_LENGTH,
    SAMPLE_SIZE,
    COLOR_PALETTE,
    BERTOPIC_PARAMS,
    VECTORIZER_PARAMS,
    PERSONALITY_TRAITS,
    SENTIMENT_WEIGHTS,
    TIME_BINS,
    CPU_OPTIMIZATION
)

__all__ = [
    # Core functionality
    'load_and_parse_chat',
    'count_emojis',
    'clean_text',
    'preprocess_dataframe',

    # Analysis functions
    'analyze_sentiment',
    'analyze_sentiment_batch',
    'analyze_personality_texts',
    'perform_topic_modeling',
    'analyze_response_times',
    'temporal_analysis',
    'analyze_engagement',
    'analyze_user_engagement',

    # Visualization
    'plot_sentiment',
    'plot_temporal',
    'plot_participants',
    'plot_personality',
    'plot_topics_bertopic',
    'plot_user_engagement',

    # Parallel processing
    'parallel_process',
    'set_multiprocessing_context',
    '_process_sentiment_batch',
    'parallel_sentiment_analysis',
    '_process_personality_batch',
    'parallel_personality_analysis',
    'parallel_topic_modeling',
    'parallel_user_engagement',
    'parallel_response_times',
    'parallel_temporal_analysis',

    # Hardware configuration
    'DEVICE',

    # Configuration
    'INPUT_CHAT_FILE',
    'RAW_CSV_OUTPUT',
    'CLEANED_CSV_OUTPUT',
    'SENTIMENT_CSV_OUTPUT',
    'PERSONALITY_CSV_OUTPUT',
    'TOPIC_CSV_OUTPUT',
    'POSITIVE_THRESHOLD',
    'NEGATIVE_THRESHOLD',
    'NUM_TOPICS',
    'TOP_N_WORDS',
    'MIN_MESSAGE_LENGTH',
    'SAMPLE_SIZE',
    'COLOR_PALETTE',
    'BERTOPIC_PARAMS',
    'VECTORIZER_PARAMS',
    'PERSONALITY_TRAITS',
    'SENTIMENT_WEIGHTS',
    'TIME_BINS',
    'CPU_OPTIMIZATION'
]
