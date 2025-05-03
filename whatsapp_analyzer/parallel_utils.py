# parallel_utils.py

import os
import platform
from multiprocessing import Pool, cpu_count, set_start_method
from tqdm import tqdm
import pandas as pd
from typing import List, Callable, Any, Dict
from functools import partial

# Optimized for M1 Mac (8 cores)
CORES_AVAILABLE = cpu_count  # Force using all 8 cores
USE_CORES = CORES_AVAILABLE
CHUNK_SIZE = 100  # Optimal chunk size for M1

os.environ.update({
    'OPENBLAS_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'OMP_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1'
})

MP_CONTEXT = 'spawn'  # Required for M1 Mac

_mp_context_initialized = False

def set_multiprocessing_context():
    global _mp_context_initialized
    if not _mp_context_initialized:
        try:
            set_start_method(MP_CONTEXT, force=True)
            _mp_context_initialized = True
        except RuntimeError:
            pass

set_multiprocessing_context()

def parallel_process(
    func: Callable,
    data: List[Any],
    desc: str = "Processing",
    min_batch_size: int = 50
) -> List[Any]:
    if len(data) < min_batch_size:
        return [func(item) for item in data]

    batch_size = max(min(len(data) // USE_CORES, CHUNK_SIZE), 1)
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    set_multiprocessing_context()

    with Pool(processes=USE_CORES) as pool:
        results = list(tqdm(
            pool.imap(func, batches),
            total=len(batches),
            desc=desc
        ))

    return [item for batch in results for item in batch]

def _process_sentiment_batch(batch: List[str]) -> List[dict]:
    from analyzer import analyze_sentiment
    return [analyze_sentiment(text) for text in batch]

def parallel_sentiment_analysis(texts: List[str]) -> pd.DataFrame:
    results = parallel_process(_process_sentiment_batch, texts, "Sentiment Analysis")
    return pd.DataFrame(results)

def _process_personality_batch(batch: List[str], traits: List[str]) -> List[dict]:
    from empath import Empath
    lexicon = Empath()
    results = []
    for text in batch:
        if isinstance(text, str):
            analysis = lexicon.analyze(text, categories=traits, normalize=True)
            results.append({trait: analysis.get(trait, 0) for trait in traits})
        else:
            results.append({trait: 0 for trait in traits})
    return results

def parallel_personality_analysis(texts: List[str], traits: List[str] = None) -> pd.DataFrame:
    from analyzer import default_personality_traits
    traits = traits or default_personality_traits

    func = partial(_process_personality_batch, traits=traits)
    results = parallel_process(func, texts, "Personality Analysis")
    return pd.DataFrame(results)

def parallel_topic_modeling(texts: List[str]):
    from bertopic import BERTopic
    from analyzer import perform_topic_modeling
    return perform_topic_modeling(texts)

def parallel_user_engagement(df: pd.DataFrame) -> Dict:
    from analyzer import analyze_user_engagement
    return analyze_user_engagement(df)

def parallel_response_times(df: pd.DataFrame) -> pd.DataFrame:
    from analyzer import analyze_response_times
    return analyze_response_times(df)

def parallel_temporal_analysis(df: pd.DataFrame) -> Dict:
    from analyzer import temporal_analysis
    return temporal_analysis(df)