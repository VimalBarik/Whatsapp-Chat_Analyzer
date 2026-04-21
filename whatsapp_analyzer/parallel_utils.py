# parallel_utils.py

import os
from multiprocessing import Pool, cpu_count, set_start_method
from tqdm import tqdm
import pandas as pd
from typing import List, Callable, Any, Dict
from functools import partial

# ── Core count ────────────────────────────────────────────────────────────────
# FIX: call cpu_count() instead of storing the function reference.
CORES_AVAILABLE = cpu_count()
USE_CORES       = max(1, CORES_AVAILABLE - 2)
CHUNK_SIZE      = 100

os.environ.update({
    'OPENBLAS_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'OMP_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1',
})

MP_CONTEXT = 'spawn'
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


# ── Generic parallel processor ────────────────────────────────────────────────

def parallel_process(
    func: Callable,
    data: List[Any],
    desc: str = "Processing",
    min_batch_size: int = 50,
) -> List[Any]:
    """Run *func* over *data* in parallel batches. Falls back to serial for small inputs."""
    if len(data) < min_batch_size:
        return [func(item) for item in data]

    batch_size = max(min(len(data) // USE_CORES, CHUNK_SIZE), 1)
    batches    = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    set_multiprocessing_context()
    with Pool(processes=USE_CORES) as pool:
        results = list(tqdm(
            pool.imap(func, batches),
            total=len(batches),
            desc=desc,
        ))

    return [item for batch in results for item in batch]


# ── Sentiment ─────────────────────────────────────────────────────────────────

def _process_sentiment_batch(args):
    """Worker: analyse sentiment for a batch of texts."""
    batch, multilingual = args
    # Support both package-level and direct invocation
    try:
        from whatsapp_analyzer.analyzer import analyze_sentiment
    except ImportError:
        from analyzer import analyze_sentiment
    return [analyze_sentiment(text, use_multilingual=multilingual) for text in batch]


def parallel_sentiment_analysis(
    texts: List[str], use_multilingual: bool = True
) -> pd.DataFrame:
    """Parallel sentiment analysis."""
    batch_size = max(min(len(texts) // USE_CORES, CHUNK_SIZE), 1)
    batches    = [(texts[i:i + batch_size], use_multilingual)
                  for i in range(0, len(texts), batch_size)]

    if len(texts) < 50:
        results = [r for args in batches for r in _process_sentiment_batch(args)]
    else:
        set_multiprocessing_context()
        with Pool(processes=USE_CORES) as pool:
            nested = list(tqdm(
                pool.imap(_process_sentiment_batch, batches),
                total=len(batches),
                desc="Sentiment Analysis",
            ))
        results = [item for batch in nested for item in batch]

    return pd.DataFrame(results)


# ── Personality ───────────────────────────────────────────────────────────────

def _process_personality_batch(args):
    """Worker: translate non-English text then run Empath analysis."""
    batch, traits = args
    from empath import Empath
    lexicon = Empath()
    results = []
    for text in batch:
        if not isinstance(text, str):
            results.append({t: 0 for t in traits})
            continue
        try:
            try:
                from whatsapp_analyzer.analyzer import _detect_language, _translate_to_english
            except ImportError:
                from analyzer import _detect_language, _translate_to_english
            lang = _detect_language(text)
            if lang != 'en':
                text = _translate_to_english(text)
        except Exception:
            pass
        analysis = lexicon.analyze(text, categories=traits, normalize=True)
        results.append({t: analysis.get(t, 0) for t in traits})
    return results


def parallel_personality_analysis(
    texts: List[str], traits: List[str] = None
) -> pd.DataFrame:
    try:
        from whatsapp_analyzer.analyzer import default_personality_traits
    except ImportError:
        from analyzer import default_personality_traits
    traits = traits or default_personality_traits

    batch_size = max(min(len(texts) // USE_CORES, CHUNK_SIZE), 1)
    batches    = [(texts[i:i + batch_size], traits)
                  for i in range(0, len(texts), batch_size)]

    if len(texts) < 50:
        results = [r for args in batches for r in _process_personality_batch(args)]
    else:
        set_multiprocessing_context()
        with Pool(processes=USE_CORES) as pool:
            nested = list(tqdm(
                pool.imap(_process_personality_batch, batches),
                total=len(batches),
                desc="Personality Analysis",
            ))
        results = [item for batch in nested for item in batch]

    return pd.DataFrame(results)


# ── Topic / engagement (delegate to analyzer) ─────────────────────────────────

def parallel_topic_modeling(texts: List[str]):
    try:
        from whatsapp_analyzer.analyzer import perform_topic_modeling
    except ImportError:
        from analyzer import perform_topic_modeling
    return perform_topic_modeling(texts)


def parallel_user_engagement(df: pd.DataFrame) -> Dict:
    try:
        from whatsapp_analyzer.analyzer import analyze_user_engagement
    except ImportError:
        from analyzer import analyze_user_engagement
    return analyze_user_engagement(df)


def parallel_response_times(df: pd.DataFrame) -> pd.DataFrame:
    try:
        from whatsapp_analyzer.analyzer import analyze_response_times
    except ImportError:
        from analyzer import analyze_response_times
    return analyze_response_times(df)


def parallel_temporal_analysis(df: pd.DataFrame) -> Dict:
    try:
        from whatsapp_analyzer.analyzer import temporal_analysis
    except ImportError:
        from analyzer import temporal_analysis
    return temporal_analysis(df)
