# analyzer.py

import os                          # FIX 1: was missing — os.getenv("HF_TOKEN") crashed on import
import re
import time
import logging
import requests
import numpy as np
import pandas as pd
import emoji
from tqdm import tqdm
from textblob import TextBlob
from empath import Empath
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime

# ── HF Inference API setup ────────────────────────────────────────────────────

HF_API_URL = (
    "https://api-inference.huggingface.co/models/"
    "cardiffnlp/twitter-xlm-roberta-base-sentiment"
)
HF_TOKEN_ENV_VAR = "HF_TOKEN"

# FIX 2: removed stray `print("Using HF API")` that was sitting between
#         two function definitions and executed on every import.

_SENTIMENT_CACHE: Dict[Tuple[str, str], Dict[str, Union[float, str]]] = {}

logger = logging.getLogger(__name__)

# ── Sentiment back-ends ───────────────────────────────────────────────────────

_vader = SentimentIntensityAnalyzer()


def _detect_language(text: str) -> str:
    """Lightweight language detection. Returns ISO-639-1 code or 'en' on failure."""
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "en"


def _translate_to_english(text: str) -> str:
    """Translate text to English using deep_translator (Google backend)."""
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text


def configure_hf_token(token: Optional[str]) -> None:
    """Set or clear the Hugging Face token used by the sentiment backend."""
    if token and str(token).strip():
        os.environ[HF_TOKEN_ENV_VAR] = str(token).strip()
    else:
        os.environ.pop(HF_TOKEN_ENV_VAR, None)


def get_hf_token() -> Optional[str]:
    token = os.getenv(HF_TOKEN_ENV_VAR)
    return token.strip() if token and token.strip() else None


def has_hf_token() -> bool:
    return get_hf_token() is not None


# ── HF API helpers ────────────────────────────────────────────────────────────

def _emoji_sentiment(text: str) -> Optional[str]:
    """Fast emoji-based sentiment shortcut before hitting the API."""
    positive = {"😂", "😄", "😍", "🥰", "😁", "👍"}
    negative = {"😡", "😢", "😭", "👎", "😤"}
    chars = set(text)
    if chars & positive:
        return "positive"
    if chars & negative:
        return "negative"
    return None


def _hf_request(inputs: Union[str, List[str]], max_retries: int = 5) -> list:
    """
    POST text to the HF Inference API and return the raw scores list.

    FIX 3: added exponential backoff with jitter instead of flat 2 s sleep.
    FIX 4: raise on non-200 HTTP status so transient errors are properly retried.
    FIX 6: when the model is still loading HF returns {"estimated_time": N};
            we now sleep for exactly that duration instead of guessing.
    """
    hf_token = get_hf_token()
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN not set; skipping Hugging Face API sentiment backend"
        )

    for attempt in range(max_retries):
        response = requests.post(
            HF_API_URL,
            headers={"Authorization": f"Bearer {hf_token}"},
            json={"inputs": inputs},
            timeout=30,
        )

        # FIX 4: surface HTTP errors (503 model-loading, 429 rate-limit, etc.)
        if response.status_code not in (200, 503):
            response.raise_for_status()

        result = response.json()

        # FIX 6: model is still warming up — respect the estimated_time hint
        if isinstance(result, dict) and "error" in result:
            wait = result.get("estimated_time", 2 ** attempt)   # exponential fallback
            logger.warning(
                "HF API: %s  (attempt %d/%d — waiting %.1f s)",
                result["error"], attempt + 1, max_retries, wait,
            )
            time.sleep(wait + 0.5)   # small jitter
            continue

        return result

    raise RuntimeError(
        "HF Inference API failed after multiple retries"
    )


def _parse_hf_output(result) -> Tuple[str, float]:
    """
    Extract the top-scoring label and its score from the HF API response.

    FIX 5: guard against both response shapes the API can return:
      - Nested:  [[{"label": ..., "score": ...}, ...]]   (standard)
      - Flat:    [ {"label": ..., "score": ...}, ...]    (some model variants)
    Returns (label, score) normalised to "positive" / "neutral" / "negative".
    """
    if not isinstance(result, list) or not result:
        raise ValueError(f"Unexpected HF response shape: {result!r}")

    # Unwrap one level of nesting if needed.
    scores = result[0] if isinstance(result[0], list) else result

    best = max(scores, key=lambda x: x["score"])
    label = best["label"].lower()
    score = best["score"]

    label_map = {
        "label_0": "negative",
        "label_1": "neutral",
        "label_2": "positive",
    }
    if label in label_map:
        return label_map[label], score
    if label in ("positive", "neutral", "negative"):
        return label, score
    if "pos" in label:
        return "positive", score
    if "neg" in label:
        return "negative", score
    return "neutral", score


def _empty_sentiment(backend: str = "local") -> Dict[str, Union[float, str]]:
    return {
        "polarity": 0.0,
        "subjectivity": 0.0,
        "compound": 0.0,
        "sentiment": "neutral",
        "sentiment_backend": backend,
    }


def _local_sentiment(text: str) -> Dict[str, Union[float, str]]:
    clean = re.sub(r"[^\w\s]", "", text.lower())
    blob = TextBlob(clean)
    polarity = blob.sentiment.polarity
    compound = _vader.polarity_scores(clean)["compound"]

    if compound >= 0.25 or polarity >= 0.25:
        sentiment = "positive"
    elif compound <= -0.25 or polarity <= -0.25:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {
        "polarity":     round(polarity, 4),
        "subjectivity": round(blob.sentiment.subjectivity, 4),
        "compound":     round(compound, 4),
        "sentiment":    sentiment,
        "sentiment_backend": "local",
    }


def _hf_sentiment_from_scores(scores) -> Dict[str, Union[float, str]]:
    sentiment, hf_score = _parse_hf_output(scores)
    compound_proxy = hf_score if sentiment == "positive" else (
        -hf_score if sentiment == "negative" else 0.0
    )
    return {
        "polarity":     round(compound_proxy, 4),
        "subjectivity": 0.5,
        "compound":     round(compound_proxy, 4),
        "sentiment":    sentiment,
        "sentiment_backend": "hf",
    }


def _resolve_sentiment_backend(backend: str, use_multilingual: bool) -> str:
    if backend not in {"auto", "hf", "local"}:
        raise ValueError("backend must be one of: auto, hf, local")
    if backend != "auto":
        return backend
    return "hf" if use_multilingual and has_hf_token() else "local"


# ── Core sentiment function ───────────────────────────────────────────────────

def analyze_sentiment(
    text: str, use_multilingual: bool = True, backend: str = "auto"
) -> Dict[str, Union[float, str]]:
    """
    Analyse the sentiment of *text* in any language.

    Priority:
    1. Cache lookup          (instant — avoids re-processing duplicates)
    2. Emoji shortcut        (instant — no network call)
    3. Skip very short text
    4. HF Inference API      (GPU-backed, cardiffnlp/twitter-xlm-roberta-base-sentiment)
    5. VADER + TextBlob      (fallback when the API is unavailable)
    """
    resolved_backend = _resolve_sentiment_backend(backend, use_multilingual)
    empty = _empty_sentiment(resolved_backend)

    if not isinstance(text, str) or not text.strip():
        return empty

    text = text.strip()

    # 1. Cache
    cache_key = (resolved_backend, text)
    if cache_key in _SENTIMENT_CACHE:
        return _SENTIMENT_CACHE[cache_key].copy()

    # 2. Emoji shortcut
    emoji_result = _emoji_sentiment(text)
    if emoji_result:
        result = {**empty, "sentiment": emoji_result, "sentiment_backend": "emoji"}
        _SENTIMENT_CACHE[cache_key] = result
        return result.copy()

    # 3. Too short to be meaningful
    if len(text) < 3:
        return empty

    if resolved_backend == "hf":
        raw = _hf_request(text[:512])
        result = _hf_sentiment_from_scores(raw)
    else:
        result = _local_sentiment(text)

    _SENTIMENT_CACHE[cache_key] = result
    return result.copy()


def analyze_sentiment_batch(
    texts: List[str], use_multilingual: bool = True, backend: str = "auto",
    batch_size: int = 32
) -> pd.DataFrame:
    resolved_backend = _resolve_sentiment_backend(backend, use_multilingual)

    if resolved_backend != "hf":
        results = [
            analyze_sentiment(t, use_multilingual=use_multilingual, backend=resolved_backend)
            for t in tqdm(texts, desc=f"Analysing sentiment ({resolved_backend})")
        ]
        return pd.DataFrame(results)

    if not has_hf_token():
        raise RuntimeError("Hugging Face sentiment selected, but HF_TOKEN is not set.")

    results: List[Optional[Dict[str, Union[float, str]]]] = [None] * len(texts)
    pending: Dict[str, List[int]] = {}

    for index, value in enumerate(texts):
        text = value.strip() if isinstance(value, str) else ""
        if not text:
            results[index] = _empty_sentiment("hf")
            continue
        cache_key = ("hf", text)
        if cache_key in _SENTIMENT_CACHE:
            results[index] = _SENTIMENT_CACHE[cache_key].copy()
            continue
        emoji_result = _emoji_sentiment(text)
        if emoji_result:
            result = {
                **_empty_sentiment("hf"),
                "sentiment": emoji_result,
                "sentiment_backend": "emoji",
            }
            _SENTIMENT_CACHE[cache_key] = result
            results[index] = result.copy()
            continue
        if len(text) < 3:
            results[index] = _empty_sentiment("hf")
            continue
        pending.setdefault(text, []).append(index)

    unique_texts = list(pending.keys())
    for start in tqdm(
        range(0, len(unique_texts), batch_size),
        desc="Analysing sentiment (hf)",
    ):
        batch_texts = unique_texts[start:start + batch_size]
        raw = _hf_request([text[:512] for text in batch_texts])
        if len(batch_texts) == 1:
            raw_items = [raw]
        elif isinstance(raw, list) and len(raw) == len(batch_texts):
            raw_items = raw
        else:
            raise ValueError(f"Unexpected HF batch response shape: {raw!r}")

        for text, scores in zip(batch_texts, raw_items):
            result = _hf_sentiment_from_scores(scores)
            _SENTIMENT_CACHE[("hf", text)] = result
            for index in pending[text]:
                results[index] = result.copy()

    if any(result is None for result in results):
        raise RuntimeError("Sentiment analysis did not produce a result for every message.")

    return pd.DataFrame(results)


# ── Personality / Empath ──────────────────────────────────────────────────────

default_personality_traits = [
    "joy", "anger", "sadness", "family",
    "friend", "social", "positive_emotion",
    "negative_emotion",
]


def analyze_personality_texts(
    texts: List[str], traits: List[str] = None
) -> pd.DataFrame:
    """Empath analysis. Translates non-English text first."""
    if traits is None:
        traits = default_personality_traits

    lexicon = Empath()
    results = []

    for text in tqdm(texts, desc="Analysing personality"):
        if not isinstance(text, str) or not text.strip():
            results.append({trait: 0 for trait in traits})
            continue

        lang = _detect_language(text)
        analysis_text = _translate_to_english(text) if lang != "en" else text

        analysis = lexicon.analyze(analysis_text, categories=traits, normalize=True)
        results.append({trait: analysis.get(trait, 0) for trait in traits})

    return pd.DataFrame(results)


# ── Topic modelling ───────────────────────────────────────────────────────────

def perform_topic_modeling(texts: List[str]) -> Tuple:
    """
    Run BERTopic using settings from config.py
    (BERTOPIC_PARAMS, VECTORIZER_PARAMS, BERTOPIC_EMBEDDING_MODEL).
    """
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer

    try:
        from whatsapp_analyzer.config import (
            BERTOPIC_PARAMS, VECTORIZER_PARAMS, BERTOPIC_EMBEDDING_MODEL,
        )
    except ImportError:
        try:
            from config import BERTOPIC_PARAMS, VECTORIZER_PARAMS, BERTOPIC_EMBEDDING_MODEL
        except ImportError:
            BERTOPIC_PARAMS = {}
            VECTORIZER_PARAMS = {}
            BERTOPIC_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    if not texts or len(texts) < 10:
        logger.warning("Insufficient texts for topic modeling")
        return None, None, None

    vectorizer_model = CountVectorizer(**VECTORIZER_PARAMS) if VECTORIZER_PARAMS else None

    embedding_model = None
    if BERTOPIC_EMBEDDING_MODEL:
        try:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer(BERTOPIC_EMBEDDING_MODEL)
        except Exception as e:
            logger.warning("Could not load embedding model '%s': %s", BERTOPIC_EMBEDDING_MODEL, e)

    bertopic_kwargs = {
        k: v for k, v in BERTOPIC_PARAMS.items()
        if k not in ("language",)
    }
    if embedding_model is not None:
        bertopic_kwargs["embedding_model"] = embedding_model
    if vectorizer_model is not None:
        bertopic_kwargs["vectorizer_model"] = vectorizer_model

    topic_model = BERTopic(**bertopic_kwargs)
    topics, probs = topic_model.fit_transform(texts)
    topic_info = topic_model.get_topic_info()
    return topic_model, topics, topic_info


# ── Temporal / engagement helpers (language-agnostic) ────────────────────────

def analyze_response_times(df: pd.DataFrame) -> pd.DataFrame:
    if "Datetime" not in df.columns or "Name" not in df.columns:
        raise ValueError("DataFrame must contain 'Datetime' and 'Name' columns")

    df = df.sort_values("Datetime")
    results = []
    for i in range(1, len(df)):
        current, previous = df.iloc[i], df.iloc[i - 1]
        if current["Name"] != previous["Name"]:
            rt = (current["Datetime"] - previous["Datetime"]).total_seconds() / 60
            results.append({
                "responder":        current["Name"],
                "response_time_min": rt,
                "previous_sender":  previous["Name"],
            })
    return pd.DataFrame(results)


def temporal_analysis(df: pd.DataFrame) -> Dict:
    if "Datetime" not in df.columns:
        raise ValueError("DataFrame must contain 'Datetime' column")

    df = df.copy()
    df["hour"]        = df["Datetime"].dt.hour
    df["day_of_week"] = df["Datetime"].dt.day_name()
    df["month"]       = df["Datetime"].dt.month_name()

    return {
        "hourly":  df.groupby("hour").size(),
        "daily":   df.groupby("day_of_week").size(),
        "monthly": df.groupby("month").size(),
    }


def analyze_engagement(df: pd.DataFrame) -> Dict:
    if "Name" not in df.columns:
        raise ValueError("DataFrame must contain 'Name' column")

    results = {"message_counts": df["Name"].value_counts()}
    if "IsMedia" in df.columns:
        results["media_counts"] = df[df["IsMedia"]]["Name"].value_counts()
    if "Text" in df.columns:
        df = df.copy()
        df["word_count"] = df["Text"].apply(lambda x: len(str(x).split()))
        results["word_stats"] = df.groupby("Name")["word_count"].agg(["mean", "median", "sum"])
    return results


def analyze_user_engagement(df: pd.DataFrame) -> Dict:
    results = {}
    df_analysis = df.copy()
    media_pattern = re.compile(r"<Media.*?>|<.+omesso>", re.IGNORECASE)

    # Media
    df_analysis["IsMedia"] = df_analysis["Text"].apply(
        lambda x: bool(media_pattern.search(str(x)))
    )
    if "Name" in df_analysis.columns:
        mc = (
            df_analysis[df_analysis["IsMedia"]]
            .groupby("Name").size()
            .reset_index(name="Media_Count")
        )
        total = mc["Media_Count"].sum()
        mc["Percentage"] = (mc["Media_Count"] / total * 100).round(1) if total > 0 else 0.0
        results["media_analysis"] = {
            "data":       mc,
            "counts":     total,
            "percentage": mc.set_index("Name")["Percentage"].to_dict(),
        }

    # Emoji
    emoji_pattern = re.compile(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        r"\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251"
        r"\U0001f926-\U0001f937\U00010000-\U0010ffff\u2640-\u2642"
        r"\u2600-\u2B55\u200d\u23cf\u23e9\u231a\ufe0f\u3030]|:[a-zA-Z0-9_]+:"
    )

    def count_emojis_local(text):
        if not isinstance(text, str) or media_pattern.search(text):
            return 0
        return len(emoji_pattern.findall(text))

    df_analysis["Emoji_Count"] = df_analysis["Text"].apply(count_emojis_local)
    ec = df_analysis.groupby("Name")["Emoji_Count"].agg(["sum", "mean"]).reset_index()
    ec.columns = ["Name", "Total_Emojis", "Avg_Emojis_Per_Message"]
    results["emoji_analysis"] = ec

    # Word count
    def count_words(text):
        if not isinstance(text, str) or media_pattern.search(text):
            return 0
        clean = emoji.replace_emoji(text, replace=" ")
        return len(re.sub(r"[^\w\s]", "", clean).split())

    df_analysis["Word_Count"] = df_analysis["Text"].apply(count_words)
    wc = df_analysis.groupby("Name")["Word_Count"].agg(["sum", "mean", "median"]).reset_index()
    wc.columns = ["Name", "Total_Words", "Avg_Words_Per_Message", "Median_Words_Per_Message"]
    results["word_analysis"] = wc

    return results
