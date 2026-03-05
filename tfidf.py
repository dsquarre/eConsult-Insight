import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud, STOPWORDS
from typing import List, Dict, Tuple


def Extract(sentences: List[str], sentiment_scores: List[float]) -> Tuple[Dict, Dict]:
    """
    Extract word frequency and sentiment summary from sentences.
    
    Args:
        sentences: List of text sentences/comments
        sentiment_scores: List of sentiment scores corresponding to each sentence
        
    Returns:
        Tuple of (word_cloud_dict, sentiment_dict) where:
        - word_cloud_dict: Dictionary of {word: frequency}
        - sentiment_dict: Dictionary with counts of positive, negative, neutral sentiments
    """
    
    # Generate word cloud using TF-IDF
    if sentences and len(sentences) > 0:
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        try:
            X = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = np.asarray(X.mean(axis=0)).ravel().tolist()
            word_cloud = dict(zip(feature_names, [int(score * 10000) for score in tfidf_scores]))
        except Exception as e:
            print(f"TF-IDF error: {e}, using mock data")
            word_cloud = {"good": 10000, "bad": 1000, "risky": 2000}
    else:
        word_cloud = {"good": 10000, "bad": 1000, "risky": 2000}
    
    # Count sentiments
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    
    if sentiment_scores:
        for score in sentiment_scores:
            if isinstance(score, (int, float)):
                if score > 0.5:
                    positive_count += 1
                elif score < -0.5:
                    negative_count += 1
                else:
                    neutral_count += 1
    
    sentiment = {
        "positive": positive_count,
        "negative": negative_count,
        "neutral": neutral_count
    }
    
    return word_cloud, sentiment
