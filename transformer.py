import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import re
import warnings
import os
import tempfile
warnings.filterwarnings('ignore')

class SentimentSummarization:

    def __init__(self):
        """Initialize with rule-based patterns."""

        # Enhanced filler words and patterns
        self.filler_patterns = {
            'fillers': r'\b(um|uh|like|you know|actually|basically|literally|definitely|obviously|really|very|quite|rather|somewhat|kind of|sort of)\b',
            'repetitions': r'\b(\w+)\s+\1\b',  # repeated words
            'excessive_punctuation': r'[!?]{2,}',
            'urls': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'emails': r'\S+@\S+',
            'phones': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        }

    def advanced_preprocess(self, text: str) -> Dict[str, str]:
        
        if not isinstance(text, str):
            return {"original": "", "cleaned": "", "anonymized": ""}

        original = text

        # Step 1: Basic cleaning
        cleaned = text.lower().strip()

        # Step 2: Remove URLs, emails, phones
        for pattern_name, pattern in self.filler_patterns.items():
            if pattern_name in ['urls', 'emails', 'phones']:
                cleaned = re.sub(pattern, f'[{pattern_name.upper()[:-1]}]', cleaned)

        # Step 3: Remove filler words
        cleaned = re.sub(self.filler_patterns['fillers'], '', cleaned, flags=re.IGNORECASE)

        # Step 4: Remove repetitions
        cleaned = re.sub(self.filler_patterns['repetitions'], r'\1', cleaned, flags=re.IGNORECASE)

        # Step 5: Clean excessive punctuation
        cleaned = re.sub(self.filler_patterns['excessive_punctuation'], '!', cleaned)

        # Step 6: Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        # Step 7: Anonymization (replace potential names)
        anonymized = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]', cleaned)

        return {
            "original": original,
            "cleaned": cleaned, 
            "anonymized": anonymized
        }

    def sentiment_analysis(self, text: str) -> Dict:
        """Lexicon-based sentiment analysis"""
        positive_indicators = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                              'love', 'perfect', 'outstanding', 'brilliant', 'superb', 'awesome', 
                              'happy', 'satisfied', 'helpful']
        negative_indicators = ['bad', 'terrible', 'awful', 'hate', 'horrible', 'worst', 
                              'disappointing', 'poor', 'unsatisfactory', 'failed', 'broken', 
                              'sad', 'unhappy', 'angry', 'useless']
        neutral_indicators = ['okay', 'average', 'normal', 'fine', 'acceptable', 'decent']

        text_lower = text.lower()

        pos_score = sum(2 if word in text_lower else 0 for word in positive_indicators)
        neg_score = sum(2 if word in text_lower else 0 for word in negative_indicators)
        neu_score = sum(1 if word in text_lower else 0 for word in neutral_indicators)

        total_score = pos_score + neg_score + neu_score

        if total_score == 0:
            return {'label': 'NEUTRAL', 'score': 0.0, 'confidence': 'low'}

        if pos_score > neg_score and pos_score > neu_score:
            score = min(0.6 + (pos_score / total_score) * 0.4, 0.99)
            return {'label': 'POSITIVE', 'score': score, 'confidence': 'medium'}
        elif neg_score > pos_score and neg_score > neu_score:
            score = min(0.6 + (neg_score / total_score) * 0.4, 0.99)
            return {'label': 'NEGATIVE', 'score': -score, 'confidence': 'medium'}
        else:
            return {'label': 'NEUTRAL', 'score': 0.0, 'confidence': 'medium'}

    def intelligent_summarization(self, text: str, max_length: int = 100) -> Dict:
        """Simple extractive summarization"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        if len(sentences) <= 1:
            return {'summary': text, 'method': 'original', 'compression_ratio': 1.0}

        # Score sentences by length and position
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            # Prefer longer sentences and those at beginning/end
            length_score = min(len(sentence.split()) / 10, 1.0)
            position_score = 1.0 if i in [0, len(sentences)-1] else 0.5
            total_score = length_score + position_score
            scored_sentences.append((sentence, total_score))

        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        summary_sentences = []
        current_length = 0

        for sentence, score in scored_sentences:
            if current_length + len(sentence) <= max_length:
                summary_sentences.append(sentence)
                current_length += len(sentence)

            if len(summary_sentences) >= 2:  # Limit to 2 sentences
                break

        summary = '. '.join(summary_sentences) + '.'
        compression_ratio = len(summary) / len(text) if len(text) > 0 else 1.0

        return {
            'summary': summary,
            'method': 'extractive',
            'compression_ratio': compression_ratio
        }

    def calculate_importance_score(self, text: str, all_comments: List[str]) -> Dict:
        """Calculate importance score based on keywords and length."""
        score = 0.0
        is_important = False
        
        # Check for urgent/risk keywords
        urgent_words = ['urgent', 'risk', 'danger', 'fail', 'error', 'critical', 'warning', 'help', 'safety']
        if any(w in text.lower() for w in urgent_words):
            score += 0.5
            is_important = True
            
        # Check length (very long comments might be detailed feedback)
        if len(text.split()) > 50:
            score += 0.3
            
        return {'score': score, 'is_important': is_important}

    def analyze_batch(self, comments: List[str]) -> Dict:
        """Analyze a batch of comments and return sentiments and summaries"""
        
        if not comments:
            return {
                'sentences': [],
                'sentiments': [],
                'important_rare': []
            }
        
        all_sentiments = []
        summaries = []
        important_comments = []
        
        for comment in comments:
            if not comment or not str(comment).strip():
                continue
                
            # Preprocess
            preprocessed = self.advanced_preprocess(str(comment))
            cleaned = preprocessed['cleaned']
            
            # Sentiment analysis
            sentiment = self.sentiment_analysis(cleaned)
            sentiment_score = sentiment.get('score', 0.0)
            all_sentiments.append(sentiment_score)
            
            # Summarization
            summary = self.intelligent_summarization(cleaned)
            summaries.append(summary.get('summary', cleaned))
            
            # Check importance
            importance = self.calculate_importance_score(cleaned, comments)
            if importance['is_important']:
                important_comments.append(comment)
        
        return {
            'sentences': summaries,
            'sentiments': all_sentiments,
            'important_rare': important_comments[:10]  # Top 10
        }


def Analysis(file_upload):
    """
    Main analysis function for API endpoint
    
    Args:
        file_upload: FastAPI UploadFile object
        
    Returns:
        Tuple of (sentences, sentiments, important_comments)
    """
    try:
        # Read uploaded file
        filename = file_upload.filename
        file_ext = os.path.splitext(filename)[1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            content = file_upload.file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Load data
        try:
            if file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(tmp_path)
            else:
                try:
                    df = pd.read_csv(tmp_path)
                except UnicodeDecodeError:
                    df = pd.read_csv(tmp_path, encoding='latin1')
        except Exception as e:
            print(f"Error loading dataframe: {e}")
            os.remove(tmp_path)
            return [], [], []
        
        # Find comment column (try common names)
        comment_col = None
        possible_names = ['comments', 'comment', 'text', 'message', 'content', 'body', 'feedback', 'review', 'comment_text']
        
        for col in df.columns:
            if col.lower() in possible_names:
                comment_col = col
                break
        
        if not comment_col:
            comment_col = df.columns[0]  # Use first column
        
        comments = df[comment_col].dropna().astype(str).tolist()
        
        # Perform analysis
        analyzer = SentimentSummarization()
        results = analyzer.analyze_batch(comments)
        
        os.remove(tmp_path)
        
        return results['sentences'], results['sentiments'], results['important_rare']
        
    except Exception as e:
        print(f"Analysis error: {e}")
        return [], [], []