import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import re
import warnings
import os
warnings.filterwarnings('ignore')

class EnhancedSentimentCommentAnalyzer:
    

    def __init__(self, 
                 sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
                 summarization_model: str = "sshleifer/distilbart-cnn-6-6"):
        
        self.sentiment_model_name = sentiment_model
        self.summarization_model_name = summarization_model
        self.models_loaded = False

        # Enhanced filler words and patterns
        self.filler_patterns = {
            'fillers': r'\b(um|uh|like|you know|actually|basically|literally|definitely|obviously|really|very|quite|rather|somewhat|kind of|sort of)\b',
            'repetitions': r'\b(\w+)\s+\1\b',  # repeated words
            'excessive_punctuation': r'[!?]{2,}',
            'urls': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'emails': r'\S+@\S+',
            'phones': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        }

    def load_models(self):
        
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

            print("Loading sentiment analysis model...")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.sentiment_model_name,
                tokenizer=self.sentiment_model_name,
                return_all_scores=True,
                device=-1  # Use CPU
            )

            
            self.summarization_pipeline = pipeline(
                "summarization",
                model=self.summarization_model_name,
                tokenizer=self.summarization_model_name,
                max_length=100,
                min_length=20,
                do_sample=False,
                device=-1  # Use CPU
            )

            self.models_loaded = True
            print("All models loaded successfully!")
            return True

        except ImportError:
            print("Transformers library not found. Install with: pip install transformers torch")
            return False
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Falling back to lightweight mock implementation...")
            return False

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

    def enhanced_sentiment_analysis(self, text: str) -> Dict:
       
        if self.models_loaded and hasattr(self, 'sentiment_pipeline'):
            try:
                results = self.sentiment_pipeline(text)
                # Get the top prediction
                top_result = max(results[0], key=lambda x: x['score'])

                return {
                    'label': top_result['label'],
                    'score': top_result['score'],
                    'confidence': 'high' if top_result['score'] > 0.8 else 'medium' if top_result['score'] > 0.6 else 'low',
                    'all_scores': results[0]
                }
            except Exception as e:
                print(f"Sentiment analysis error: {e}")

        # Fallback to enhanced mock implementation
        return self._mock_sentiment_enhanced(text)

    def _mock_sentiment_enhanced(self, text: str) -> Dict:
        """Enhanced mock sentiment analysis"""
        positive_indicators = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                              'love', 'perfect', 'outstanding', 'brilliant', 'superb', 'awesome']
        negative_indicators = ['bad', 'terrible', 'awful', 'hate', 'horrible', 'worst', 
                              'disappointing', 'poor', 'unsatisfactory', 'failed', 'broken']
        neutral_indicators = ['okay', 'average', 'normal', 'fine', 'acceptable', 'decent']

        text_lower = text.lower()

        pos_score = sum(2 if word in text_lower else 0 for word in positive_indicators)
        neg_score = sum(2 if word in text_lower else 0 for word in negative_indicators)
        neu_score = sum(1 if word in text_lower else 0 for word in neutral_indicators)

        total_score = pos_score + neg_score + neu_score

        if total_score == 0:
            return {'label': 'NEUTRAL', 'score': 0.5, 'confidence': 'low', 'all_scores': []}

        if pos_score > neg_score and pos_score > neu_score:
            score = min(0.6 + (pos_score / total_score) * 0.4, 0.99)
            return {'label': 'POSITIVE', 'score': score, 'confidence': 'medium', 'all_scores': []}
        elif neg_score > pos_score and neg_score > neu_score:
            score = min(0.6 + (neg_score / total_score) * 0.4, 0.99)
            return {'label': 'NEGATIVE', 'score': score, 'confidence': 'medium', 'all_scores': []}
        else:
            return {'label': 'NEUTRAL', 'score': 0.5, 'confidence': 'medium', 'all_scores': []}

    def intelligent_summarization(self, text: str, max_length: int = 100) -> Dict:
        """Intelligent summarization with fallback options"""
        if len(text.strip()) < 20:
            return {'summary': text, 'method': 'original', 'compression_ratio': 1.0}

        if self.models_loaded and hasattr(self, 'summarization_pipeline'):
            try:
                result = self.summarization_pipeline(
                    text, 
                    max_length=min(max_length, len(text.split()) // 2),
                    min_length=max(10, len(text.split()) // 4)
                )
                summary = result[0]['summary_text']
                compression_ratio = len(summary) / len(text)

                return {
                    'summary': summary,
                    'method': 'transformer',
                    'compression_ratio': compression_ratio
                }
            except Exception as e:
                print(f"Summarization error: {e}")

        # Extractive summarization fallback
        return self._extractive_summary(text, max_length)

    def _extractive_summary(self, text: str, max_length: int) -> Dict:
        """Simple extractive summarization"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        if len(sentences) <= 2:
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
        compression_ratio = len(summary) / len(text)

        return {
            'summary': summary,
            'method': 'extractive',
            'compression_ratio': compression_ratio
        }

    def calculate_importance_score(self, comment: str, all_comments: List[str]) -> Dict:
        """Calculate comprehensive importance score"""
        scores = {}

        # Length score (not too short, not too long)
        word_count = len(comment.split())
        if 5 <= word_count <= 50:
            scores['length'] = min(word_count / 20, 1.0)
        else:
            scores['length'] = 0.3

        # Uniqueness score (TF-IDF like)
        comment_words = set(word.lower() for word in comment.split() if len(word) > 3)
        if comment_words:
            uniqueness_scores = []
            for word in comment_words:
                word_freq = sum(1 for c in all_comments if word in c.lower())
                idf_score = len(all_comments) / (word_freq + 1)
                uniqueness_scores.append(idf_score)
            scores['uniqueness'] = np.mean(uniqueness_scores)
        else:
            scores['uniqueness'] = 0.0

        # Sentiment extremity score
        sentiment_result = self.enhanced_sentiment_analysis(comment)
        scores['sentiment_extremity'] = abs(sentiment_result['score'] - 0.5) * 2

        # Overall importance score
        weights = {'length': 0.2, 'uniqueness': 0.5, 'sentiment_extremity': 0.3}
        overall_score = sum(scores[key] * weights[key] for key in scores)

        return {
            'overall_score': overall_score,
            'component_scores': scores,
            'is_important': overall_score > 0.6
        }

    def comprehensive_analysis(self, 
                             file_path: str,
                             comment_column: str = 'comments',
                             chunk_size: int = 10,
                             importance_threshold: float = 0.6) -> Dict:
        
        try:
            # Load data
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                raise ValueError("Unsupported file format. Use .xlsx, .xls, or .csv")

            if comment_column not in df.columns:
                raise ValueError(f"Column '{comment_column}' not found. Available columns: {list(df.columns)}")

            # Extract and clean comments
            raw_comments = df[comment_column].dropna().astype(str).tolist()

            if not raw_comments:
                return {"error": "No comments found in the specified column"}

            print(f"🔄 Processing {len(raw_comments)} comments in chunks of {chunk_size}...")

            # Preprocess all comments
            preprocessed_data = [self.advanced_preprocess(comment) for comment in raw_comments]
            processed_comments = [item['anonymized'] for item in preprocessed_data]

            # Create chunks
            chunks = [processed_comments[i:i + chunk_size] 
                     for i in range(0, len(processed_comments), chunk_size)]

            # Initialize comprehensive results
            results = {
                'metadata': {
                    'total_comments': len(raw_comments),
                    'total_chunks': len(chunks),
                    'chunk_size': chunk_size,
                    'models_used': 'HuggingFace Transformers' if self.models_loaded else 'Mock Implementation',
                    'processing_timestamp': pd.Timestamp.now().isoformat()
                },
                'preprocessing_stats': {
                    'avg_original_length': np.mean([len(item['original']) for item in preprocessed_data]),
                    'avg_cleaned_length': np.mean([len(item['cleaned']) for item in preprocessed_data]),
                    'compression_ratio': np.mean([len(item['cleaned'])/max(len(item['original']), 1) 
                                                for item in preprocessed_data])
                },
                'chunk_analysis': [],
                'sentiment_distribution': {},
                'important_comments': [],
                'summary_statistics': {}
            }

            # Analyze each chunk
            all_sentiments = []
            all_importance_scores = []

            for chunk_idx, chunk in enumerate(chunks):
                chunk_sentiments = []
                chunk_summaries = []
                chunk_importance = []

                for comment_idx, comment in enumerate(chunk):
                    if comment.strip():
                        # Sentiment analysis
                        sentiment = self.enhanced_sentiment_analysis(comment)
                        chunk_sentiments.append(sentiment)
                        all_sentiments.append(sentiment)

                        # Summarization
                        summary_result = self.intelligent_summarization(comment)
                        chunk_summaries.append(summary_result)

                        # Importance scoring
                        global_idx = chunk_idx * chunk_size + comment_idx
                        if global_idx < len(raw_comments):
                            importance = self.calculate_importance_score(comment, processed_comments)
                            importance['global_index'] = global_idx
                            importance['original_comment'] = raw_comments[global_idx]
                            chunk_importance.append(importance)
                            all_importance_scores.append(importance)

                # Chunk results
                chunk_result = {
                    'chunk_index': chunk_idx,
                    'comments_count': len(chunk),
                    'avg_sentiment_score': np.mean([s['score'] for s in chunk_sentiments]) if chunk_sentiments else 0,
                    'sentiment_distribution': self._get_sentiment_distribution(chunk_sentiments),
                    'summaries': chunk_summaries,
                    'important_in_chunk': [imp for imp in chunk_importance if imp['is_important']]
                }

                results['chunk_analysis'].append(chunk_result)

            # Overall sentiment distribution
            results['sentiment_distribution'] = self._get_sentiment_distribution(all_sentiments)

            # Identify most important comments
            important_comments = [score for score in all_importance_scores 
                                if score['overall_score'] >= importance_threshold]
            important_comments.sort(key=lambda x: x['overall_score'], reverse=True)

            results['important_comments'] = important_comments[:10]  # Top 10
            results['important_comment_indices'] = [c['global_index'] for c in important_comments] if important_comments else None

            # Summary statistics
            results['summary_statistics'] = {
                'avg_sentiment_score': np.mean([s['score'] for s in all_sentiments]),
                'sentiment_confidence_distribution': self._get_confidence_distribution(all_sentiments),
                'total_important_comments': len(important_comments),
                'importance_score_stats': {
                    'mean': np.mean([s['overall_score'] for s in all_importance_scores]),
                    'std': np.std([s['overall_score'] for s in all_importance_scores]),
                    'min': np.min([s['overall_score'] for s in all_importance_scores]),
                    'max': np.max([s['overall_score'] for s in all_importance_scores])
                }
            }

            print("Analysis completed successfully!")
            return results

        except Exception as e:
            print(f"Analysis failed: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}

    def _get_sentiment_distribution(self, sentiments: List[Dict]) -> Dict:
        """Calculate sentiment distribution"""
        if not sentiments:
            return {}

        labels = [s['label'] for s in sentiments]
        unique_labels, counts = np.unique(labels, return_counts=True)
        total = len(labels)

        return {
            label: {'count': int(count), 'percentage': round((count/total) * 100, 1)}
            for label, count in zip(unique_labels, counts)
        }

    def _get_confidence_distribution(self, sentiments: List[Dict]) -> Dict:
        """Calculate confidence level distribution"""
        if not sentiments:
            return {}

        confidences = [s.get('confidence', 'unknown') for s in sentiments]
        unique_conf, counts = np.unique(confidences, return_counts=True)
        total = len(confidences)

        return {
            conf: {'count': int(count), 'percentage': round((count/total) * 100, 1)}
            for conf, count in zip(unique_conf, counts)
        }