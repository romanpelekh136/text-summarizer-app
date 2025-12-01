from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import nltk

import google.generativeai as genai
import os
from langdetect import detect
from sumy.utils import get_stop_words

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        # Map langdetect codes to sumy languages if needed
        # Sumy supports: czech, english, french, german, italian, japanese, portuguese, slovak, spanish, ukrainian
        # We can add a simple mapping or default to english
        mapping = {
            'cs': 'czech',
            'en': 'english',
            'fr': 'french',
            'de': 'german',
            'it': 'italian',
            'ja': 'japanese',
            'pt': 'portuguese',
            'sk': 'slovak',
            'es': 'spanish',
            'uk': 'ukrainian'
        }
        return mapping.get(lang, 'english')
    except:
        return 'english'

def summarize_with_gemini(text: str, sentences_count: int, api_key: str) -> str:
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file. Please configure it to use Gemini model.")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-flash-latest')
    
    prompt = f"Summarize the following text in approximately {sentences_count} sentences. Ensure the summary is in the same language as the original text:\n\n{text}"
    
    response = model.generate_content(prompt)
    return response.text

def summarize_with_rf(text: str, sentences_count: int) -> str:
    import pickle
    import re
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Load the trained model
    try:
        with open('summarizer_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
            clf = model_data['classifier']
            vectorizer = model_data['vectorizer']
    except FileNotFoundError:
        raise ValueError("Random Forest model not found. Please train the model first by running: python train_model.py")
    
    # Tokenize sentences
    sentences = nltk.sent_tokenize(text)
    
    if len(sentences) == 0:
        return ""
    
    # Extract features for each sentence
    features = []
    text_lower = text.lower()
    
    for i, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()
        
        # Feature 1 & 2: TF-IDF scores
        tfidf_vector = vectorizer.transform([sentence]).toarray()
        tfidf_score = np.mean(tfidf_vector) if tfidf_vector.size > 0 else 0
        tfidf_max = np.max(tfidf_vector) if tfidf_vector.size > 0 else 0
        
        # Feature 3: Position with strong exponential decay
        position_score = np.exp(-i / max(len(sentences) * 0.2, 1))
        
        # Feature 4 & 5: Sentence length features
        word_count = len(sentence.split())
        length_score = min(word_count, 50) / 50
        optimal_length = 1 if 15 <= word_count <= 35 else 0
        
        # Feature 6: Contains numbers
        has_numbers = 1 if re.search(r'\d', sentence) else 0
        
        # Feature 7: Proper noun count
        proper_nouns = len([w for w in sentence.split() if w and w[0].isupper()])
        proper_noun_score = min(proper_nouns, 10) / 10
        
        # Feature 8: Is in lead (first 2 sentences)
        is_lead = 1 if i < 2 else 0
        
        # Feature 9: Contains quotes
        has_quotes = 1 if '"' in sentence or "'" in sentence else 0
        
        # Feature 10: Sentence centrality
        words = set(sentence_lower.split())
        centrality = sum(1 for s in sentences[:min(10, len(sentences))] 
                        if len(words & set(s.lower().split())) > 3) / min(10, len(sentences))
        
        # Feature 11: Similarity to full text (word overlap)
        text_words = set(text_lower.split())
        overlap = len(words & text_words) / max(len(words), 1)
        
        # Feature 12: Contains key news words
        news_keywords = {'said', 'told', 'according', 'reported', 'announced', 'confirmed'}
        has_news_words = 1 if any(kw in sentence_lower for kw in news_keywords) else 0
        
        features.append([
            tfidf_score, tfidf_max, position_score, length_score, optimal_length,
            has_numbers, proper_noun_score, is_lead, has_quotes, centrality,
            overlap, has_news_words
        ])
    
    # Predict sentence importance
    features = np.array(features)
    probabilities = clf.predict_proba(features)[:, 1]
    
    # Select top N sentences
    top_indices = np.argsort(probabilities)[-sentences_count:]
    top_indices = sorted(top_indices)  # Maintain original order
    
    summary = " ".join([sentences[i] for i in top_indices])
    return summary

def summarize_text(text: str, sentences_count: int = 3, language: str = "english", model_type: str = "lexrank", api_key: str = None) -> str:
    if model_type == "gemini":
        return summarize_with_gemini(text, sentences_count, api_key)
    
    if model_type == "random_forest":
        return summarize_with_rf(text, sentences_count)
    
    # Detect language for LexRank
    detected_lang = detect_language(text)
    
    parser = PlaintextParser.from_string(text, Tokenizer(detected_lang))
    stemmer = Stemmer(detected_lang)
    summarizer = LexRankSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(detected_lang)

    summary = summarizer(parser.document, sentences_count)
    
    return " ".join([str(sentence) for sentence in summary])
