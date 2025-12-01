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

def summarize_text(text: str, sentences_count: int = 3, language: str = "english", model_type: str = "lexrank", api_key: str = None) -> str:
    if model_type == "gemini":
        return summarize_with_gemini(text, sentences_count, api_key)
    
    # Detect language for LexRank
    detected_lang = detect_language(text)
    
    parser = PlaintextParser.from_string(text, Tokenizer(detected_lang))
    stemmer = Stemmer(detected_lang)
    summarizer = LexRankSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(detected_lang)

    summary = summarizer(parser.document, sentences_count)
    
    return " ".join([str(sentence) for sentence in summary])
