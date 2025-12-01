from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import numpy as np
import nltk
import pickle
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split
import re

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

print("Loading CNN/DailyMail dataset...")
# 10,000 articles - optimal for performance
dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:10000]")

print("Extracting features and labels...")

X = []  # Features
y = []  # Labels (1 if sentence in summary, 0 otherwise)

vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2))

# First pass: fit vectorizer on all sentences
all_sentences = []
for article in dataset:
    sentences = sent_tokenize(article['article'])
    all_sentences.extend(sentences)

print(f"Fitting TF-IDF vectorizer on {len(all_sentences)} sentences...")
vectorizer.fit(all_sentences)

# Second pass: extract features with IMPROVED labeling
for idx, article in enumerate(dataset):
    if idx % 1000 == 0:
        print(f"Processing article {idx}/10000...")
    
    article_text = article['article']
    summary_text = article['highlights'].lower()
    
    sentences = sent_tokenize(article_text)
    
    for i, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()
        
        # Feature 1 & 2: TF-IDF scores
        tfidf_vector = vectorizer.transform([sentence]).toarray()
        tfidf_score = np.mean(tfidf_vector) if tfidf_vector.size > 0 else 0
        tfidf_max = np.max(tfidf_vector) if tfidf_vector.size > 0 else 0
        
        # Feature 3: Position with STRONG exponential decay for news articles
        # News articles: first sentences are CRITICAL
        position_score = np.exp(-i / max(len(sentences) * 0.2, 1))
        
        # Feature 4 & 5: Sentence length features
        word_count = len(sentence.split())
        length_score = min(word_count, 50) / 50
        optimal_length = 1 if 15 <= word_count <= 35 else 0
        
        # Feature 6: Contains numbers
        has_numbers = 1 if re.search(r'\d', sentence) else 0
        
        # Feature 7: Proper noun count (IMPORTANT for news)
        proper_nouns = len([w for w in sentence.split() if w and w[0].isupper()])
        proper_noun_score = min(proper_nouns, 10) / 10
        
        # Feature 8: Is in LEAD (first 2 sentences) - CRITICAL for news
        is_lead = 1 if i < 2 else 0
        
        # Feature 9: Contains quotes
        has_quotes = 1 if '"' in sentence or "'" in sentence else 0
        
        # Feature 10: Sentence centrality
        words = set(sentence_lower.split())
        centrality = sum(1 for s in sentences[:min(10, len(sentences))] 
                        if len(words & set(s.lower().split())) > 3) / min(10, len(sentences))
        
        # Feature 11: NEW - Similarity to summary (word overlap)
        summary_words = set(summary_text.split())
        overlap = len(words & summary_words) / max(len(words), 1)
        
        # Feature 12: NEW - Contains key news words
        news_keywords = {'said', 'told', 'according', 'reported', 'announced', 'confirmed'}
        has_news_words = 1 if any(kw in sentence_lower for kw in news_keywords) else 0
        
        features = [
            tfidf_score, tfidf_max, position_score, length_score, optimal_length,
            has_numbers, proper_noun_score, is_lead, has_quotes, centrality,
            overlap, has_news_words
        ]
        
        # IMPROVED LABELING: Use fuzzy matching instead of exact match
        # Label as positive if:
        # 1. Exact match in summary, OR
        # 2. High word overlap with summary (>50%), OR
        # 3. First sentence (news lead is almost always important)
        exact_match = sentence in article['highlights']
        high_overlap = overlap > 0.5
        is_first = i == 0
        
        label = 1 if (exact_match or high_overlap or is_first) else 0
        
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

print(f"Total samples: {len(X)}")
print(f"Positive samples (before SMOTE): {sum(y)}")
print(f"Negative samples (before SMOTE): {len(y) - sum(y)}")

# Apply SMOTE to handle class imbalance
print("Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42, k_neighbors=3)
X_resampled, y_resampled = smote.fit_resample(X, y)

print(f"After SMOTE - Positive: {sum(y_resampled)}, Negative: {len(y_resampled) - sum(y_resampled)}")

print("Training Random Forest classifier...")
clf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    class_weight='balanced',  # Additional balancing
    n_jobs=-1
)
clf.fit(X_resampled, y_resampled)

print("Saving model and vectorizer...")
with open('summarizer_model.pkl', 'wb') as f:
    pickle.dump({'classifier': clf, 'vectorizer': vectorizer}, f)

print("Training complete! Model saved to summarizer_model.pkl")
print(f"Model accuracy on training set: {clf.score(X_resampled, y_resampled):.2f}")
