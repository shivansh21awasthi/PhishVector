import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

nltk.download('stopwords', quiet=True)

print("Loading dataset...")
df = pd.read_csv("data/emails.csv")

# 1. Identify columns automatically
possible_text = ['text', 'message', 'body', 'content', 'v2', 'email text']
possible_labels = ['label', 'spam', 'category', 'v1', 'target', 'class']

text_col = None
label_col = None

for col in df.columns:
    if col.lower() in possible_text and text_col is None:
        text_col = col
    if col.lower() in possible_labels and label_col is None:
        label_col = col

if not text_col or not label_col:
    print(f"Columns found: {list(df.columns)}")
    raise ValueError("Could not find text and label columns automatically.")

print(f"Using text column: '{text_col}' and label column: '{label_col}'")

# Drop null values
df = df.dropna(subset=[text_col, label_col])

# 2. Standardize labels to 0 and 1
# 2. Standardize labels to 0 and 1 (FLIPPED for this specific dataset)
label_map = {
    'spam': 0, 'ham': 1,  
    'phishing': 1, 'legitimate': 0, 'safe': 0,
    '1': 1, '0': 0, 1: 1, 0: 0, 1.0: 1, 0.0: 0
}

# Apply mapping or try numeric conversion
df[label_col] = df[label_col].astype(str).str.strip().str.lower().map(label_map)
df[label_col] = pd.to_numeric(df[label_col], errors='coerce')
df = df.dropna(subset=[label_col])
df[label_col] = df[label_col].astype(int)

# 3. Clean text
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    return ' '.join(w for w in text.split() if w not in stop_words)

print("Cleaning text...")
df['cleaned_text'] = df[text_col].apply(clean_text)

# 4. Vectorize
print("Vectorizing text with TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['cleaned_text']).toarray()
y = df[label_col]

# 5. Train / Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training model...")
model = MultinomialNB()
model.fit(X_train, y_train)

# 6. Evaluation
y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred))

# 7. Save artifacts
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/phishing_model.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
print("Saved model and vectorizer into 'models/' folder successfully!")