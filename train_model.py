import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
import os

print("📂 Current Working Directory:", os.getcwd())

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

csv_path = "/Users/yashwanthbanka/Documents/ai/ai-sentiment-classifier/imdb_master.csv"
print("✅ File exists:", os.path.exists(csv_path))
df = pd.read_csv(csv_path, encoding='ISO-8859-1')

print("🧾 Columns in CSV:", df.columns)
print(df.head())

TEXT_COLUMN = 'review'
LABEL_COLUMN = 'label'

def clean_text(text):
    text = re.sub(r"[^\w\s]", "", str(text).lower())
    return " ".join([word for word in text.split() if word not in stop_words])

df['cleaned_text'] = df[TEXT_COLUMN].apply(clean_text)
df['label_mapped'] = df[LABEL_COLUMN].map({'positive': 1, 'negative': 0, 'pos': 1, 'neg': 0})
df = df.dropna(subset=['label_mapped'])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label_mapped']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

def predict_sentiment(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    result = model.predict(vec)[0]
    return "Positive" if result == 1 else "Negative"

sample = "The movie was absolutely fantastic!"
print(f"\n💬 Input: \"{sample}\"")
print("Sentiment:", predict_sentiment(sample))