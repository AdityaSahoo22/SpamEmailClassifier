import pandas as pd
import pickle
import string
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stop_words and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)

df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'spam': 1, 'ham': 0})
df = df.drop_duplicates()

print(f"Dataset: {len(df)} rows | Spam: {df['label'].sum()} | Ham: {(df['label']==0).sum()}")

df['transformed'] = df['message'].apply(transform_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['transformed'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# Use more features + ngrams to catch spam patterns better
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# Lower alpha = less smoothing = more sensitive to spam words
model = MultinomialNB(alpha=0.1)
model.fit(X_train_vec, y_train)

preds = model.predict(X_test_vec)
print("\nClassification Report:")
print(classification_report(y_test, preds, target_names=['Ham', 'Spam']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, preds))

# Test with common spam messages
test_msgs = [
    "Congratulations! You've won a FREE iPhone. Click here to claim now!",
    "WINNER!! As a valued network customer you have been selected to receive a 900 prize reward!",
    "Hey, are we still meeting for lunch tomorrow?",
    "Your account has been compromised. Verify now or it will be suspended.",
    "Free entry in 2 a weekly competition to win FA Cup final tickets!"
]
print("\nSample Predictions:")
for msg in test_msgs:
    t = transform_text(msg)
    pred = model.predict(tfidf.transform([t]))[0]
    print(f"  {'SPAM' if pred==1 else 'HAM '} -> {msg[:60]}")

pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(model, open('model.pkl', 'wb'))
print("\nvectorizer.pkl and model.pkl saved successfully.")
