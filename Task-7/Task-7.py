import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

df = pd.read_csv("D:/Work/ML Tasks/spam_Emails_data.csv")   # change path if needed

df['text'] = df['text'].fillna("")
X = df['text']
y = df['label']

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Test with custom sample email
sample = ["Congrats! You won a free lottery. Claim now"]
sample_tfidf = vectorizer.transform(sample)
print("Sample email is :",sample)
print("Prediction:", model.predict(sample_tfidf))
