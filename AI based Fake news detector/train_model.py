import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

df = pd.read_csv('news.csv')
df = df[['text', 'label']]
df['label'] = df['label'].map({'REAL': 1, 'FAKE': 0})

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
model = LogisticRegression()
model.fit(X_train_vec, y_train)

os.makedirs("model", exist_ok=True)
pickle.dump(model, open('model/classifier.pkl', 'wb'))
pickle.dump(vectorizer, open('model/vectorizer.pkl', 'wb'))
print("✅ Model trained and saved.")